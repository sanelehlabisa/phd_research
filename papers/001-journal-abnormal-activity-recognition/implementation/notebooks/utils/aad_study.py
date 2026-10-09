"""Run the active AAD search and comparison from Colab."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import zipfile

from src.dataset_source import resolve_dataset

PROFILES = (
    ("local smoke", "configs/experiments/aad_local_smoke.json"),
    ("custom search", "configs/experiments/aad_custom_search_colab.json"),
    ("model comparison", "configs/experiments/aad_model_comparison_colab.json"),
)


def _run_command(command: list[str], root: Path, log_path: Path | None = None) -> None:
    """Stream a modular command to the notebook and preserve its output."""
    print("$", " ".join(command), flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    ) as process:
        log_file = log_path.open("w", encoding="utf-8") if log_path else None
        try:
            while chunk := os.read(process.stdout.fileno(), 4096):
                text = chunk.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                if log_file:
                    log_file.write(text)
                    log_file.flush()
            result = process.wait()
        finally:
            if log_file:
                log_file.close()
    if result:
        raise subprocess.CalledProcessError(result, command)


def _run_directories(root: Path) -> set[Path]:
    """Find immediate children created under the standard run categories."""
    found = set()
    for purpose in ("experiments", "studies", "evaluate"):
        directory = root / "runs" / purpose
        if directory.is_dir():
            found.update(path.resolve() for path in directory.iterdir() if path.is_dir())
    return found


def _write_progress(path: Path, stages: list[dict[str, object]]) -> None:
    """Atomically save stage status so an interrupted notebook is diagnosable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps({"stages": stages}, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _make_archive(root: Path, archive_path: Path, progress_path: Path, run_directories: set[Path]) -> Path:
    """Package this study's profiles, logs, checkpoints, metrics and examples."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    files = [root / config for _, config in PROFILES]
    files.extend(
        file_path
        for run_directory in sorted(run_directories)
        for file_path in run_directory.rglob("*")
        if file_path.is_file() and file_path != archive_path
    )
    files.extend(
        path for path in progress_path.parent.rglob("*")
        if path.is_file() and path != archive_path
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in dict.fromkeys(files):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(root))
    return archive_path


def _download_archive(path: Path) -> None:
    """Download on Colab or show the local archive path."""
    try:
        from google.colab import files
    except ImportError:
        print(f"Artifacts saved locally: {path}")
        return
    try:
        files.download(str(path))
    except Exception as error:
        print(f"Automatic download failed; retrieve this archive manually: {path}")
        print(f"Download detail: {error}")


def _resolved_profile(path: Path, dataset_path: Path) -> dict[str, object]:
    """Load a study profile and pin it to the one resolved local dataset."""
    values = json.loads(path.read_text(encoding="utf-8"))
    values["dataset"]["path"] = str(dataset_path)
    return values


def _choose_batch_size(results: list[dict[str, object]]) -> tuple[int, str]:
    """Choose the faster batch only when it leaves a safe CUDA-memory margin."""
    available = {
        int(item["batch_size"]): item
        for item in results
        if item["available"] and item["memory_safe"]
    }
    if 16 not in available:
        raise RuntimeError("batch 16 did not complete with a safe memory margin")
    if 32 in available and float(available[32]["samples_per_second"]) >= 1.1 * float(
        available[16]["samples_per_second"]
    ):
        return 32, "batch 32 was at least 10% faster and retained the required free-memory margin"
    return 16, "batch 32 was unavailable, unsafe, or less than 10% faster"


def _benchmark_batch_sizes(
    implementation_root: Path,
    dataset_path: Path,
    search_values: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    """Measure training throughput for batches 16 and 32 on cached train clips."""
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, Subset

    from src.dataset import AHARDataset, CachedAHARDataset, load_split_subsets
    from src.metrics import train_classifier_steps
    from src.model import CustomConvLSTM
    from src.utils import data_loader_generator, seed_everything, seed_data_loader_worker

    if not torch.cuda.is_available():
        raise RuntimeError("batch-size benchmark requires the Colab CUDA device")

    training = search_values["training"]
    reference = search_values["reference_input"]
    seed = search_values["seeds"][0]
    source_dataset = AHARDataset(
        dataset_path,
        reference["sequence_length"],
        (reference["frame_size"], reference["frame_size"]),
    )
    manifest_value = search_values["dataset"]["split_manifest"]
    manifest_path = (
        Path(manifest_value).expanduser()
        if manifest_value
        else implementation_root / "splits" / f"{dataset_path.name}_seed{seed}.json"
    )
    if not manifest_path.is_absolute():
        manifest_path = implementation_root / manifest_path
    train_set, validation_set, _, split_metadata = load_split_subsets(
        source_dataset,
        manifest_path,
        seed=seed,
    )
    cached_indices = sorted(set(train_set.indices + validation_set.indices))
    dataset = CachedAHARDataset(
        dataset_path,
        reference["sequence_length"],
        (reference["frame_size"], reference["frame_size"]),
        cache_indices=cached_indices,
    )
    cache_bytes = sum(video.numel() * video.element_size() for video, _ in dataset._cache.values())
    train_set, _, _, _ = load_split_subsets(
        dataset,
        manifest_path,
        seed=seed,
    )
    search_values["dataset"]["split_manifest"] = str(manifest_path.resolve())
    candidate = max(
        search_values["custom_candidates"],
        key=lambda item: sum(layer[0] ** 2 for layer in item["convlstm_layers"]),
    )
    layers = [(layer[0], tuple(layer[1])) for layer in candidate["convlstm_layers"]]
    device = torch.device("cuda")
    results: list[dict[str, object]] = []
    steps = 8
    for batch_size in (16, 32):
        subset = Subset(train_set, list(range(min(len(train_set), batch_size * steps))))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=training["num_workers"],
            pin_memory=training["pin_memory"],
            worker_init_fn=seed_data_loader_worker,
            generator=data_loader_generator(seed),
        )
        try:
            seed_everything(seed)
            model = CustomConvLSTM(len(dataset.class_names), layers=layers).to(device)
            optimizer = optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=0.0)
            criterion = nn.CrossEntropyLoss()
            warmup_loader = DataLoader(
                Subset(train_set, list(range(batch_size))),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            train_classifier_steps(model, warmup_loader, criterion, optimizer, device, 1)
            seed_everything(seed)
            model = CustomConvLSTM(len(dataset.class_names), layers=layers).to(device)
            optimizer = optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=0.0)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = perf_counter()
            train_classifier_steps(model, loader, criterion, optimizer, device, steps)
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - started
            samples = min(len(train_set), batch_size * steps)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            memory_safe = free_bytes >= max(1024**3, int(total_bytes * 0.10))
            results.append({
                "batch_size": batch_size,
                "available": True,
                "memory_safe": memory_safe,
                "steps": steps,
                "samples": samples,
                "elapsed_seconds": round(elapsed, 3),
                "samples_per_second": round(samples / elapsed, 3),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                "free_cuda_memory_after_trial_bytes": free_bytes,
                "total_cuda_memory_bytes": total_bytes,
            })
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": batch_size, "available": False, "memory_safe": False, "reason": "CUDA out of memory"})
            torch.cuda.empty_cache()
        finally:
            if "model" in locals():
                del model
            if "optimizer" in locals():
                del optimizer
            torch.cuda.empty_cache()

    selected_batch_size, reason = _choose_batch_size(results)
    report = {
        "purpose": "throughput only; not model-selection evidence",
        "dataset_dir": str(dataset_path),
        "split_manifest": str(manifest_path.resolve()),
        "split_manifest_hash": split_metadata["manifest_hash"],
        "seed": seed,
        "model": candidate["name"],
        "input": {"sequence_length": reference["sequence_length"], "height": reference["frame_size"], "width": reference["frame_size"]},
        "cache_memory_bytes": cache_bytes,
        "cache_memory_gib": round(cache_bytes / (1024**3), 3),
        "cached_partition_indices": {"train": len(train_set), "validation": len(validation_set), "test": 0},
        "results": results,
        "selected_batch_size": selected_batch_size,
        "selection_reason": reason,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Batch throughput: 16={next(item.get('samples_per_second', 0) for item in results if item['batch_size'] == 16):.2f} samples/s; "
        f"32={next(item.get('samples_per_second', 0) for item in results if item['batch_size'] == 32):.2f} samples/s; "
        f"selected={selected_batch_size} ({reason})",
        flush=True,
    )
    return report


def run_aad_study(root: str | Path, run_full_study: bool = True) -> Path | None:
    """Run only the validation-driven AAD custom search and archive its results."""
    implementation_root = Path(root).expanduser().resolve()
    python = sys.executable
    local_config = implementation_root / PROFILES[0][1]
    search_config = implementation_root / PROFILES[1][1]
    comparison_template = implementation_root / PROFILES[2][1]

    print("Checking local smoke profile", flush=True)
    _run_command([python, "-m", "src.experiments", "--config", str(local_config), "--list-plan"], implementation_root)
    print("Checking custom search profile", flush=True)
    _run_command([python, "-m", "src.experiments", "--config", str(search_config), "--list-plan"], implementation_root)
    if not run_full_study:
        template = json.loads(comparison_template.read_text(encoding="utf-8"))
        assert template["models"] == ["custom_selected", "paper_convlstm_published", "r3d_18", "mc3_18", "swin3d_t", "swin3d_s"]
        print("Smoke check complete; no dataset download or training was started.")
        return None

    search_values = json.loads(search_config.read_text(encoding="utf-8"))
    dataset = resolve_dataset(search_values["dataset"]["name"], search_values["dataset"]["path"], implementation_root)
    dataset = Path(dataset).resolve()
    study_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    study_root = implementation_root / "runs" / "notebook_studies" / study_id
    progress_path = study_root / "progress.json"
    archive_path = study_root / "artifacts.zip"
    stage_records: list[dict[str, object]] = []
    created_runs: set[Path] = set()
    _write_progress(progress_path, stage_records)

    try:
        benchmark_record = {"stage": "batch_throughput_check", "status": "running"}
        stage_records.append(benchmark_record)
        _write_progress(progress_path, stage_records)
        resolved_search = _resolved_profile(search_config, dataset)
        benchmark = _benchmark_batch_sizes(
            implementation_root,
            dataset,
            resolved_search,
            study_root / "batch_benchmark.json",
        )
        split_manifest = Path(benchmark["split_manifest"]).resolve()
        archived_manifest = study_root / "split_manifest.json"
        archived_manifest.write_bytes(split_manifest.read_bytes())
        resolved_search["dataset"]["split_manifest"] = str(archived_manifest)
        resolved_search["training"]["batch_size"] = benchmark["selected_batch_size"]
        benchmark_record.update({
            "status": "complete",
            "selected_batch_size": benchmark["selected_batch_size"],
            "split_manifest": str(archived_manifest),
        })
        _write_progress(progress_path, stage_records)

        before = _run_directories(implementation_root)
        search_record = {"stage": "custom_search", "status": "running", "run_directories": []}
        stage_records.append(search_record)
        _write_progress(progress_path, stage_records)
        resolved_search_path = study_root / "resolved_custom_search.json"
        resolved_search_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_search_path.write_text(json.dumps(resolved_search, indent=2) + "\n", encoding="utf-8")
        try:
            _run_command(
                [python, "-m", "src.experiments", "--config", str(resolved_search_path)],
                implementation_root,
                study_root / "logs" / "custom_search.log",
            )
            search_record["status"] = "complete"
        except KeyboardInterrupt:
            search_record["status"] = "interrupted"
            raise
        except Exception as error:
            search_record["status"] = "failed"
            search_record["error"] = str(error)
            raise
        finally:
            found = _run_directories(implementation_root) - before
            created_runs.update(found)
            search_record["run_directories"] = [str(path.relative_to(implementation_root)) for path in sorted(found)]
            _write_progress(progress_path, stage_records)

        search_runs = [path for path in created_runs if path.parent.name == "studies" and (path / "selected_config.json").is_file()]
        if not search_runs:
            raise FileNotFoundError("custom search finished without selected_config.json")
        selected_path = max(search_runs, key=lambda path: path.stat().st_mtime) / "selected_config.json"
        search_record["selected_config"] = str(selected_path)
        print(f"Validation-selected config for later comparison: {selected_path}", flush=True)
        print("Custom search complete; model-family comparison is a separate run.", flush=True)
    except KeyboardInterrupt:
        print("Run interrupted. Packaging artifacts saved so far.", flush=True)
    except Exception as error:
        print(f"Study stopped: {error}. Packaging artifacts saved so far.", flush=True)
        search_record = next((item for item in stage_records if item["stage"] == "custom_search"), None)
        if search_record is None:
            stage_records[-1]["error"] = str(error)
    finally:
        created_runs.add(study_root)
        _write_progress(progress_path, stage_records)
        archive = _make_archive(implementation_root, archive_path, progress_path, created_runs)
        print(f"Study archive: {archive}", flush=True)
        _download_archive(archive)

    incomplete = [record for record in stage_records if record["status"] != "complete"]
    if incomplete:
        raise RuntimeError("Search did not complete. Review the downloaded progress and logs before resuming.")
    return archive_path


def run_saved_comparison(root: str | Path, selected_config_path: str | Path) -> Path:
    """Compare model families using an already frozen validation selection."""
    implementation_root = Path(root).expanduser().resolve()
    selected_path = Path(selected_config_path).expanduser().resolve()
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("selection_partition") != "validation" or selected.get("test_access") != "locked":
        raise ValueError("selected config must come from a validation-only custom search")
    candidate = selected["candidate"]
    dataset_path = resolve_dataset("aad", selected["dataset_dir"], implementation_root)
    dataset_path = Path(dataset_path).resolve()
    comparison_template = implementation_root / PROFILES[2][1]
    resolved_comparison = _resolved_profile(comparison_template, dataset_path)
    resolved_comparison["selected_config"] = str(selected_path)
    resolved_comparison["dataset"]["split_manifest"] = selected["split"]["manifest_path"]
    resolved_comparison["custom_candidates"] = [{
        "name": "custom_selected",
        "research_question": "Architecture selected using validation evidence by the custom search.",
        "convlstm_layers": candidate["convlstm_layers"],
        "hidden_classifier_width": candidate.get("hidden_classifier_width"),
    }]

    study_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    study_root = implementation_root / "runs" / "notebook_studies" / f"comparison_{study_id}"
    progress_path = study_root / "progress.json"
    archive_path = study_root / "artifacts.zip"
    created_runs = {selected_path.parent}
    stage_records: list[dict[str, object]] = []
    comparison_path = study_root / "resolved_model_comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(resolved_comparison, indent=2) + "\n", encoding="utf-8")
    _write_progress(progress_path, stage_records)

    try:
        _run_command(
            [sys.executable, "-m", "src.experiments", "--config", str(comparison_path), "--list-plan"],
            implementation_root,
            study_root / "logs" / "comparison_plan.log",
        )
        before = _run_directories(implementation_root)
        record = {"stage": "model_comparison_and_frozen_test", "status": "running", "run_directories": [], "selected_config": str(selected_path)}
        stage_records.append(record)
        _write_progress(progress_path, stage_records)
        _run_command(
            [sys.executable, "-m", "src.experiments", "--config", str(comparison_path)],
            implementation_root,
            study_root / "logs" / "model_comparison_and_test.log",
        )
        record["status"] = "complete"
        record["run_directories"] = [str(path.relative_to(implementation_root)) for path in sorted(_run_directories(implementation_root) - before)]
    except Exception as error:
        if stage_records:
            stage_records[-1].update({"status": "failed", "error": str(error)})
        else:
            stage_records.append({"stage": "model_comparison_and_frozen_test", "status": "failed", "error": str(error)})
        raise
    finally:
        created_runs.add(study_root)
        _write_progress(progress_path, stage_records)
        archive = _make_archive(implementation_root, archive_path, progress_path, created_runs)
        print(f"Comparison archive: {archive}", flush=True)
        _download_archive(archive)
    return archive_path
