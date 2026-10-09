"""Run the active AAD search and comparison from Colab."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
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


def run_aad_study(root: str | Path, run_full_study: bool = True) -> Path | None:
    """Check profiles, run search then selected-model comparison, and download results."""
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
        for stage_name, profile_path in (("custom_search", search_config),):
            before = _run_directories(implementation_root)
            record = {"stage": stage_name, "status": "running", "run_directories": []}
            stage_records.append(record)
            _write_progress(progress_path, stage_records)
            resolved_search = _resolved_profile(profile_path, dataset)
            resolved_search_path = study_root / "resolved_custom_search.json"
            resolved_search_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_search_path.write_text(json.dumps(resolved_search, indent=2) + "\n", encoding="utf-8")
            try:
                _run_command(
                    [python, "-m", "src.experiments", "--config", str(resolved_search_path)],
                    implementation_root,
                    study_root / "logs" / "custom_search.log",
                )
                record["status"] = "complete"
            except KeyboardInterrupt:
                record["status"] = "interrupted"
                raise
            except Exception as error:
                record["status"] = "failed"
                record["error"] = str(error)
                raise
            finally:
                found = _run_directories(implementation_root) - before
                created_runs.update(found)
                record["run_directories"] = [str(path.relative_to(implementation_root)) for path in sorted(found)]
                _write_progress(progress_path, stage_records)

        search_runs = [path for path in created_runs if path.parent.name == "studies" and (path / "selected_config.json").is_file()]
        if not search_runs:
            raise FileNotFoundError("custom search finished without selected_config.json")
        selected_path = max(search_runs, key=lambda path: path.stat().st_mtime) / "selected_config.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        candidate = selected["candidate"]
        resolved_comparison = _resolved_profile(comparison_template, dataset)
        resolved_comparison["selected_config"] = str(selected_path.resolve())
        resolved_comparison["custom_candidates"] = [{
            "name": "custom_selected",
            "research_question": "Architecture selected using validation evidence by this study's custom search.",
            "convlstm_layers": candidate["convlstm_layers"],
            "hidden_classifier_width": candidate.get("hidden_classifier_width"),
        }]
        resolved_comparison_path = study_root / "resolved_model_comparison.json"
        resolved_comparison_path.write_text(json.dumps(resolved_comparison, indent=2) + "\n", encoding="utf-8")
        _run_command(
            [python, "-m", "src.experiments", "--config", str(resolved_comparison_path), "--list-plan"],
            implementation_root,
            study_root / "logs" / "comparison_plan.log",
        )
        before = _run_directories(implementation_root)
        record = {"stage": "model_comparison_and_frozen_test", "status": "running", "run_directories": [], "selected_config": str(selected_path)}
        stage_records.append(record)
        _write_progress(progress_path, stage_records)
        try:
            _run_command(
                [python, "-m", "src.experiments", "--config", str(resolved_comparison_path)],
                implementation_root,
                study_root / "logs" / "model_comparison_and_test.log",
            )
            record["status"] = "complete"
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            raise
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            raise
        finally:
            found = _run_directories(implementation_root) - before
            created_runs.update(found)
            record["run_directories"] = [str(path.relative_to(implementation_root)) for path in sorted(found)]
            _write_progress(progress_path, stage_records)
    except KeyboardInterrupt:
        print("Run interrupted. Packaging artifacts saved so far.", flush=True)
    except Exception as error:
        print(f"Study stopped: {error}. Packaging artifacts saved so far.", flush=True)
    finally:
        created_runs.add(study_root)
        _write_progress(progress_path, stage_records)
        archive = _make_archive(implementation_root, archive_path, progress_path, created_runs)
        print(f"Study archive: {archive}", flush=True)
        _download_archive(archive)

    incomplete = [record for record in stage_records if record["status"] != "complete"]
    if incomplete:
        raise RuntimeError("Study did not complete. Review the downloaded progress and logs before resuming.")
    return archive_path
