"""Run the configured AAD search and comparison from Colab."""

from __future__ import annotations

from datetime import datetime
import json
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
ACTIVE_STAGES = PROFILES[1:]


def _load_dataset_config(path: Path) -> tuple[str, str]:
    """Read the dataset name and path from either supported profile shape."""
    values = json.loads(path.read_text(encoding="utf-8"))
    dataset = values.get("dataset")
    if isinstance(dataset, dict):
        return dataset["name"], dataset["path"]
    return values["dataset_name"], values["dataset_dir"]


def _run_command(command: list[str], root: Path) -> None:
    """Print and execute a modular command without imposing a time limit."""
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _run_directories(root: Path) -> set[Path]:
    """Return current experiment-run directories."""
    directory = root / "runs" / "experiments"
    if not directory.is_dir():
        return set()
    return {path.resolve() for path in directory.iterdir() if path.is_dir()}


def _write_progress(path: Path, stages: list[dict[str, object]]) -> None:
    """Save completed-stage information so it survives notebook disconnects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"stages": stages}, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _make_archive(
    root: Path,
    archive_path: Path,
    progress_path: Path,
    run_directories: set[Path],
) -> Path:
    """Zip the current study's configs, run evidence, and stage manifest."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    files = [root / config for _, config in PROFILES]
    files.extend(
        file_path
        for run_directory in sorted(run_directories)
        for file_path in run_directory.rglob("*")
        if file_path.is_file()
    )
    files.append(progress_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(root))
    return archive_path


def _download_archive(path: Path) -> None:
    """Download in Colab; print the path in local development."""
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


def run_aad_study(root: str | Path, run_full_study: bool = True) -> Path | None:
    """Validate profiles, optionally run both AAD stages, then package evidence."""
    implementation_root = Path(root).expanduser().resolve()
    python = sys.executable

    for label, config in PROFILES:
        print(f"\nChecking {label} profile", flush=True)
        _run_command(
            [python, "-m", "src.experiments", "--config", config, "--list-plan"],
            implementation_root,
        )

    if not run_full_study:
        print("Smoke check complete; no dataset download or training was started.")
        return None

    dataset_name, dataset_path = _load_dataset_config(
        implementation_root / ACTIVE_STAGES[0][1]
    )
    resolve_dataset(dataset_name, dataset_path, implementation_root)

    study_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    study_root = implementation_root / "runs" / "notebook_studies" / study_id
    progress_path = study_root / "progress.json"
    archive_path = study_root / "artifacts.zip"
    stage_records: list[dict[str, object]] = []
    new_run_directories: set[Path] = set()
    _write_progress(progress_path, stage_records)

    try:
        for stage_name, config in ACTIVE_STAGES:
            before = _run_directories(implementation_root)
            record: dict[str, object] = {
                "stage": stage_name,
                "config": config,
                "status": "running",
                "run_directories": [],
            }
            stage_records.append(record)
            _write_progress(progress_path, stage_records)
            try:
                _run_command(
                    [python, "-m", "src.experiments", "--config", config],
                    implementation_root,
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
                new_run_directories.update(found)
                record["run_directories"] = [
                    str(path.relative_to(implementation_root))
                    for path in sorted(found)
                ]
                _write_progress(progress_path, stage_records)
                print(
                    f"Saved {stage_name} stage status and found "
                    f"{len(found)} run folder(s).",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Run interrupted. Packaging artifacts saved so far.", flush=True)
    except Exception as error:
        print(f"Study stopped: {error}. Packaging artifacts saved so far.", flush=True)
    finally:
        _write_progress(progress_path, stage_records)
        archive = _make_archive(
            implementation_root,
            archive_path,
            progress_path,
            new_run_directories,
        )
        print(f"Study archive: {archive}", flush=True)
        _download_archive(archive)

    incomplete = [record for record in stage_records if record["status"] != "complete"]
    if incomplete:
        raise RuntimeError(
            "One or more stages did not complete. Review the downloaded "
            "progress.json and resume only after checking the saved runs."
        )
    return archive_path
