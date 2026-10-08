"""Resolve configured video datasets and download public AAD when needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import kagglehub

AAD_DATASET_NAME = "aad"
AAD_KAGGLE_HANDLE = "sanelehlabisa/abnormal-activities-dataset"
# Used by the existing notebook diagnostics; modular runs discover labels from disk.
AAD_CLASS_NAMES = (
    "Begging",
    "Drunkenness",
    "Fight",
    "Harassment",
    "Hijack",
    "Knife Hazard",
    "Normal Videos",
    "Pollution",
    "Property Damage",
    "Robbery",
    "Terrorism",
)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _is_video_dataset_root(path: Path) -> bool:
    """Check for two or more populated class folders with supported videos."""
    if not path.is_dir():
        return False
    class_directories = sorted(
        item
        for item in path.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    )
    if len(class_directories) < 2:
        return False
    return all(
        any(
            video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS
            for video in class_directory.iterdir()
        )
        for class_directory in class_directories
    )


def _find_video_dataset_root(path: Path) -> Path | None:
    """Find a class-folder root itself or one nested directory below it."""
    candidates = [path]
    if path.is_dir():
        candidates.extend(
            sorted(child for child in path.iterdir() if child.is_dir())
        )
    return next(
        (
            candidate.resolve()
            for candidate in candidates
            if _is_video_dataset_root(candidate)
        ),
        None,
    )


def resolve_downloaded_dataset_root(download_root: str | Path) -> Path:
    """Find a downloaded dataset's class-folder root without dataset-specific names."""
    root = Path(download_root).expanduser().resolve()
    pending = [root]
    while pending:
        candidate = pending.pop(0)
        class_directories = sorted(
            item
            for item in candidate.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        )
        populated = [
            directory
            for directory in class_directories
            if any(
                video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS
                for video in directory.iterdir()
            )
        ]
        if len(populated) >= 2:
            return candidate
        pending.extend(directory for directory in class_directories if directory not in populated)
    raise FileNotFoundError(
        f"could not find at least two populated video class folders under {root}"
    )


def resolve_dataset(
    dataset_name: str,
    dataset_dir: str | Path | None = None,
    implementation_root: str | Path | None = None,
) -> Path:
    """Resolve a local class-folder dataset or download public AAD.

    Parameters:
        dataset_name: Dataset label saved with run provenance.
        dataset_dir: Optional local dataset path, relative to the implementation.
        implementation_root: Implementation directory for relative paths.

    Returns:
        The validated directory containing at least two populated class folders.
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")
    dataset_name = dataset_name.strip()

    root = (
        Path(implementation_root).expanduser().resolve()
        if implementation_root is not None
        else Path(__file__).resolve().parent.parent
    )
    if dataset_dir is None or not str(dataset_dir).strip():
        if dataset_name != AAD_DATASET_NAME:
            raise ValueError(
                f"dataset_dir is required for {dataset_name!r}; only 'aad' has "
                "an automatic public download"
            )
        configured_path = Path(
            "datasets/abnormal-activities-dataset/abnormal-activities-dataset"
        )
    else:
        configured_path = Path(dataset_dir).expanduser()
    local_path = (
        configured_path
        if configured_path.is_absolute()
        else root / configured_path
    )

    if local_path.exists():
        resolved = _find_video_dataset_root(local_path)
        if resolved is None:
            raise ValueError(
                f"dataset path has an unexpected layout: {local_path}. Expected "
                "at least two class folders, each containing supported video files."
            )
        print(f"Using local {dataset_name} dataset: {resolved}", flush=True)
        return resolved

    if dataset_name != AAD_DATASET_NAME:
        raise FileNotFoundError(
            f"dataset {dataset_name!r} was not found at {local_path}. Set "
            "dataset_dir in the JSON config to its local class-folder directory; "
            "only 'aad' has an automatic public download."
        )

    print(f"Downloading AAD from Kaggle: {AAD_KAGGLE_HANDLE}", flush=True)
    try:
        downloaded_path = Path(kagglehub.dataset_download(AAD_KAGGLE_HANDLE))
    except Exception as error:
        raise RuntimeError(
            "Could not download the public AAD dataset through KaggleHub. "
            "This download does not require a Kaggle token. Check the internet "
            "connection and that the public dataset is available; Kaggle may "
            "require accepting dataset terms in a browser. "
            f"Details: {error}"
        ) from error

    resolved = _find_video_dataset_root(downloaded_path)
    if resolved is None:
        raise RuntimeError(
            "Kaggle returned files, but a valid AAD video class-folder layout "
            f"was not found under {downloaded_path}."
        )
    print(f"AAD dataset ready: {resolved}", flush=True)
    return resolved


def _read_config(path: Path) -> dict[str, Any]:
    """Read a JSON object containing the dataset name and optional path."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"Dataset config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {path}: {error.msg}") from error
    if not isinstance(values, dict):
        raise SystemExit("Dataset config must contain a JSON object")
    return values


def main(argv: list[str] | None = None) -> None:
    """Resolve the dataset named by a config or explicit command arguments."""
    parser = argparse.ArgumentParser(description="Find or download a video dataset")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-dir")
    arguments = vars(parser.parse_args(argv))
    values = _read_config(arguments["config"]) if arguments["config"] else {}
    dataset_name = arguments["dataset_name"] or values.get("dataset_name", "aad")
    dataset_dir = arguments["dataset_dir"] or values.get("dataset_dir")
    resolved = resolve_dataset(
        dataset_name,
        dataset_dir,
        implementation_root=Path(__file__).resolve().parent.parent,
    )
    print(resolved)


if __name__ == "__main__":
    main()
