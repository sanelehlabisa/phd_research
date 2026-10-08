"""Resolve the approved AAD dataset from a local path or Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import kagglehub

AAD_DATASET_NAME = "aad"
AAD_KAGGLE_HANDLE = "sanelehlabisa/abnormal-activities-dataset"
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


def _is_aad_root(path: Path) -> bool:
    """Check that a directory has the expected AAD class folders and videos."""
    if not path.is_dir():
        return False
    folders = {item.name for item in path.iterdir() if item.is_dir()}
    if folders != set(AAD_CLASS_NAMES):
        return False
    return all(
        any(
            video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS
            for video in (path / class_name).rglob("*")
        )
        for class_name in AAD_CLASS_NAMES
    )


def _find_aad_root(path: Path) -> Path | None:
    """Find the dataset root itself or one nested directory below it."""
    candidates = [path]
    if path.is_dir():
        candidates.extend(
            sorted(child for child in path.iterdir() if child.is_dir())
        )
    return next((candidate.resolve() for candidate in candidates if _is_aad_root(candidate)), None)


def resolve_dataset(
    dataset_name: str,
    dataset_dir: str | Path | None = None,
    implementation_root: str | Path | None = None,
) -> Path:
    """Reuse a valid local AAD dataset or download the approved Kaggle source.

    Parameters:
        dataset_name: Registered dataset name; currently `aad`.
        dataset_dir: Optional local dataset path, relative to the implementation.
        implementation_root: Implementation directory for relative paths.

    Returns:
        The validated directory containing AAD's eleven class folders.
    """
    if dataset_name != AAD_DATASET_NAME:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; supported dataset: 'aad'"
        )

    root = (
        Path(implementation_root).expanduser().resolve()
        if implementation_root is not None
        else Path(__file__).resolve().parent.parent
    )
    configured_path = (
        Path(dataset_dir).expanduser()
        if dataset_dir is not None and str(dataset_dir).strip()
        else Path("datasets/abnormal-activities-dataset/abnormal-activities-dataset")
    )
    local_path = configured_path if configured_path.is_absolute() else root / configured_path

    if local_path.exists():
        resolved = _find_aad_root(local_path)
        if resolved is None:
            raise ValueError(
                f"AAD path has an unexpected layout: {local_path}. Expected the "
                "eleven AAD class folders, each containing video files."
            )
        print(f"Using local AAD dataset: {resolved}", flush=True)
        return resolved

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

    resolved = _find_aad_root(downloaded_path)
    if resolved is None:
        raise RuntimeError(
            "Kaggle returned files, but the expected eleven-class AAD video "
            f"layout was not found under {downloaded_path}."
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
