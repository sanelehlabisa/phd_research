"""Tests for dataset-neutral download resolution and training subsets."""

from pathlib import Path

import pytest

from notebooks.utils.diagnostics import balanced_tiny_indices
from src.dataset_source import resolve_downloaded_dataset_root


class DatasetStub:
    class_names = ["calm", "event"]

    def __init__(self, root: Path) -> None:
        self.samples = [
            (root / "calm" / "b.mp4", 0),
            (root / "event" / "b.mp4", 1),
            (root / "calm" / "a.mp4", 0),
            (root / "event" / "a.mp4", 1),
        ]


def test_download_root_finds_nested_class_folder_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "version" / "dataset"
    for class_name in ("calm", "event"):
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "clip.mp4").touch()

    assert resolve_downloaded_dataset_root(tmp_path) == dataset_root.resolve()


def test_balanced_tiny_indices_are_stable_and_training_only(tmp_path: Path) -> None:
    dataset = DatasetStub(tmp_path)
    assert balanced_tiny_indices(dataset, [0, 1, 2, 3], 1) == [2, 3]


def test_balanced_tiny_indices_reject_missing_class(tmp_path: Path) -> None:
    dataset = DatasetStub(tmp_path)
    with pytest.raises(ValueError, match="event"):
        balanced_tiny_indices(dataset, [0, 2], 1)
