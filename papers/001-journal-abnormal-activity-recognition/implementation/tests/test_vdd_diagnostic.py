"""Focused tests for the VDD learnability diagnostic."""

from pathlib import Path

import pytest

from src.vdd_diagnostic import balanced_tiny_indices, resolve_vdd_root
from src.dataset import AHARDataset


class _DatasetStub:
    class_names = ["non_violence", "violence"]

    def __init__(self, root: Path) -> None:
        self.samples = [
            (root / "non_violence" / "b.mp4", 0),
            (root / "violence" / "b.mp4", 1),
            (root / "non_violence" / "a.mp4", 0),
            (root / "violence" / "a.mp4", 1),
        ]


def test_resolve_vdd_root_finds_nested_class_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "version" / "dataset"
    for class_name in ("violence", "non_violence"):
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "clip.mp4").touch()

    assert resolve_vdd_root(tmp_path) == dataset_root.resolve()


def test_balanced_tiny_indices_are_stable_and_training_only(tmp_path: Path) -> None:
    dataset = _DatasetStub(tmp_path)
    assert balanced_tiny_indices(dataset, [0, 1, 2, 3], 1) == [2, 3]


def test_balanced_tiny_indices_reject_missing_class(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="violence"):
        balanced_tiny_indices(_DatasetStub(tmp_path), [0, 2], 1)


def test_dataset_rejects_unknown_accepted_class(tmp_path: Path) -> None:
    for class_name in ("fight", "normal"):
        (tmp_path / class_name).mkdir()

    with pytest.raises(ValueError, match="missing"):
        AHARDataset(tmp_path, accepted_classes=["fight", "missing"])
