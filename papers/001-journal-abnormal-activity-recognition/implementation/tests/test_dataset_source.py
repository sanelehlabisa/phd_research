"""Tests for local-first AAD dataset resolution."""

from pathlib import Path

import pytest

from src import dataset_source


def make_aad_root(root: Path) -> Path:
    """Create the expected AAD class layout with one placeholder clip each."""
    for class_name in dataset_source.AAD_CLASS_NAMES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "clip.mp4").touch()
    return root


def test_local_aad_is_reused_without_downloading(tmp_path, monkeypatch):
    implementation_root = tmp_path / "implementation"
    local_root = make_aad_root(
        implementation_root
        / "datasets/abnormal-activities-dataset/abnormal-activities-dataset"
    )

    def unexpected_download(_handle):
        raise AssertionError("a valid local dataset must not trigger a download")

    monkeypatch.setattr(dataset_source.kagglehub, "dataset_download", unexpected_download)
    assert dataset_source.resolve_dataset("aad", implementation_root=implementation_root) == local_root.resolve()


def test_downloaded_aad_nested_under_cache_is_located(tmp_path, monkeypatch):
    implementation_root = tmp_path / "implementation"
    downloaded_root = make_aad_root(tmp_path / "cache" / "download" / "dataset")
    monkeypatch.setattr(
        dataset_source.kagglehub,
        "dataset_download",
        lambda handle: str(downloaded_root.parent),
    )

    resolved = dataset_source.resolve_dataset(
        "aad", "datasets/not-present", implementation_root
    )

    assert resolved == downloaded_root.resolve()


def test_malformed_local_layout_fails_with_expected_structure(tmp_path):
    malformed = tmp_path / "local"
    malformed.mkdir()
    (malformed / "Fight").mkdir()

    with pytest.raises(ValueError, match="eleven AAD class folders"):
        dataset_source.resolve_dataset("aad", malformed, tmp_path)


def test_unknown_dataset_name_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        dataset_source.resolve_dataset("unknown")


def test_download_failure_does_not_claim_public_aad_needs_token(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_source.kagglehub,
        "dataset_download",
        lambda _handle: (_ for _ in ()).throw(RuntimeError("unauthorized")),
    )

    with pytest.raises(RuntimeError, match="does not require a Kaggle token"):
        dataset_source.resolve_dataset("aad", "missing", tmp_path)


def test_malformed_download_fails_clearly(tmp_path, monkeypatch):
    malformed = tmp_path / "download"
    malformed.mkdir()
    monkeypatch.setattr(
        dataset_source.kagglehub, "dataset_download", lambda _handle: str(malformed)
    )

    with pytest.raises(RuntimeError, match="expected eleven-class AAD video layout"):
        dataset_source.resolve_dataset("aad", "missing", tmp_path)


def test_dataset_preview_resolves_missing_aad_before_loading(tmp_path, monkeypatch):
    from src import dataset

    downloaded_root = tmp_path / "kaggle-cache" / "aad"
    calls = []

    def resolve(name, dataset_dir, implementation_root):
        calls.append((name, dataset_dir, implementation_root))
        return downloaded_root

    class PreviewDataset:
        class_names = ["Fight"]
        samples = [(downloaded_root / "Fight" / "clip.mp4", 0)]

        def __init__(self, dataset_dir, sequence_length, frame_size):
            assert Path(dataset_dir) == downloaded_root

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return "video", 0

    monkeypatch.setattr(dataset, "resolve_dataset", resolve)
    monkeypatch.setattr(dataset, "AHARDataset", PreviewDataset)
    monkeypatch.setattr(
        dataset,
        "write_video_torchvision",
        lambda _video, path, fps: Path(path).touch(),
    )
    monkeypatch.chdir(tmp_path)

    dataset.main(["--num_samples", "1"])

    assert calls == [("aad", None, Path(dataset.__file__).resolve().parent.parent)]
    assert list((tmp_path / "outputs/dataset_samples").glob("*/*.mp4"))
