"""Selective public metadata/downloads, label filtering and split isolation."""

from dataclasses import replace
import io
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import av
import pytest

from notebooks.utils import kinetics_subset as subset
from notebooks.utils import config as settings
from notebooks.utils import data

HANDLE = "sanelehlabisa/kinetics-400-dataset/versions/1"
SLUG = "kinetics-400-dataset"


def entry(label, index=0, size=12):
    return dict(name=f"{SLUG}/{label}/{index}.mp4", totalBytes=size)


def fake_pages(monkeypatch):
    calls = []

    def page(handle, token):
        assert handle == HANDLE
        calls.append(token)
        if not token:
            return dict(
                datasetFiles=[entry("headbutting"), entry("unrelated")],
                nextPageToken="next",
            )
        assert token == "next"
        return dict(datasetFiles=[entry("hugging")])

    monkeypatch.setattr(subset, "_get_page", page)
    return calls


def test_api_request_is_version_pinned(monkeypatch):
    def open_request(request, timeout):
        url = urlparse(request.full_url)
        assert url.netloc == "www.kaggle.com"
        assert parse_qs(url.query) == {
            "datasetVersionNumber": ["1"],
            "pageSize": ["200"],
            "pageToken": ["a&b"],
        }
        assert timeout == 30
        return io.BytesIO(b'{"datasetFiles": []}')

    monkeypatch.setattr(subset, "urlopen", open_request)
    assert subset._get_page(HANDLE, "a&b") == {"datasetFiles": []}
    with pytest.raises(ValueError, match="versions/NUMBER"):
        subset.versioned_handle("sanelehlabisa/kinetics-400-dataset")


def test_download_only_matches_reuses_cache_and_preserves_environment(
    tmp_path, monkeypatch, capsys
):
    calls = fake_pages(monkeypatch)
    downloads = []
    monkeypatch.setenv("DISABLE_COLAB_CACHE", "original")
    monkeypatch.delenv("DISABLE_KAGGLE_CACHE", raising=False)

    def download(handle, *, path, output_dir, force_download):
        assert handle == HANDLE and path and "unrelated" not in path
        assert os.environ["DISABLE_COLAB_CACHE"] == "true"
        assert os.environ["DISABLE_KAGGLE_CACHE"] == "true"
        downloads.append((path, force_download))
        target = Path(output_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"video bytes!")
        return str(target)

    monkeypatch.setattr(subset.kagglehub, "dataset_download", download)
    interests = ("headbutting", "hugging", "Fight", "absent")
    root, classes = subset.download_subset(tmp_path, HANDLE, interests)
    assert classes == ["headbutting", "hugging"]
    assert len(downloads) == 2 and calls == ["", "next"]
    assert not (root / "unrelated").exists()
    assert os.environ["DISABLE_COLAB_CACHE"] == "original"
    assert "DISABLE_KAGGLE_CACHE" not in os.environ
    assert "Skipped absent interests: ['Fight', 'absent']" in capsys.readouterr().out

    # Neither inventory nor video downloads repeat; ignored names do not matter.
    assert subset.download_subset(tmp_path, HANDLE, interests) == (root, classes)
    assert len(downloads) == 2 and len(calls) == 2
    (root / "headbutting" / "0.mp4").write_bytes(b"partial")
    subset.download_subset(tmp_path, HANDLE, interests)
    assert len(downloads) == 3 and downloads[-1][1] is True


@pytest.mark.parametrize(
    "interests", [("absent", "missing"), ("headbutting", "absent")]
)
def test_zero_or_one_match_never_downloads(tmp_path, monkeypatch, interests):
    fake_pages(monkeypatch)
    monkeypatch.setattr(
        subset.kagglehub,
        "dataset_download",
        lambda *a, **k: pytest.fail("downloaded before checking classes"),
    )
    with pytest.raises(ValueError, match="at least two"):
        subset.download_subset(tmp_path, HANDLE, interests)


@pytest.mark.parametrize("failure", ["network", "short"])
def test_download_failure_does_not_fallback_or_leave_mounts_disabled(
    tmp_path, monkeypatch, failure
):
    fake_pages(monkeypatch)
    monkeypatch.delenv("DISABLE_COLAB_CACHE", raising=False)
    monkeypatch.setenv("DISABLE_KAGGLE_CACHE", "false")
    calls = []

    def download(handle, *, path, output_dir, force_download):
        calls.append(path)
        assert path
        if failure == "network":
            raise ConnectionError("offline")
        target = Path(output_dir) / path
        target.parent.mkdir(parents=True)
        target.write_bytes(b"short")
        return str(target)

    monkeypatch.setattr(subset.kagglehub, "dataset_download", download)
    with pytest.raises((ConnectionError, ValueError)):
        subset.download_subset(tmp_path, HANDLE, ("headbutting", "hugging"))
    assert len(calls) == 1
    assert "DISABLE_COLAB_CACHE" not in os.environ
    assert os.environ["DISABLE_KAGGLE_CACHE"] == "false"


@pytest.mark.parametrize(
    "name",
    [
        "../hugging/0.mp4",
        f"{SLUG}/../0.mp4",
        f"{SLUG}/hugging/C:bad.mp4",
        f"{SLUG}/hugging/a\\b.mp4",
    ],
)
def test_unsafe_inventory_paths_rejected(tmp_path, monkeypatch, name):
    monkeypatch.setattr(
        subset,
        "_get_page",
        lambda *a: dict(datasetFiles=[dict(name=name, totalBytes=12)]),
    )
    with pytest.raises(ValueError, match="Invalid"):
        subset.load_inventory(HANDLE, tmp_path)
    assert not (tmp_path / "inventory.json").exists()


def test_pagination_and_wrong_cached_version_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subset, "_get_page", lambda *a: dict(datasetFiles=[], nextPageToken="repeated")
    )
    with pytest.raises(ValueError, match="repeated pagination"):
        subset.load_inventory(HANDLE, tmp_path)
    (tmp_path / "inventory.json").write_text(
        json.dumps(dict(handle=HANDLE + "0", files=[]))
    )
    with pytest.raises(ValueError, match="another version"):
        subset.load_inventory(HANDLE, tmp_path)


def test_partial_and_nonvideo_entries_are_not_selected():
    clip = dict(name=f"{SLUG}/headbutting/a.mp4", bytes=12)
    assert subset._validated_files(
        [
            clip,
            dict(name=f"{SLUG}/headbutting/b.mp4.part", bytes=7),
            dict(name="README.txt", bytes=20),
        ],
        SLUG,
    ) == [clip]
    with pytest.raises(ValueError, match="duplicate"):
        subset._validated_files([clip, clip], SLUG)


def local_inventory(root, classes, count=12):
    # Preparation should use only paths/sizes, never inspect this invalid video.
    for name in classes:
        directory = root / name
        directory.mkdir(parents=True)
        for i in range(count):
            (directory / f"{i}.mp4").write_bytes(b"not decoded")


def test_local_filter_manifests_version_and_baselines(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(settings, "SELECTED_DIAGNOSTIC_DATASET", "kinetics-subset")
    monkeypatch.setattr(
        settings, "CLASSES_OF_INTEREST", ("headbutting", "hugging", "Fight")
    )
    monkeypatch.setattr(
        subset, "_get_page", lambda *a: pytest.fail("local root fetched metadata")
    )
    monkeypatch.setattr(
        subset.kagglehub,
        "dataset_download",
        lambda *a, **k: pytest.fail("local root downloaded"),
    )
    monkeypatch.setattr(
        av, "open", lambda *a, **k: pytest.fail("preparation decoded a video")
    )
    download = tmp_path / "local"
    local_inventory(
        download / SLUG, ("headbutting", "hugging", "slapping", "unrelated")
    )
    first = data.prepare_data(tmp_path, download)
    manifest = first["manifest_path"]
    original = manifest.read_bytes()
    assert first["dataset"].class_names == ["headbutting", "hugging"]
    assert len(first["dataset"]) == 24
    assert first["majority_baselines"]["train"] == 0.5
    assert "validation majority-class baseline" in capsys.readouterr().out
    assert first["dataset"].allowed_indices.isdisjoint(first["test"].indices)
    with pytest.raises(PermissionError):
        first["dataset"][first["test"].indices[0]]
    assert data.prepare_data(tmp_path, download)["manifest_path"] == manifest

    # Interests absent from the copy do not disturb an existing split.
    monkeypatch.setattr(
        settings, "CLASSES_OF_INTEREST", ("hugging", "headbutting", "Missing")
    )
    assert data.prepare_data(tmp_path, download)["manifest_path"] == manifest
    monkeypatch.setattr(
        settings, "CLASSES_OF_INTEREST", ("headbutting", "hugging", "slapping")
    )
    three = data.prepare_data(tmp_path, download)
    assert three["manifest_path"] != manifest
    assert data.data_identity(three) != data.data_identity(first)
    spec = settings.DIAGNOSTIC_DATASETS["kinetics-subset"]
    monkeypatch.setitem(
        settings.DIAGNOSTIC_DATASETS,
        "kinetics-subset",
        replace(spec, kaggle_handle=HANDLE[:-1] + "2"),
    )
    newer = data.prepare_data(tmp_path, download)
    assert newer["manifest_path"] not in (manifest, three["manifest_path"])
    assert data.data_identity(newer)["kaggle_handle"].endswith("/versions/2")
    assert manifest.read_bytes() == original


def test_insufficient_local_samples_fail_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SELECTED_DIAGNOSTIC_DATASET", "kinetics-subset")
    local_inventory(tmp_path / "local", ("headbutting", "hugging"), count=1)
    monkeypatch.setattr(av, "open", lambda *a, **k: pytest.fail("decoded video"))
    with pytest.raises(ValueError, match="stratified split failed"):
        data.prepare_data(tmp_path, tmp_path / "local")


def test_interest_list_contains_existing_labels_and_verified_subset():
    repo = Path(__file__).resolve().parents[4]
    evidence = json.loads(
        (
            repo / "agents/work/040-kinetics-interest-filter/dataset-inventory.json"
        ).read_text()
    )
    selected = subset.matched_interests(
        (r["class"] for r in evidence["selected_classes"]), settings.CLASSES_OF_INTEREST
    )
    assert len(selected) == 5
    assert len(evidence["files"]) == 87
    assert sum(f["bytes"] for f in evidence["files"]) == 91_112_992
    aad = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "splits/abnormal-activities-dataset_seed42.json"
        ).read_text()
    )
    assert set(aad["class_names"]) <= set(settings.CLASSES_OF_INTEREST)
    assert {"non-violent", "violent"} <= set(settings.CLASSES_OF_INTEREST)
