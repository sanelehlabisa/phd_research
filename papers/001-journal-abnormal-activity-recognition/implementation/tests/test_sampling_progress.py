"""Timestamp preflight progress and exact failing-clip diagnostics, without videos."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import temporal_sampling


def source_dataset(tmp_path):
    return SimpleNamespace(
        _mode="video",
        samples=[
            (tmp_path / name, 0)
            for name in ("train.mp4", "locked-test.mp4", "validation.mp4", "last.mp4")
        ],
        target_fps=16,
        sequence_length=8,
        sampling_plans={},
    )


def test_progress_counts_unique_sorted_indices_including_cached_plans(tmp_path, monkeypatch):
    dataset = source_dataset(tmp_path)
    cached_key = str(dataset.samples[2][0].resolve())
    dataset.sampling_plans[cached_key] = {"cached": True}
    calls = []

    def probe(path, fps, count):
        calls.append(path)
        assert (fps, count) == (16, 8)
        return {"probe": Path(path).name}

    stream = io.StringIO()
    bars = []
    real_tqdm = temporal_sampling.tqdm

    def progress(iterable, **kwargs):
        bar = real_tqdm(iterable, file=stream, mininterval=0, disable=False, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(temporal_sampling, "video_plan", probe)
    monkeypatch.setattr(temporal_sampling, "tqdm", progress)
    partitions = {0: "train", 2: "validation", 3: "train"}
    result = temporal_sampling.prepare_sampling_audit(dataset, [3, 0, 2, 3], partitions)

    assert len(bars) == 1 and bars[0].total == bars[0].n == 3
    assert "3/3" in stream.getvalue() and "clip" in stream.getvalue()
    assert calls == [dataset.samples[0][0], dataset.samples[3][0]]
    assert [row["source"] for row in result["records"]] == [
        str(dataset.samples[index][0].resolve()) for index in (0, 2, 3)
    ]
    assert result["records"][1] == {
        "source": cached_key, "partition": "validation", "cached": True,
    }
    assert set(result) == {"sampling_version", "test_clips", "records", "note"}
    assert result["sampling_version"] == "timestamps_v1" and result["test_clips"] == 0


def test_probe_failure_names_clip_chains_original_and_stops(tmp_path, monkeypatch):
    dataset = source_dataset(tmp_path)
    failure = ValueError("presentation timestamps must increase strictly")
    bad_path = dataset.samples[2][0]
    calls = []

    def probe(path, fps, count):
        calls.append(path)
        if path == bad_path:
            raise failure
        return {"valid": True}

    monkeypatch.setattr(temporal_sampling, "video_plan", probe)
    with pytest.raises(RuntimeError, match="Timestamp audit failed for clip") as caught:
        temporal_sampling.prepare_sampling_audit(
            dataset, [0, 2, 3], {0: "train", 2: "validation", 3: "train"}
        )
    assert str(bad_path) in str(caught.value)
    assert str(failure) in str(caught.value)
    assert caught.value.__cause__ is failure
    assert calls == [dataset.samples[0][0], bad_path]
    assert str(bad_path.resolve()) not in dataset.sampling_plans


def test_locked_test_clip_is_never_opened(tmp_path, monkeypatch):
    dataset = source_dataset(tmp_path)
    locked = dataset.samples[1][0]
    calls = []

    def probe(path, fps, count):
        assert path != locked, "The locked test clip must not be probed"
        calls.append(path)
        return {"valid": True}

    monkeypatch.setattr(temporal_sampling, "video_plan", probe)
    result = temporal_sampling.prepare_sampling_audit(
        dataset, [2, 0], {0: "train", 2: "validation"}
    )
    assert calls == [dataset.samples[0][0], dataset.samples[2][0]]
    assert locked not in calls and str(locked.resolve()) not in dataset.sampling_plans
    assert result["test_clips"] == 0
