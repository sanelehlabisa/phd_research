"""Bounded timestamp probing preserves causal inputs and reports its scope."""

from contextlib import contextmanager
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from src import temporal_sampling
from src.dataset import AHARDataset
from src.experiment_config import ExperimentConfig


def mock_video(tmp_path, monkeypatch, times, fps, *, video_stream=True):
    path = tmp_path / "clip.mp4"
    path.touch()
    counts = []
    frames = [
        SimpleNamespace(
            pts=time, time_base=1, duration=1 / fps,
            to_ndarray=lambda format, index=index: np.full((2, 2, 3), index, dtype=np.uint8),
        )
        for index, time in enumerate(times)
    ]

    @contextmanager
    def open_video(_path):
        count = [0]
        counts.append(count)

        def decode(*args, **kwargs):
            for frame in frames:
                count[0] += 1
                yield frame

        yield SimpleNamespace(
            streams=SimpleNamespace(video=[SimpleNamespace(average_rate=fps)] if video_stream else []),
            decode=decode,
        )

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_video))
    return path, counts


@pytest.mark.parametrize(
    "times,fps,target_fps,count",
    [
        ([10 + i / 30 for i in range(300)], 30, 16, 8),
        ([5, 5.02, 5.11, 5.3, 5.4, 5.8, 5.9, 6.1], 10, 8, 8),
        ([0, 0.25, 0.5, 0.75, 1], 4, 16, 8),
        ([0, 0.1], 10, 10, 8),
        ([0, 0.4375, 0.4375 + 5e-10, 0.5], 16, 16, 8),
    ],
)
def test_prefix_matches_full_indices_padding_and_tensors(
    tmp_path, monkeypatch, times, fps, target_fps, count
):
    path, probes = mock_video(tmp_path, monkeypatch, times, fps)
    full = temporal_sampling.video_plan(path, target_fps, count)
    clip = temporal_sampling.video_plan(
        path, target_fps, count, sampling_version="timestamps_clip_v1"
    )
    for field in (
        "selected_indices", "selected_timestamps_seconds", "target_timestamps_seconds",
        "padding_mask", "padding_fraction", "duplicate_fraction", "unique_frames",
        "temporal_coverage_seconds", "requested_span_seconds",
    ):
        assert clip[field] == full[field], field
    assert torch.equal(
        temporal_sampling.load_timestamp_clip(path, full, (2, 2)),
        temporal_sampling.load_timestamp_clip(path, clip, (2, 2)),
    )
    assert full["source_frame_count"] == len(times)
    assert clip["sampling_version"] == "timestamps_clip_v1"
    assert probes[1][0] <= probes[0][0]
    if not clip["source_complete"]:
        assert clip["audit_scope"] == "requested_clip_prefix"
        assert clip["source_frame_count"] is None
        assert clip["source_duration_seconds"] is None
        assert clip["variable_frame_rate"] is None
        assert clip["observed_prefix_frame_count"] == probes[1][0]
    else:
        assert clip["source_duration_seconds"] == full["source_duration_seconds"]
        assert clip["source_frame_count"] == len(times)


def test_long_video_probe_stops_after_one_lookahead(tmp_path, monkeypatch):
    path, probes = mock_video(tmp_path, monkeypatch, [i / 30 for i in range(300)], 30)
    plan = temporal_sampling.video_plan(path, 16, 8, sampling_version="timestamps_clip_v1")
    assert probes[0][0] == 15  # 0 through 0.4667 s, instead of all 300 frames.
    assert plan["selected_indices"] == [0, 1, 3, 5, 7, 9, 11, 13]
    assert plan["observed_prefix_span_seconds"] == pytest.approx(14 / 30)
    assert not plan["source_complete"] and plan["padding_fraction"] == 0


@pytest.mark.parametrize("version", ["timestamps_v1", "timestamps_clip_v1"])
@pytest.mark.parametrize("video_stream,error", [(False, "No video stream"), (True, "No decodable video frames")])
def test_streamless_and_frameless_files_have_explicit_context(tmp_path, monkeypatch, version, video_stream, error):
    path, _ = mock_video(tmp_path, monkeypatch, [], 30, video_stream=video_stream)
    with pytest.raises(ValueError, match=error) as caught:
        temporal_sampling.video_plan(path, 16, 8, sampling_version=version)
    assert str(path) in str(caught.value)


def test_real_audio_only_mp4_reports_missing_video_stream(tmp_path):
    import av

    path = tmp_path / "audio-only.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("aac", rate=16000)
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 1024), dtype=np.int16), format="s16", layout="mono"
        )
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for version in ("timestamps_v1", "timestamps_clip_v1"):
        with pytest.raises(ValueError, match="No video stream") as caught:
            temporal_sampling.video_plan(path, 16, 8, sampling_version=version)
        assert str(path) in str(caught.value)


def test_config_dataset_and_audit_use_prefix_mode_without_probing_test(tmp_path, monkeypatch):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "clip.mp4").touch()
    config = ExperimentConfig.from_mapping({
        "dataset_name": "kinetics400-exploratory", "dataset_dir": str(tmp_path),
        "sampling_version": "timestamps_clip_v1",
    })
    source = AHARDataset(tmp_path, 8, (2, 2), sampling_version=config.sampling_version)
    calls = []

    def probe(path, fps, count, *, sampling_version):
        assert sampling_version == "timestamps_clip_v1"
        calls.append(path)
        return {"sampling_version": sampling_version, "selected_indices": [0] * 8}

    monkeypatch.setattr(temporal_sampling, "video_plan", probe)
    monkeypatch.setattr(temporal_sampling, "load_timestamp_clip", lambda *args: torch.zeros(8, 3, 2, 2))
    report = temporal_sampling.prepare_sampling_audit(source, [0], {0: "train"})
    assert report["sampling_version"] == config.sampling_version
    assert report["test_clips"] == 0 and calls == [source.samples[0][0]]
    source.sampling_plans.clear()
    assert source._load_video(source.samples[0][0]).shape == (8, 3, 2, 2)
    assert calls == [source.samples[0][0]] * 2


def test_exclusions_normalize_and_empty_default_keeps_historical_schema():
    assert "excluded_samples" not in ExperimentConfig().to_dict()
    config = ExperimentConfig.from_mapping({"excluded_samples": ["a/bad.mp4"]})
    assert config.excluded_samples == ("a/bad.mp4",)
    assert config.to_dict()["excluded_samples"] == ["a/bad.mp4"]


@pytest.mark.parametrize("paths", [
    "a/bad.mp4", ["../bad.mp4"], ["a/../bad.mp4"], ["/bad.mp4"],
    ["C:/bad.mp4"], ["a\\bad.mp4"], ["a//bad.mp4"], ["a/bad.mp4", "a/bad.mp4"],
])
def test_unsafe_or_duplicate_exclusions_are_rejected(paths):
    with pytest.raises(ValueError, match="excluded_samples"):
        ExperimentConfig.from_mapping({"excluded_samples": paths})


def test_exclusion_filters_exact_clip_preserves_classes_and_originals(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        for clip in ("bad.mp4", "good.mp4"):
            (tmp_path / name / clip).touch()
    source = AHARDataset(tmp_path, excluded_samples=["a/bad.mp4"])
    assert source.class_names == ["a", "b"]
    assert [path.relative_to(tmp_path).as_posix() for path, _ in source.samples] == [
        "a/good.mp4", "b/bad.mp4", "b/good.mp4",
    ]
    assert (tmp_path / "a/bad.mp4").is_file()
    with pytest.raises(ValueError, match="paths not found"):
        AHARDataset(tmp_path, excluded_samples=["a/unknown.mp4"])
    with pytest.raises(ValueError, match="classes without clips"):
        AHARDataset(tmp_path, excluded_samples=["a/bad.mp4", "a/good.mp4"])
