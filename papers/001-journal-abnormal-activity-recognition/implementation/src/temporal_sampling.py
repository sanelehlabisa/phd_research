"""Opt-in presentation-timestamp sampling; never used for legacy studies.

Zero-order hold: each target instant uses the last frame at or before it.
Low-FPS sources repeat frames without speeding up motion. After source duration,
repeat the last frame and explicitly record padding. Missing/non-monotonic PTS
are errors, not fabricated timing. No test clip is probed by the search.
"""

from bisect import bisect_right
from functools import lru_cache
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

VERSION = "timestamps_v1"


def timestamp_plan(timestamps, target_fps, count, source_fps=None, duration=None):
    if type(count) is not int or count < 1:
        raise ValueError("frame count must be positive")
    if type(target_fps) is not int or target_fps < 1:
        raise ValueError("target FPS must be a positive integer")
    times = [float(t) for t in timestamps]
    if not times or not all(math.isfinite(t) for t in times):
        raise ValueError("finite presentation timestamps are required")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("presentation timestamps must increase strictly")
    start = times[0]
    times = [t - start for t in times]
    intervals = [b - a for a, b in zip(times, times[1:])]
    fps = float(source_fps) if source_fps is not None else None
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("invalid source FPS")
    if duration is None:
        step = statistics.median(intervals) if intervals else (1 / fps if fps else None)
        if step is None:
            raise ValueError("single-frame sources need FPS or duration metadata")
        duration = times[-1] + step
    if not math.isfinite(duration) or duration <= times[-1]:
        raise ValueError("source duration must extend past its last frame")
    targets = [i / target_fps for i in range(count)]
    indices = [max(0, bisect_right(times, t + 1e-9) - 1) for t in targets]
    padding = [t >= duration - 1e-9 for t in targets]
    selected = [times[i] for i in indices]
    return {
        "sampling_version": VERSION,
        "source_fps": fps,
        "target_fps": target_fps,
        "source_frame_count": len(times),
        "source_duration_seconds": duration,
        "variable_frame_rate": bool(
            intervals and max(intervals) - min(intervals) > 1e-4
        ),
        "target_timestamps_seconds": targets,
        "selected_indices": indices,
        "selected_timestamps_seconds": selected,
        "unique_frames": len(set(indices)),
        "padding_mask": padding,
        "padding_fraction": sum(padding) / count,
        "duplicate_fraction": 1 - len(set(indices)) / count,
        "temporal_coverage_seconds": selected[-1] - selected[0],
        "requested_span_seconds": targets[-1],
        "policy": "clip_start_zero; causal_zero_order_hold; repeat_last_after_end",
    }


@lru_cache(maxsize=2048)
def _video_timeline(path, size, modified_ns):
    """Inspect train/validation PTS without materialising RGB video tensors."""
    import av

    timestamps = []
    last_duration = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Missing presentation timestamps: {path}")
            timestamps.append(float(frame.pts * frame.time_base))
            last_duration = (
                float(frame.duration * frame.time_base)
                if getattr(frame, "duration", 0)
                else None
            )
    duration = (
        timestamps[-1] - timestamps[0] + last_duration
        if timestamps and last_duration and last_duration > 0
        else None
    )
    return tuple(timestamps), fps, duration


def video_plan(path, target_fps, count):
    path = Path(path).resolve()
    stat = path.stat()
    timestamps, fps, duration = _video_timeline(
        str(path), stat.st_size, stat.st_mtime_ns
    )
    return timestamp_plan(timestamps, target_fps, count, fps, duration)


def load_timestamp_clip(path, plan, frame_size):
    """Decode only selected RGB frames; reuse duplicates and stop after the last."""
    import av

    wanted = set(plan["selected_indices"])
    selected = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                pixels = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                pixels = pixels.permute(2, 0, 1).float().unsqueeze(0)
                selected[i] = F.interpolate(
                    pixels, size=frame_size, mode="bilinear", align_corners=False
                )[0].div(255)
            if i >= max(wanted):
                break
    if set(selected) != wanted:
        raise ValueError(f"Video changed or decoding was incomplete: {path}")
    return torch.stack([selected[i] for i in plan["selected_indices"]])


def prepare_sampling_audit(dataset, indices, partitions):
    """Probe explicitly supplied training/validation indices only."""
    if dataset._mode != "video":
        raise ValueError(
            "timestamps_v1 requires original videos with presentation timestamps"
        )
    records = []
    for i in sorted(set(indices)):
        path = dataset.samples[i][0]
        key = str(Path(path).resolve())
        if key not in dataset.sampling_plans:
            dataset.sampling_plans[key] = video_plan(
                path, dataset.target_fps, dataset.sequence_length
            )
        records.append(
            {"source": key, "partition": partitions[i], **dataset.sampling_plans[key]}
        )
    return {
        "sampling_version": VERSION,
        "test_clips": 0,
        "records": records,
        "note": "Coverage is first-clip only, not full-video/stateful processing.",
    }
