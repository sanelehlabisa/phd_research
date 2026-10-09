"""Bounded train/validation RAM plus reusable, configuration-keyed disk caches."""

import hashlib
import json
from pathlib import Path
import shutil
import torch

from .dataset import CachedAHARDataset

MAX_CACHE_BYTES = 2 * 1024**3
MAX_DISK_BYTES = 16 * 1024**3
DISK_RESERVE_BYTES = 2 * 1024**3
_CACHE = {}


def cached_training_dataset(
    source, train_indices, validation_indices, split_hash, cache_dir=None
):
    indices = sorted(set(train_indices + validation_indices))
    # Only selected train/validation file metadata is consulted; no test decoding.
    files = []
    for i in indices:
        path = Path(source.samples[i][0])
        children = (
            sorted(p for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else [path]
        )
        files.extend(
            (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in children
        )
    payload = {
        "root": str(source.dataset_dir.resolve()),
        "split": split_hash,
        "indices": indices,
        "files": files,
        "frames": source.sequence_length,
        "size": source.frame_size,
        "fps": source.target_fps,
        "sampling_version": getattr(source, "sampling_version", "legacy"),
        "mode": source._mode,
        "classes": source.class_names,
        "labels": [source.samples[i][1] for i in indices],
        "preprocessing_sha256": hashlib.sha256(
            Path(__file__).with_name("dataset.py").read_bytes()
        ).hexdigest(),
        "timestamp_sampling_sha256": hashlib.sha256(
            Path(__file__).with_name("temporal_sampling.py").read_bytes()
        ).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    estimate = (
        len(indices)
        * source.sequence_length
        * 3
        * source.frame_size[0]
        * source.frame_size[1]
        * 4
    )
    report = {
        "key": key,
        "maximum_bytes": MAX_CACHE_BYTES,
        "estimated_bytes": estimate,
        "train_clips": len(train_indices),
        "validation_clips": len(validation_indices),
        "test_clips": 0,
        "reused": key in _CACHE,
    }
    if estimate > MAX_CACHE_BYTES:
        _CACHE.clear()
        return source, {**report, "bytes": 0, "mode": "lazy_over_cache_limit"}
    if key not in _CACHE:
        _CACHE.clear()  # Only one input configuration retained, never an unbounded cache.
        dataset = CachedAHARDataset.__new__(CachedAHARDataset)
        dataset.__dict__ = source.__dict__.copy()
        dataset._cache = {}
        disk = Path(cache_dir) / f"{key}.pt" if cache_dir is not None else None
        marker = disk.with_suffix(".json") if disk is not None else None
        if marker is not None and marker.is_file():
            from .study_matrix import file_hash, read_json

            receipt = read_json(marker)
            if (
                receipt.get("key") != key
                or not disk.is_file()
                or file_hash(disk) != receipt.get("sha256")
            ):
                raise ValueError(
                    f"Processed data cache is incomplete or corrupt: {marker}"
                )
            dataset._cache = torch.load(disk, map_location="cpu", weights_only=True)
            expected_shape = (source.sequence_length, 3, *source.frame_size)
            if set(dataset._cache) != set(indices) or any(
                tuple(video.shape) != expected_shape
                or video.dtype != torch.float32
                or label != source.samples[i][1]
                for i, (video, label) in dataset._cache.items()
            ):
                raise ValueError(
                    "Processed cache tensors/labels do not match its configuration"
                )
            report.update(reused=True, reuse_source="disk", disk_path=str(disk))
        else:
            print(
                f"Caching {len(indices)} train/validation clips (test excluded)...",
                flush=True,
            )
            # Decode exactly the requested source. Never use legacy fallback to
            # another sample, which could silently substitute a held-out clip.
            for i in indices:
                path, label = source.samples[i]
                video = (
                    source._load_frames(path)
                    if source._mode == "frames"
                    else source._load_video(path)
                )
                dataset._cache[i] = (video, label)
                if len(dataset._cache) % 50 == 0 or len(dataset._cache) == len(indices):
                    print(f"  Cached {len(dataset._cache)}/{len(indices)}", flush=True)
            if disk is not None:
                from .study_matrix import atomic_json, file_hash

                disk.parent.mkdir(parents=True, exist_ok=True)
                used = sum(
                    p.stat().st_size for p in disk.parent.iterdir() if p.is_file()
                )
                required = int(estimate * 1.05) + 1024**2
                if (
                    used + required <= MAX_DISK_BYTES
                    and shutil.disk_usage(disk.parent).free
                    >= required + DISK_RESERVE_BYTES
                ):
                    temporary = disk.with_suffix(".tmp")
                    torch.save(dataset._cache, temporary)
                    temporary.replace(disk)
                    atomic_json(
                        marker,
                        {
                            "key": key,
                            "sha256": file_hash(disk),
                            "bytes": disk.stat().st_size,
                        },
                    )
                    report["disk_path"] = str(disk)
                else:
                    report["disk_skip_reason"] = "disk budget or free-space reserve"
        _CACHE[key] = dataset
    else:
        report["reuse_source"] = "ram"
    dataset = _CACHE[key]
    actual = sum(v.numel() * v.element_size() for v, _ in dataset._cache.values())
    return dataset, {
        **report,
        "bytes": actual,
        "mode": "bounded_train_validation_ram",
        "maximum_disk_bytes": MAX_DISK_BYTES,
    }
