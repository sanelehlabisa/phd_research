"""One bounded, configuration-keyed train/validation RAM cache per runner process."""

import hashlib
import json
from pathlib import Path

from .dataset import CachedAHARDataset

MAX_CACHE_BYTES = 2 * 1024**3
_CACHE = {}


def cached_training_dataset(source, train_indices, validation_indices, split_hash):
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
        _CACHE[key] = CachedAHARDataset(
            source.dataset_dir,
            source.sequence_length,
            source.frame_size,
            target_fps=source.target_fps,
            sampling_version=getattr(source, "sampling_version", "legacy"),
            sampling_plans=getattr(source, "sampling_plans", {}),
            cache_indices=indices,
        )
    dataset = _CACHE[key]
    actual = sum(v.numel() * v.element_size() for v, _ in dataset._cache.values())
    return dataset, {**report, "bytes": actual, "mode": "bounded_train_validation_ram"}
