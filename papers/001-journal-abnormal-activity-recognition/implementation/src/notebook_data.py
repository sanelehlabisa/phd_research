"""Bounded, strict video loading for exploratory notebooks (not the AAD protocol)."""

from collections import OrderedDict
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import av
import kagglehub
import torch

from . import notebook_config as settings
from .dataset import AHARDataset, create_split_manifest, load_split_subsets
from .kinetics_subset import download_subset, matched_interests, versioned_handle
from .kinetics600_subset import (
    GROUPING,
    content_hash,
    load_grouped_subsets,
    prepare_subset,
    unique_clip_records,
)
from .vdd_diagnostic import resolve_vdd_root


class NotebookVideoDataset(AHARDataset):
    """Decode only the input window; forbid locked indices and corrupt fallbacks."""

    preprocessing = "timestamp-window-rgb-resize-v2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.sequence_length, self.target_fps, *self.frame_size)
        ):
            raise ValueError(
                "Frame count, FPS and frame dimensions must be positive integers"
            )
        if self._mode != "video":
            raise ValueError(
                "Notebook inputs must be videos under exact class directories"
            )
        self.allowed_indices = set()
        self._cache = OrderedDict()
        self._cached_windows = {}
        self.cache_bytes = 0
        self.cache_limit_bytes = 64 * 1024 * 1024
        self.training_indices = set()
        self.training_windows = False
        self.epoch = 0
        self.window_seed = settings.SEED
        self.window_records = {}

    def set_epoch(self, epoch, seed, training=False):
        self.epoch, self.window_seed, self.training_windows = epoch, seed, training

    def window(self, index):
        """Inspect only an allowed video; never use test headers during setup."""
        if index not in self.allowed_indices:
            raise PermissionError(f"Clip {index} is locked for this stage")
        with av.open(str(self.samples[index][0])) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 0)
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else float(container.duration or 0) / av.time_base
            )
            if duration <= 0 or fps <= 0:
                raise ValueError("Source duration/FPS unavailable; audit the video")
        span = self.sequence_length / self.target_fps
        available = max(0.0, duration - span)
        training = self.training_windows and index in self.training_indices
        seed = f"{self.window_seed}:{self.epoch}:{index}"
        start = random.Random(seed).uniform(0, available) if training else available / 2
        return dict(
            start_seconds=start,
            duration_seconds=duration,
            native_fps=fps,
            window_seconds=span,
            sampling="random" if training else "center",
        )

    def __getitem__(self, index):
        if index not in self.allowed_indices:
            raise PermissionError(f"Clip {index} is locked for this stage")
        path, label = self.samples[index]
        training = self.training_windows and index in self.training_indices
        key = (
            index,
            self.epoch if training else None,
            self.window_seed if training else None,
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            self.window_records[index] = self._cached_windows[key]
            return self._cache[key].clone(), label
        frames = []
        try:
            window = self.window(index)
            self.window_records[index] = window
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                rate = float(stream.average_rate or 0)
                if rate <= 0:
                    raise ValueError("source FPS is unavailable")
                start = None
                previous = None
                for number, frame in enumerate(container.decode(stream)):
                    timestamp = (
                        float(frame.time) if frame.time is not None else number / rate
                    )
                    if start is None:
                        start = timestamp
                    if previous is not None and timestamp <= previous:
                        raise ValueError("non-increasing source timestamps")
                    previous = timestamp
                    relative = timestamp - start - window["start_seconds"]
                    if relative + 1e-7 < len(frames) / self.target_fps:
                        continue
                    array = frame.to_ndarray(
                        format="rgb24",
                        width=self.frame_size[1],
                        height=self.frame_size[0],
                    )
                    tensor = (
                        torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255
                    )
                    while (
                        len(frames) < self.sequence_length
                        and len(frames) / self.target_fps <= relative + 1e-7
                    ):
                        frames.append(tensor)
                    if len(frames) == self.sequence_length:
                        break
            if not frames:
                raise ValueError("no decodable frames")
            frames.extend([frames[-1]] * (self.sequence_length - len(frames)))
            clip = torch.stack(frames)
        except Exception as error:
            raise ValueError(
                f"Cannot decode {path}; no sample was substituted: {error}"
            ) from error
        size = clip.numel() * clip.element_size()
        while self._cache and self.cache_bytes + size > self.cache_limit_bytes:
            evicted_key, evicted = self._cache.popitem(last=False)
            self._cached_windows.pop(evicted_key)
            self.cache_bytes -= evicted.numel() * evicted.element_size()
        if size <= self.cache_limit_bytes:
            self._cache[key] = clip
            self._cached_windows[key] = window
            self.cache_bytes += size
        return clip.clone(), label


def prepare_data(
    root, dataset_root=None, frame_size=None, sequence_length=None, *, target_fps=None
):
    """Create/reuse a split from file inventory only; never open test videos."""
    started = perf_counter()
    root = Path(root)
    specification = settings.selected_diagnostic_dataset()
    print(
        f"Dataset: {specification.key} | {specification.source_url or specification.kaggle_handle}",
        flush=True,
    )
    kinetics = specification.key == "kinetics-subset"
    kinetics600 = specification.key == "kinetics600-subset"
    source = None
    if kinetics600:
        if (
            specification.source_url != settings.KINETICS600_SOURCE_URL
            or specification.accepted_classes != settings.KINETICS600_CLASSES
        ):
            raise ValueError(
                "Kinetics-600 requires the approved source and five exact activities"
            )
        dataset_root, source = prepare_subset(root, dataset_root)
        accepted = specification.accepted_classes
    elif kinetics:
        _, slug, _ = versioned_handle(specification.kaggle_handle)
        if dataset_root is None:
            dataset_root, accepted = download_subset(
                root, specification.kaggle_handle, settings.CLASSES_OF_INTEREST
            )
        else:
            dataset_root = Path(dataset_root)
            if (dataset_root / slug).is_dir():
                dataset_root = dataset_root / slug
            accepted = matched_interests(
                (p.name for p in dataset_root.iterdir() if p.is_dir()),
                settings.CLASSES_OF_INTEREST,
            )
    else:
        accepted = specification.accepted_classes
    if not kinetics and not kinetics600 and dataset_root is None:
        print(
            "Downloading/reusing Kaggle cache (download/extraction may take time)...",
            flush=True,
        )
        dataset_root = kagglehub.dataset_download(specification.kaggle_handle)
    if not kinetics and not kinetics600:
        dataset_root = resolve_vdd_root(dataset_root)
    print(f"Accepted classes: {accepted}", flush=True)
    print(
        "Locating class folders and building file inventory (no video decoding)...",
        flush=True,
    )
    size = frame_size or settings.FRAME_SIZE
    dataset = NotebookVideoDataset(
        dataset_root,
        sequence_length=(
            settings.SEQUENCE_LENGTH if sequence_length is None else sequence_length
        ),
        frame_size=(size, size),
        target_fps=target_fps or settings.TARGET_FPS,
        accepted_classes=accepted,
    )
    if kinetics600:
        unique_paths = {row["path"] for row in unique_clip_records(source["files"])}
        samples = [
            (path, label)
            for path, label in dataset.samples
            if path.relative_to(dataset.dataset_dir).as_posix() in unique_paths
        ]
        sample_paths = {
            path.relative_to(dataset.dataset_dir).as_posix() for path, _ in samples
        }
        if sample_paths != unique_paths:
            raise ValueError(
                "Kinetics-600 unique inventory differs from files found by the dataset"
            )
        excluded = len(dataset.samples) - len(samples)
        dataset.samples = samples
        print(
            f"Excluded {excluded} exact-duplicate or ambiguous-label files.", flush=True
        )
    seed = settings.SEED
    suffix = ""
    if kinetics:
        identity = json.dumps(
            dict(
                handle=specification.kaggle_handle,
                classes=dataset.class_names,
                seed=seed,
            ),
            sort_keys=True,
        )
        suffix = "_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    if kinetics600:
        suffix = (
            "_" + content_hash(dict(source=source, grouping=GROUPING, seed=seed))[:16]
        )
    manifest = (
        root
        / "runs"
        / "manifests"
        / f"{specification.key}{suffix}_diagnostic_seed{seed}.json"
    )
    if kinetics600:
        train, validation, test, split = load_grouped_subsets(
            dataset, manifest, source, seed
        )
        print(
            f"Source-ID grouped diagnostic split: {split['actual_ratios']} | "
            f"{split['known_source_clips']}/{len(dataset)} clips have identified source IDs. "
            "Upstream train clips only; not the official benchmark split. "
            "Subject independence and near-duplicate absence are not established.",
            flush=True,
        )
    else:
        if not manifest.is_file():
            create_split_manifest(dataset, manifest, seed=seed)
        train, validation, test, split = load_split_subsets(
            dataset, manifest, seed=seed
        )
    dataset.allowed_indices = set(train.indices) | set(validation.indices)
    dataset.training_indices = set(train.indices)
    baselines = {}
    for partition in ("train", "validation"):
        counts = {
            name: split["class_counts"][name][partition] for name in dataset.class_names
        }
        baselines[partition] = max(counts.values()) / sum(counts.values())
        print(f"{partition} class counts: {counts}", flush=True)
        print(
            f"{partition} majority-class baseline: {baselines[partition]:.1%}",
            flush=True,
        )
    print(
        f"Ready in {perf_counter() - started:.1f}s | train={len(train)}, "
        f"validation={len(validation)}, test={len(test)} (locked)",
        flush=True,
    )
    return dict(
        root=root,
        dataset=dataset,
        train=train,
        validation=validation,
        test=test,
        split=split,
        manifest_path=manifest,
        specification=specification,
        majority_baselines=baselines,
        source=source,
    )


def data_identity(prepared):
    """Record labels and preprocessing alongside the split and dataset handle."""
    dataset = prepared["dataset"]
    identity = dict(
        dataset_key=prepared["specification"].key,
        kaggle_handle=prepared["specification"].kaggle_handle,
        classes=dataset.class_names,
        target_fps=dataset.target_fps,
        sequence_length=dataset.sequence_length,
        frame_size=list(dataset.frame_size),
        preprocessing=dataset.preprocessing,
        training_sampling="seeded-random-window-per-epoch",
        evaluation_sampling="center-window",
        split_manifest_hash=prepared["split"]["manifest_hash"],
    )
    if prepared.get("source") is not None:
        identity.update(
            source_url=prepared["specification"].source_url,
            source_hash=content_hash(prepared["source"]),
            source_inventory_hash=prepared["source"]["inventory_hash"],
            release=prepared["source"]["release"],
            upstream_split=prepared["source"]["upstream_split"],
            grouping=prepared["split"]["grouping"],
        )
    return identity
