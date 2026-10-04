"""Bounded, strict video loading for exploratory notebooks (not the AAD protocol)."""

from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import av
import kagglehub
import torch

from . import notebook_config as settings
from .dataset import AHARDataset, create_split_manifest, load_split_subsets
from .vdd_diagnostic import resolve_vdd_root


class NotebookVideoDataset(AHARDataset):
    """Decode only the input window; forbid locked indices and corrupt fallbacks."""

    preprocessing = "timestamp-next-frame-rgb-resize-v1"

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
        self.cache_bytes = 0
        self.cache_limit_bytes = 64 * 1024 * 1024

    def __getitem__(self, index):
        if index not in self.allowed_indices:
            raise PermissionError(f"Clip {index} is locked for this stage")
        path, label = self.samples[index]
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index].clone(), label
        frames = []
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                rate = float(stream.average_rate or 0)
                if rate <= 0:
                    raise ValueError("source FPS is unavailable")
                start = None
                for number, frame in enumerate(container.decode(stream)):
                    timestamp = (
                        float(frame.time) if frame.time is not None else number / rate
                    )
                    if start is None:
                        start = timestamp
                    if timestamp - start + 1e-7 < len(frames) / self.target_fps:
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
                        and len(frames) / self.target_fps <= timestamp - start + 1e-7
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
            _, evicted = self._cache.popitem(last=False)
            self.cache_bytes -= evicted.numel() * evicted.element_size()
        if size <= self.cache_limit_bytes:
            self._cache[index] = clip
            self.cache_bytes += size
        return clip.clone(), label


def prepare_data(root, dataset_root=None, frame_size=None):
    """Create/reuse a split from file inventory only; never open test videos."""
    started = perf_counter()
    specification = settings.selected_diagnostic_dataset()
    print(f"Dataset: {specification.key} | {specification.kaggle_handle}", flush=True)
    print(f"Accepted classes: {specification.accepted_classes}", flush=True)
    if dataset_root is None:
        print(
            "Downloading/reusing Kaggle cache (download/extraction may take time)...",
            flush=True,
        )
        dataset_root = kagglehub.dataset_download(specification.kaggle_handle)
    print(
        "Locating class folders and building file inventory (no video decoding)...",
        flush=True,
    )
    size = frame_size or settings.FRAME_SIZE
    dataset = NotebookVideoDataset(
        resolve_vdd_root(dataset_root),
        sequence_length=settings.SEQUENCE_LENGTH,
        frame_size=(size, size),
        target_fps=settings.TARGET_FPS,
        accepted_classes=specification.accepted_classes,
    )
    root = Path(root)
    manifest = (
        root / "runs" / "manifests" / f"{specification.key}_diagnostic_seed42.json"
    )
    if not manifest.is_file():
        create_split_manifest(dataset, manifest, seed=42)
    train, validation, test, split = load_split_subsets(dataset, manifest, seed=42)
    dataset.allowed_indices = set(train.indices) | set(validation.indices)
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
    )


def data_identity(prepared):
    """Record labels and preprocessing alongside the split and dataset handle."""
    dataset = prepared["dataset"]
    return dict(
        dataset_key=prepared["specification"].key,
        kaggle_handle=prepared["specification"].kaggle_handle,
        classes=dataset.class_names,
        target_fps=dataset.target_fps,
        sequence_length=dataset.sequence_length,
        frame_size=list(dataset.frame_size),
        preprocessing=dataset.preprocessing,
        split_manifest_hash=prepared["split"]["manifest_hash"],
    )
