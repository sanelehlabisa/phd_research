"""
dataset.py

Video and frames dataset classes for AHAR.

Author: Sanele Hlabisa

python -m src.dataset \
    --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
    --seed 42 \
    --augment \
    --num_samples 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from .utils import (
    TARGET_FPS,
    read_video_torchvision,
    safe_filename,
    seed_everything,
    write_json,
    write_video_torchvision,
)

DEFAULT_SPLIT_SEED = 42


class VideoAugmentation:
    """Apply one conservative, temporally consistent transform to a video.

    Parameters:
        None.

    Returns:
        A callable video augmentation policy.
    """

    @staticmethod
    def _chance(probability: float) -> bool:
        """Draw one Boolean augmentation decision.

        Parameters:
            probability: Probability that the transform is applied.

        Returns:
            Whether to apply the transform.
        """
        return torch.rand(()).item() < probability

    @staticmethod
    def _uniform(lower: float, upper: float) -> float:
        """Draw one floating-point value from a uniform range.

        Parameters:
            lower: Inclusive lower bound.
            upper: Exclusive upper bound.

        Returns:
            A sampled value within the range.
        """
        return lower + (upper - lower) * torch.rand(()).item()

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Augment a complete clip using one parameter set for every frame.

        Parameters:
            video: Clip shaped `(T, C, H, W)` with values in `[0, 1]`.

        Returns:
            The augmented clip with the same shape and value range.
        """
        if video.ndim != 4:
            raise ValueError("video must have shape (T, C, H, W)")
        _, _, height, width = video.shape
        augmented = video

        if self._chance(0.5):
            augmented = transform_functional.hflip(augmented)

        if self._chance(0.3):
            angle = self._uniform(-8.0, 8.0)
            translate = [
                round(self._uniform(-0.05, 0.05) * width),
                round(self._uniform(-0.05, 0.05) * height),
            ]
            scale = self._uniform(0.95, 1.05)
            augmented = torch.stack(
                [
                    transform_functional.affine(
                        frame,
                        angle=angle,
                        translate=translate,
                        scale=scale,
                        shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR,
                    )
                    for frame in augmented
                ]
            )

        if self._chance(0.3):
            brightness = self._uniform(0.85, 1.15)
            contrast = self._uniform(0.85, 1.15)
            augmented = torch.stack(
                [
                    transform_functional.adjust_contrast(
                        transform_functional.adjust_brightness(frame, brightness),
                        contrast,
                    )
                    for frame in augmented
                ]
            )

        if self._chance(0.1):
            sigma = self._uniform(0.1, 1.0)
            augmented = transform_functional.gaussian_blur(
                augmented,
                kernel_size=[3, 3],
                sigma=[sigma, sigma],
            )

        return augmented.clamp(0.0, 1.0)


class AHARDataset(Dataset):
    """
    Unified dataset class for loading video or pre-extracted frame datasets.
    """

    SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        dataset_dir: str | Path,
        sequence_length: int = 64,
        frame_size: tuple[int, int] = (112, 112),
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_fps: int = TARGET_FPS,
    ) -> None:
        """
        Initializes the dataset and automatically detects the data format.

        Parameters:
            dataset_dir (str | Path): Path to the dataset directory.
            sequence_length (int): Number of frames to sample per clip.
            frame_size (tuple[int, int]): Target spatial resolution for the frames.
            transform: Optional full-video transform to apply.
            target_fps (int): The consistent frame rate to sample clips at.

        Returns:
            None
        """
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.transform = transform
        self.target_fps = target_fps

        self.class_names: list[str] = sorted(
            d.name for d in self.dataset_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self._mode = self._detect_mode()

        self.samples: list[tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.dataset_dir / cls
            if self._mode == "frames":
                for clip_dir in sorted(cls_dir.iterdir()):
                    if clip_dir.is_dir() and any(clip_dir.glob("*.png")):
                        self.samples.append((clip_dir, self.class_to_idx[cls]))
            else:
                for f in sorted(cls_dir.iterdir()):
                    if f.suffix.lower() in self.SUPPORTED_EXTS:
                        self.samples.append((f, self.class_to_idx[cls]))

        print(
            f"✅ {len(self.samples)} {'clips' if self._mode == 'frames' else 'videos'} "
            f"| mode={self._mode} | {self.num_classes} classes: {self.class_names}"
        )

    def _detect_mode(self) -> str:
        """
        Checks the first class directory to determine if the dataset uses frames or videos.

        Parameters:
            None

        Returns:
            mode (str): The detected dataset format ('frames' or 'video').
        """
        for cls in self.class_names:
            cls_dir = self.dataset_dir / cls
            for item in cls_dir.iterdir():
                if item.is_dir() and any(item.glob("*.png")):
                    return "frames"
                if item.suffix.lower() in self.SUPPORTED_EXTS:
                    return "video"
        return "video"

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Parameters:
            None

        Returns:
            length (int): Total sample count.
        """
        return len(self.samples)

    def _sample_frames_tensor(
        self, frames: torch.Tensor, source_fps: float
    ) -> torch.Tensor:
        """
        Samples a fixed sequence of frames based on the target frames per second,
        preserving the natural speed of the motion regardless of video length.

        Parameters:
            frames (torch.Tensor): The input frame sequence tensor.
            source_fps (float): The original frames per second of the video.

        Returns:
            sampled_frames (torch.Tensor): The reduced and padded frame sequence tensor.
        """
        T = frames.shape[0]

        # Calculate stride to match target FPS
        stride = max(1, round(source_fps / self.target_fps))
        indices = list(range(0, T, stride))

        # Truncate or pad to exactly sequence_length
        if len(indices) >= self.sequence_length:
            indices = indices[: self.sequence_length]
        else:
            pad = self.sequence_length - len(indices)
            indices += [indices[-1]] * pad

        return frames[torch.tensor(indices)]

    def _load_video(self, path: Path) -> torch.Tensor:
        """
        Loads a video file from disk and processes it into a normalized frame tensor.

        Parameters:
            path (Path): Path to the video file.

        Returns:
            video_tensor (torch.Tensor): Processed video tensor of shape (T, C, H, W).
        """
        video, fps = read_video_torchvision(path)  # (T, H, W, C) uint8
        video = self._sample_frames_tensor(video, fps)  # (T, H, W, C)
        video = video.permute(0, 3, 1, 2).float()  # (T, C, H, W)
        video = F.interpolate(
            video, size=self.frame_size, mode="bilinear", align_corners=False
        ).div(255.0)
        return video

    def _load_frames(self, clip_dir: Path) -> torch.Tensor:
        """
        Loads pre-extracted PNG frames from a directory into a normalized tensor.

        Parameters:
            clip_dir (Path): The directory containing the PNG frames and metadata.

        Returns:
            frames_tensor (torch.Tensor): Processed image tensor of shape (T, C, H, W).
        """
        all_pngs = sorted(p for p in clip_dir.glob("*.png"))
        T = len(all_pngs)

        if T >= self.sequence_length:
            indices = torch.linspace(0, T - 1, self.sequence_length).long().tolist()
        else:
            indices = list(range(T)) + [T - 1] * (self.sequence_length - T)

        frames = []
        for i in indices:
            img = torchvision.io.read_image(str(all_pngs[i])).float().div(255.0)
            if tuple(img.shape[1:]) != tuple(self.frame_size):
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=self.frame_size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
            frames.append(img)

        return torch.stack(frames)

    def __getitem__(self, index: int):
        """
        Fetches and optionally transforms a single dataset sample by index.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            sample (tuple): A tuple containing the processed video tensor and its label index.
        """
        path, label = self.samples[index]

        try:
            if self._mode == "frames":
                video = self._load_frames(path)
            else:
                video = self._load_video(path)
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            video = self.transform(video)

        return video, label


class CachedAHARDataset(AHARDataset):
    """
    Dataset subclass that caches all processed clips in RAM to eliminate disk bottlenecks.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the dataset and pre-loads all samples into memory.

        Parameters:
            *args (Any): Positional arguments passed to the parent AHARDataset class.
            **kwargs (Any): Keyword arguments passed to the parent AHARDataset class.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)
        print(f"📥 Caching {len(self.samples)} clips into RAM...")
        self._cache: list[tuple[torch.Tensor, int]] = []

        saved_transform = self.transform
        self.transform = None  # disable during caching

        for i in range(len(self.samples)):
            video, label = super().__getitem__(i)
            self._cache.append((video, label))
            if (i + 1) % 50 == 0 or (i + 1) == len(self.samples):
                print(f"   {i+1}/{len(self.samples)}")

        self.transform = saved_transform  # restore

        mem_gb = sum(v.nbytes for v, _ in self._cache) / 1e9
        print(f"✅ Cached {len(self._cache)} clips - ~{mem_gb:.2f} GB RAM")

    def __getitem__(self, index: int):
        """
        Fetches a single dataset sample from the RAM cache and applies runtime transforms.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            sample (tuple): A tuple containing the augmented video tensor and its label index.
        """
        video, label = self._cache[index]
        if self.transform:
            video = self.transform(video)
        return video, label


def resolve_split_manifest_path(
    dataset_dir: str | Path,
    split_manifest: str | Path | None,
    seed: int,
) -> Path:
    """Resolve an explicit manifest or the standard dataset-and-seed path.

    Parameters:
        dataset_dir: Dataset root used to derive the default filename.
        split_manifest: Optional explicit manifest path.
        seed: Split seed included in the default filename.

    Returns:
        The resolved manifest path.
    """
    if split_manifest is not None:
        return Path(split_manifest)
    dataset_name = safe_filename(Path(dataset_dir).resolve().name)
    return Path("splits") / f"{dataset_name}_seed{seed}.json"


def _split_ratios(train_ratio: float, val_ratio: float) -> dict[str, float]:
    """Validate and return train, validation, and test ratios.

    Parameters:
        train_ratio: Requested training fraction.
        val_ratio: Requested validation fraction.

    Returns:
        All three named split ratios.
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 12)
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train, validation, and test ratios must all be positive")
    return {
        "train": float(train_ratio),
        "validation": float(val_ratio),
        "test": float(test_ratio),
    }


def _dataset_inventory(dataset: AHARDataset) -> list[dict[str, object]]:
    """Return a stable, relative-path inventory for a dataset.

    Parameters:
        dataset: Dataset whose clip paths and labels should be inventoried.

    Returns:
        Sorted sample records with paths and class labels.
    """
    dataset_root = dataset.dataset_dir.resolve()
    inventory: list[dict[str, object]] = []
    for sample_path, class_index in dataset.samples:
        try:
            relative_path = sample_path.resolve().relative_to(dataset_root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"sample is outside the dataset root: {sample_path}"
            ) from error
        inventory.append(
            {
                "path": relative_path,
                "class_index": class_index,
                "class_name": dataset.class_names[class_index],
            }
        )
    inventory.sort(key=lambda sample: str(sample["path"]))
    paths = [str(sample["path"]) for sample in inventory]
    if len(paths) != len(set(paths)):
        raise ValueError("dataset inventory contains duplicate sample paths")
    return inventory


def _inventory_hash(class_names: list[str], inventory: list[dict[str, object]]) -> str:
    """Hash the stable class mapping and sample inventory.

    Parameters:
        class_names: Ordered dataset class names.
        inventory: Stable sample records without split membership.

    Returns:
        A SHA-256 hexadecimal digest.
    """
    payload = json.dumps(
        {"class_names": class_names, "samples": inventory},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_counts(
    samples: list[dict[str, object]], class_names: list[str]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Count total and per-class samples for each split.

    Parameters:
        samples: Manifest sample records with split membership.
        class_names: Ordered class names.

    Returns:
        Overall counts followed by per-class counts.
    """
    split_names = ("train", "validation", "test")
    counts = {name: 0 for name in split_names}
    counts["total"] = len(samples)
    class_counts = {
        class_name: {"total": 0, **{name: 0 for name in split_names}}
        for class_name in class_names
    }
    for sample in samples:
        split_name = str(sample["split"])
        class_name = str(sample["class_name"])
        counts[split_name] += 1
        class_counts[class_name][split_name] += 1
        class_counts[class_name]["total"] += 1
    return counts, class_counts


def create_split_manifest(
    dataset: AHARDataset,
    manifest_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, object]:
    """Create and atomically save one stratified clip-level split manifest.

    Parameters:
        dataset: Dataset whose clips should be split.
        manifest_path: Destination JSON path.
        train_ratio: Requested training fraction.
        val_ratio: Requested validation fraction.
        seed: Random seed used by stratified splitting.

    Returns:
        The created manifest.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ratios = _split_ratios(train_ratio, val_ratio)
    inventory = _dataset_inventory(dataset)
    if not inventory:
        raise ValueError("cannot split an empty dataset")
    labels = [int(sample["class_index"]) for sample in inventory]
    try:
        train_samples, remaining_samples = train_test_split(
            inventory,
            train_size=ratios["train"],
            random_state=seed,
            stratify=labels,
        )
        remaining_ratio = ratios["validation"] + ratios["test"]
        validation_fraction = ratios["validation"] / remaining_ratio
        validation_samples, test_samples = train_test_split(
            remaining_samples,
            train_size=validation_fraction,
            random_state=seed,
            stratify=[int(sample["class_index"]) for sample in remaining_samples],
        )
    except ValueError as error:
        raise ValueError(f"stratified split failed: {error}") from error

    membership = {
        str(sample["path"]): split_name
        for split_name, split_samples in (
            ("train", train_samples),
            ("validation", validation_samples),
            ("test", test_samples),
        )
        for sample in split_samples
    }
    samples = [
        {**sample, "split": membership[str(sample["path"])]} for sample in inventory
    ]
    counts, class_counts = _split_counts(samples, dataset.class_names)
    if any(
        class_counts[class_name][split_name] == 0
        for class_name in dataset.class_names
        for split_name in ("train", "validation", "test")
    ):
        raise ValueError("every class must be represented in every split")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_name": dataset.dataset_dir.resolve().name,
        "split_level": "clip",
        "seed": seed,
        "requested_ratios": ratios,
        "class_names": dataset.class_names,
        "inventory_hash": _inventory_hash(dataset.class_names, inventory),
        "counts": counts,
        "class_counts": class_counts,
        "samples": samples,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_split_subsets(
    dataset: AHARDataset,
    manifest_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset, dict[str, object]]:
    """Create or validate a manifest and rebuild its dataset subsets.

    Parameters:
        dataset: Dataset to match against the manifest.
        manifest_path: Manifest to create once or reuse.
        train_ratio: Required training fraction.
        val_ratio: Required validation fraction.
        seed: Required split seed.

    Returns:
        Train, validation, and test subsets followed by split metadata.
    """
    path = Path(manifest_path)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ratios = _split_ratios(train_ratio, val_ratio)
    if not path.exists():
        create_split_manifest(dataset, path, train_ratio, val_ratio, seed)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"cannot read split manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("split manifest has an unsupported schema")
    if manifest.get("split_level") != "clip":
        raise ValueError("split manifest must describe a clip-level split")
    if manifest.get("dataset_name") != dataset.dataset_dir.resolve().name:
        raise ValueError("split manifest dataset name does not match the dataset")
    if manifest.get("seed") != seed:
        raise ValueError("split manifest seed does not match the required split seed")
    stored_ratios = manifest.get("requested_ratios")
    try:
        ratios_match = isinstance(stored_ratios, dict) and all(
            abs(float(stored_ratios.get(name, -1.0)) - ratio) <= 1e-12
            for name, ratio in ratios.items()
        )
    except (TypeError, ValueError):
        ratios_match = False
    if not ratios_match:
        raise ValueError("split manifest ratios do not match the requested ratios")
    if manifest.get("class_names") != dataset.class_names:
        raise ValueError("split manifest class mapping does not match the dataset")

    inventory = _dataset_inventory(dataset)
    expected_hash = _inventory_hash(dataset.class_names, inventory)
    if manifest.get("inventory_hash") != expected_hash:
        raise ValueError("split manifest inventory hash does not match the dataset")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("split manifest samples must be a list")
    manifest_paths = [
        str(sample.get("path")) if isinstance(sample, dict) else ""
        for sample in raw_samples
    ]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("split manifest contains duplicate sample paths")
    inventory_by_path = {str(sample["path"]): sample for sample in inventory}
    if set(manifest_paths) != set(inventory_by_path):
        raise ValueError("split manifest omits or adds dataset samples")

    allowed_splits = {"train", "validation", "test"}
    indices_by_split: dict[str, list[int]] = {name: [] for name in allowed_splits}
    dataset_index_by_path = {
        str(
            sample_path.resolve().relative_to(dataset.dataset_dir.resolve()).as_posix()
        ): index
        for index, (sample_path, _) in enumerate(dataset.samples)
    }
    for sample in raw_samples:
        if not isinstance(sample, dict):
            raise ValueError("split manifest sample entries must be objects")
        sample_path = str(sample.get("path"))
        current = inventory_by_path[sample_path]
        if (
            sample.get("class_index") != current["class_index"]
            or sample.get("class_name") != current["class_name"]
        ):
            raise ValueError(f"split manifest relabels sample: {sample_path}")
        split_name = sample.get("split")
        if split_name not in allowed_splits:
            raise ValueError(f"invalid split for sample: {sample_path}")
        indices_by_split[str(split_name)].append(dataset_index_by_path[sample_path])

    counts, class_counts = _split_counts(raw_samples, dataset.class_names)
    if manifest.get("counts") != counts or manifest.get("class_counts") != class_counts:
        raise ValueError("split manifest count summaries do not match its samples")
    if any(
        class_counts[class_name][split_name] == 0
        for class_name in dataset.class_names
        for split_name in allowed_splits
    ):
        raise ValueError("every class must be represented in every split")
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata: dict[str, object] = {
        "manifest_path": str(path.resolve()),
        "manifest_hash": manifest_hash,
        "inventory_hash": expected_hash,
        "seed": seed,
        "requested_ratios": ratios,
        "counts": counts,
        "class_counts": class_counts,
        "split_level": "clip",
    }
    return (
        Subset(dataset, indices_by_split["train"]),
        Subset(dataset, indices_by_split["validation"]),
        Subset(dataset, indices_by_split["test"]),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default="datasets/raw/abnormal_activities"
    )
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Use CachedAHARDataset")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Save a paired online-augmented preview beside each clean clip",
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "dataset_samples" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    DatasetClass = CachedAHARDataset if args.cache else AHARDataset
    dataset = DatasetClass(
        args.dataset_dir, args.sequence_length, (args.height, args.width)
    )
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    augmentation = VideoAugmentation() if args.augment else None
    preview_label = "clean/augmented pairs" if augmentation else "clean clips"
    print(f"\n🎬 Saving {len(indices)} {preview_label} → {out_dir}")
    for idx in indices:
        video, label = dataset[idx]
        stem = Path(dataset.samples[idx][0]).stem
        name = f"{stem}_class-{dataset.class_names[label]}"
        clean_name = f"{name}_clean.mp4"
        write_video_torchvision(video, out_dir / clean_name, fps=args.fps)
        print(f"  ✅ {clean_name}")
        if augmentation:
            augmented_name = f"{name}_augmented.mp4"
            write_video_torchvision(
                augmentation(video),
                out_dir / augmented_name,
                fps=args.fps,
            )
            print(f"  ✅ {augmented_name}")


class AugmentSubset(torch.utils.data.Dataset):
    """Apply one online video transform without changing subset length."""

    def __init__(
        self,
        subset: Dataset,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Initialize the online augmentation wrapper.

        Parameters:
            subset: Underlying training subset to wrap.
            transform: Full-video transform applied once per sample access.

        Returns:
            None
        """
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        """Return the unchanged number of underlying samples.

        Parameters:
            None

        Returns:
            length (int): Total sample count.
        """
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return one freshly augmented view of an underlying sample.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            sample (tuple): A tuple containing the transformed video tensor and its label.
        """
        video, label = self.subset[idx]
        return self.transform(video), label


if __name__ == "__main__":
    main()
