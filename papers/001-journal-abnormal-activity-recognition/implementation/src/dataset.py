"""
dataset.py

Video and frames dataset classes for AHAR.

Author: Sanele Hlabisa

python -m src.dataset \
    --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
    --augment \
    --num_samples 4
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from .utils import read_video_torchvision, write_video_torchvision, TARGET_FPS


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
    parser.add_argument("--cache", action="store_true", help="Use CachedAHARDataset")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Save a paired online-augmented preview beside each clean clip",
    )
    args = parser.parse_args()

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
