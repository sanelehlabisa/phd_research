"""Tests for video decoding compatibility."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src import utils


class FakeFrame:
    """Provide one small decoded RGB frame."""

    def __init__(self, value: int) -> None:
        self.value = value

    def to_ndarray(self, format: str) -> np.ndarray:
        assert format == "rgb24"
        return np.full((2, 3, 3), self.value, dtype=np.uint8)


class FakeContainer:
    """Provide the PyAV container interface used by the reader."""

    def __init__(self) -> None:
        stream = SimpleNamespace(average_rate=Fraction(25, 1), guessed_rate=None)
        self.streams = SimpleNamespace(video=[stream])

    def __enter__(self) -> "FakeContainer":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def decode(self, stream: object) -> list[FakeFrame]:
        assert stream is self.streams.video[0]
        return [FakeFrame(0), FakeFrame(255)]


def test_read_video_uses_pyav_rgb_frames(monkeypatch) -> None:
    """Decode PyAV frames as a uint8 THWC tensor with the stream frame rate."""
    monkeypatch.setattr(utils.av, "open", lambda path: FakeContainer())

    frames, fps = utils.read_video_torchvision(Path("sample.mp4"))

    assert frames.shape == (2, 2, 3, 3)
    assert frames.dtype == torch.uint8
    assert frames[0].sum().item() == 0
    assert frames[1].min().item() == 255
    assert fps == 25.0


def test_write_and_read_video_with_pyav(tmp_path: Path) -> None:
    """Round-trip a small normalized RGB clip through the shared MP4 helpers."""
    source = torch.linspace(0.0, 1.0, steps=4 * 3 * 16 * 16).reshape(
        4, 3, 16, 16
    )
    output_path = tmp_path / "preview.mp4"

    utils.write_video_torchvision(source, output_path, fps=8)
    decoded, fps = utils.read_video_torchvision(output_path)

    assert output_path.stat().st_size > 0
    assert decoded.shape == (4, 16, 16, 3)
    assert decoded.dtype == torch.uint8
    assert fps == 8.0
