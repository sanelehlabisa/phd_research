"""Local inference for the frozen AAD architecture, with explicit UI-test mode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .dataset_source import AAD_CLASS_NAMES
from .model import CustomConvLSTM, custom_model_from_checkpoint
from .temporal_sampling import load_timestamp_clip, video_plan

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
BUNDLED_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "demo_assets/winner/best_model.pth"
)
BUNDLED_METADATA = BUNDLED_CHECKPOINT.with_name("config.json")
ORIGINAL_CHECKPOINT_SHA256 = (
    "fd21edfc7058efaf39c432fb92318e768f22bef3b0103cc42f049a5e8ff6e896"
)
WINNER_LAYERS = [[64, [3, 3]], [64, [3, 3]], [128, [3, 3]]]
WINNER_INPUT = {
    "sequence_length": 8,
    "height": 64,
    "width": 64,
    "target_fps": 16,
    "sampling_version": "timestamps_v1",
}
RANDOM_WARNING = (
    "UNTRAINED RANDOM WEIGHTS — UI smoke test only; predictions are not research "
    "results and do not demonstrate recognition accuracy."
)
WINDOW_NOTE = (
    "Only the initial eight-frame window at 16 FPS is classified "
    "(0–0.4375 seconds), not the whole video. Short videos repeat their last "
    "frame. Probabilities are not calibrated reliability."
)


def _read_metadata(metadata, checkpoint: Path) -> dict:
    if isinstance(metadata, dict):
        return metadata
    if metadata is None:
        # Experiment runs: run/models/name/checkpoints/best_model.pth.
        # Standalone runs: run/checkpoints/best_model.pth.
        candidates = [
            checkpoint.parents[index] / "config.json"
            for index in (3, 1)
            if len(checkpoint.parents) > index
        ]
        metadata = next((path for path in candidates if path.is_file()), None)
        if metadata is None:
            raise ValueError(
                "Matching run config.json is missing. Supply metadata containing "
                "ordered class_names and experiment_config."
            )
    try:
        values = json.loads(Path(metadata).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("Could not read the demo metadata JSON.") from error
    if not isinstance(values, dict):
        raise ValueError("Demo metadata must be a JSON object.")
    return values


def _check_recipe(values, label: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{label} experiment_config is missing.")
    expected = {
        **WINNER_INPUT,
        "convlstm_layers": WINNER_LAYERS,
        "hidden_classifier_width": None,
    }
    for key, value in expected.items():
        if key not in values or values[key] != value:
            raise ValueError(f"{label} {key} does not match the frozen demo recipe.")


def _check_model(values) -> None:
    expected = {
        "model_name": "CustomConvLSTM",
        "num_classes": len(AAD_CLASS_NAMES),
        "input_channels": 3,
        "layers": [
            {"filters": width, "kernel_size": kernel} for width, kernel in WINNER_LAYERS
        ],
        "dropout": 0.5,
        "hidden_classifier_width": None,
    }
    if not isinstance(values, dict) or any(
        key not in values or values[key] != value for key, value in expected.items()
    ):
        raise ValueError(
            "Checkpoint architecture must be CustomConvLSTM[64,64,128], "
            "3x3 kernels, RGB, eleven classes and no hidden classifier."
        )


class DemoPredictor:
    """Load once; classify uploaded clips without loading datasets or evaluating test.

    Random weights are permitted only when explicitly selected for UI testing.
    Trained inference requires a compatible checkpoint and its ordered metadata.
    """

    def __init__(
        self, checkpoint=None, metadata=None, random_weights=False, device="auto"
    ):
        self.class_names = list(AAD_CLASS_NAMES)
        self.checkpoint_path = None
        self.checkpoint_sha256 = None
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError(
                "Choose device 'auto', 'cpu' or an available 'cuda'."
            ) from error
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("The local demo supports CPU or CUDA.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu.")

        if random_weights:
            if checkpoint is not None or metadata is not None:
                raise ValueError(
                    "Random UI-test mode cannot be combined with trained checkpoint metadata."
                )
            self.mode = "random"
            self.status = RANDOM_WARNING
            # Reproducible UI smoke weights without changing a caller's RNG state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                self.model = CustomConvLSTM(
                    num_classes=len(self.class_names),
                    layers=[(width, tuple(kernel)) for width, kernel in WINNER_LAYERS],
                    hidden_classifier_width=None,
                )
        else:
            if checkpoint is None:
                raise ValueError(
                    "A trained checkpoint is required. Use explicit --random-weights "
                    "only to test the interface; there is no random fallback."
                )
            self.checkpoint_path = Path(checkpoint).expanduser().resolve()
            if not self.checkpoint_path.is_file():
                if self.checkpoint_path == BUNDLED_CHECKPOINT.resolve():
                    raise FileNotFoundError(
                        "Bundled demo weights are missing. From the repository run "
                        "'git lfs install' then 'git lfs pull', and relaunch the demo. "
                        "No random fallback."
                    )
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}. No random fallback."
                )
            with self.checkpoint_path.open("rb") as handle:
                if handle.read(128).startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                ):
                    raise ValueError(
                        "This checkpoint is a Git LFS pointer, not model weights. "
                        "From the repository run 'git lfs install' then 'git lfs pull', "
                        "and relaunch the demo. No random fallback."
                    )
            details = _read_metadata(metadata, self.checkpoint_path)
            if details.get("class_names") != self.class_names:
                raise ValueError(
                    "Metadata class names/order must match the eleven AAD classes."
                )
            _check_recipe(details.get("experiment_config"), "Metadata")
            digest = hashlib.sha256()
            with self.checkpoint_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.checkpoint_sha256 = digest.hexdigest()
            expected_hash = details.get("checkpoint_sha256")
            if expected_hash is not None and expected_hash != self.checkpoint_sha256:
                raise ValueError("Checkpoint SHA256 does not match its metadata.")
            try:
                saved = torch.load(
                    self.checkpoint_path, map_location="cpu", weights_only=True
                )
            except Exception as error:
                raise ValueError(
                    "Could not safely load the checkpoint weights."
                ) from error
            if not isinstance(saved, dict):
                raise ValueError(
                    "Checkpoint must contain model and experiment metadata."
                )
            _check_model(saved.get("model_config"))
            _check_recipe(saved.get("experiment_config"), "Checkpoint")
            if (
                saved.get("checkpoint_role") != "validation_selected_lowest_loss"
                or saved.get("selection_partition") != "validation"
            ):
                raise ValueError("Checkpoint must be selected using validation loss.")
            if "class_names" in saved and saved["class_names"] != self.class_names:
                raise ValueError(
                    "Checkpoint class names/order differ from its metadata."
                )
            split = details.get("split", {})
            if isinstance(split, dict) and split.get("manifest_hash") is not None:
                if split["manifest_hash"] != saved.get("split_manifest_hash"):
                    raise ValueError("Checkpoint and metadata split hashes differ.")
            if "model" in details:
                _check_model(details["model"])
            try:
                self.model = custom_model_from_checkpoint(saved)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    "Checkpoint tensors do not match the frozen architecture."
                ) from error
            self.mode = "trained"
            if self.checkpoint_sha256 == ORIGINAL_CHECKPOINT_SHA256:
                self.status = (
                    "Original validation-selected October 16 checkpoint verified."
                )
            else:
                self.status = (
                    "Compatible trained checkpoint loaded; this is not the verified "
                    "original October 16 checkpoint and does not inherit its reported score."
                )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, video_path) -> dict:
        """Return ordered class probabilities and the exact initial-window sampling."""
        if video_path is None or not isinstance(video_path, (str, Path)):
            raise ValueError("Choose a local sample or upload a video first.")
        path = Path(video_path).expanduser()
        if not path.is_file():
            raise ValueError("The selected video does not exist.")
        size = path.stat().st_size
        if size == 0:
            raise ValueError("The selected video is empty.")
        if size > MAX_UPLOAD_BYTES:
            raise ValueError("The video exceeds the 200 MiB local-demo file limit.")
        try:
            plan = video_plan(
                path, WINNER_INPUT["target_fps"], WINNER_INPUT["sequence_length"]
            )
            frames = load_timestamp_clip(path, plan, (64, 64))
        except Exception as error:
            raise ValueError(
                "Cannot decode this video with valid presentation timestamps. "
                "Choose a playable video with a video track."
            ) from error
        if frames.shape != (8, 3, 64, 64) or not torch.isfinite(frames).all():
            raise ValueError("Decoded video does not match the required model input.")
        logits = self.model(frames.unsqueeze(0).to(self.device))
        if (
            logits.shape != (1, len(self.class_names))
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("The model returned invalid class scores.")
        scores = torch.softmax(logits[0], dim=0).cpu().tolist()
        probabilities = dict(zip(self.class_names, scores))
        return {
            "probabilities": probabilities,
            "prediction": self.class_names[
                max(range(len(scores)), key=scores.__getitem__)
            ],
            "status": f"{self.status}\n\n{WINDOW_NOTE}",
            "sampling": dict(plan),
        }
