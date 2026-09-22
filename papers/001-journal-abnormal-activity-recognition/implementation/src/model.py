"""ConvLSTM layers, models, and a small prediction smoke command."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from .dataset import AHARDataset
from .utils import plot_confusion_matrix, write_video_torchvision


def _normalise_kernel_size(
    kernel_size: int | tuple[int, int], field_name: str = "kernel_size"
) -> tuple[int, int]:
    """Validate an odd two-dimensional kernel size.

    Parameters:
        kernel_size: One size for both dimensions or a `(height, width)` pair.
        field_name: Field name used in validation messages.

    Returns:
        The validated `(height, width)` pair.
    """
    if isinstance(kernel_size, bool):
        raise ValueError(f"{field_name} must be an integer or a pair of integers")
    if isinstance(kernel_size, int):
        kernel = (kernel_size, kernel_size)
    elif (
        isinstance(kernel_size, tuple)
        and len(kernel_size) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in kernel_size
        )
    ):
        kernel = kernel_size
    else:
        raise ValueError(f"{field_name} must be an integer or a pair of integers")
    if any(value <= 0 for value in kernel):
        raise ValueError(f"{field_name} dimensions must be positive")
    if any(value % 2 == 0 for value in kernel):
        raise ValueError(f"{field_name} dimensions must be odd")
    return kernel


def _validate_layers(
    layers: list[tuple[int, tuple[int, int]]],
) -> list[tuple[int, tuple[int, int]]]:
    """Validate a list of ConvLSTM layer settings.

    Parameters:
        layers: `(filters, (kernel_height, kernel_width))` entries.

    Returns:
        A validated copy of the layer list.
    """
    if not isinstance(layers, list) or not layers:
        raise ValueError("layers must contain at least one layer")
    validated: list[tuple[int, tuple[int, int]]] = []
    for index, layer in enumerate(layers):
        field = f"layers[{index}]"
        if not isinstance(layer, tuple) or len(layer) != 2:
            raise ValueError(
                f"{field} must be (filters, (kernel_height, kernel_width))"
            )
        filters, kernel_size = layer
        if not isinstance(filters, int) or isinstance(filters, bool) or filters <= 0:
            raise ValueError(f"{field}.filters must be a positive integer")
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError(f"{field}.kernel_size must be a pair of integers")
        kernel = _normalise_kernel_size(kernel_size, f"{field}.kernel_size")
        validated.append((filters, kernel))
    return validated


def parse_layer_arguments(
    values: list[list[int]] | None,
) -> list[tuple[int, tuple[int, int]]]:
    """Convert repeated command-line values into ConvLSTM layer settings.

    Parameters:
        values: Repeated `[filters, kernel_height, kernel_width]` values.

    Returns:
        Layer settings accepted by `CustomConvLSTM`.
    """
    if values is None:
        return [(8, (3, 3))]
    return [(value[0], (value[1], value[2])) for value in values]


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters in a model.

    Parameters:
        model: Model whose parameters will be counted.

    Returns:
        Number of trainable scalar parameters.
    """
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class _ConvLSTMCell(nn.Module):
    """Process one timestep with ConvLSTM gates.

    Parameters:
        in_channels: Channels in one input frame.
        filters: Channels in the hidden and cell states.
        kernel_size: Integer kernel size or `(height, width)` pair.

    Returns:
        A ConvLSTM cell module.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int | tuple[int, int] = 3,
    ) -> None:
        super().__init__()
        kernel = _normalise_kernel_size(kernel_size)
        padding = tuple(value // 2 for value in kernel)
        self.gates = nn.Conv2d(
            in_channels + filters,
            filters * 4,
            kernel_size=kernel,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the next hidden and cell states.

        Parameters:
            x: Current frame features shaped `(B, C, H, W)`.
            hidden: Previous hidden state shaped `(B, filters, H, W)`.
            cell: Previous cell state shaped `(B, filters, H, W)`.

        Returns:
            The next hidden and cell states.
        """
        gates = self.gates(torch.cat([x, hidden], dim=1))
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)
        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_cell)
        return next_hidden, next_cell


class ConvLSTM(nn.Module):
    """Process a video sequence with one ConvLSTM layer.

    Parameters:
        in_channels: Channels in each input frame.
        filters: Channels in the hidden state.
        kernel_size: Integer kernel size or `(height, width)` pair.
        return_sequences: Return every hidden state instead of only the last.

    Returns:
        A ConvLSTM sequence layer.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int | tuple[int, int] = 3,
        return_sequences: bool = False,
    ) -> None:
        super().__init__()
        self.filters = filters
        self.return_sequences = return_sequences
        self.cell = _ConvLSTMCell(in_channels, filters, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ConvLSTM over all timesteps.

        Parameters:
            x: Video features shaped `(B, T, C, H, W)`.

        Returns:
            All hidden states `(B, T, filters, H, W)` or the final state
            `(B, filters, H, W)`.
        """
        if x.ndim != 5:
            raise ValueError("x must have shape (B, T, C, H, W)")
        batch, timesteps, _, height, width = x.shape
        if timesteps == 0:
            raise ValueError("x must contain at least one timestep")
        hidden = x.new_zeros(batch, self.filters, height, width)
        cell = x.new_zeros(batch, self.filters, height, width)
        outputs: list[torch.Tensor] = []
        for timestep in range(timesteps):
            hidden, cell = self.cell(x[:, timestep], hidden, cell)
            if self.return_sequences:
                outputs.append(hidden)
        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        return hidden


class PaperConvLSTM(nn.Module):
    """Reproduce the published ConvLSTM topology without simplifying its head.

    Parameters:
        num_classes: Number of output classes.
        input_shape: Frame shape `(channels, height, width)`.
        sequence_length: Number of frames in each input sequence.

    Returns:
        The paper-topology classification model.

    Notes:
        Published shapes imply same padding and full sequence output. The paper
        does not fully report convolution activations, gate initialisation, or
        channel-first batch-normalisation details. This translation adds no
        unreported ReLU, keeps the project's ConvLSTM gates, normalises the
        combined batch/time dimension, and returns logits for the loss function.
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int] = (3, 50, 50),
        sequence_length: int = 50,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in input_shape
        ):
            raise ValueError("input_shape must contain positive integers (C, H, W)")
        if (
            not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length <= 0
        ):
            raise ValueError("sequence_length must be a positive integer")
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        channels, height, width = input_shape
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.frame_conv = nn.Conv2d(channels, 16, 3, padding=1)
        self.convlstm = ConvLSTM(16, 64, 3, return_sequences=True)
        self.batch_norm = nn.BatchNorm2d(64)
        self.post_conv = nn.Conv2d(64, 16, 3, padding=1)
        self.feature_dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(sequence_length * 16 * height * width, 256)
        self.hidden_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a video batch.

        Parameters:
            x: Videos shaped `(B, T, C, H, W)`.

        Returns:
            Class logits shaped `(B, num_classes)`.
        """
        if x.ndim != 5:
            raise ValueError("x must have shape (B, T, C, H, W)")
        batch, timesteps, channels, height, width = x.shape
        actual_shape = (timesteps, channels, height, width)
        expected_shape = (self.sequence_length, *self.input_shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"x has sequence/frame shape {actual_shape}; expected {expected_shape}"
            )
        x = self.frame_conv(x.reshape(batch * timesteps, channels, height, width))
        x = x.reshape(batch, timesteps, 16, height, width)
        x = self.convlstm(x)
        x = self.batch_norm(x.reshape(batch * timesteps, 64, height, width))
        x = x.reshape(batch, timesteps, 64, height, width)
        x = self.post_conv(x.reshape(batch * timesteps, 64, height, width))
        x = x.reshape(batch, timesteps, 16, height, width)
        x = self.feature_dropout(x)
        x = self.hidden(self.flatten(x))
        return self.classifier(self.hidden_dropout(x))


class CustomConvLSTM(nn.Module):
    """Build a stackable ConvLSTM classifier from an explicit layer list.

    Parameters:
        num_classes: Number of output classes.
        layers: `(filters, (kernel_height, kernel_width))` entries.
        input_channels: Channels in each input frame.
        dropout: Dropout probability used by the classifier head.
        hidden_classifier_width: Optional hidden width before classification.

    Returns:
        A configurable ConvLSTM classification model.
    """

    def __init__(
        self,
        num_classes: int,
        layers: list[tuple[int, tuple[int, int]]],
        input_channels: int = 3,
        dropout: float = 0.5,
        hidden_classifier_width: int | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        if (
            not isinstance(input_channels, int)
            or isinstance(input_channels, bool)
            or input_channels <= 0
        ):
            raise ValueError("input_channels must be a positive integer")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError("dropout must be a number in [0, 1)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if hidden_classifier_width is not None and (
            not isinstance(hidden_classifier_width, int)
            or isinstance(hidden_classifier_width, bool)
            or hidden_classifier_width <= 0
        ):
            raise ValueError("hidden_classifier_width must be positive or None")
        self.num_classes = num_classes
        self.layers = _validate_layers(layers)
        self.input_channels = input_channels
        self.dropout = float(dropout)
        self.hidden_classifier_width = hidden_classifier_width
        recurrent_layers: list[ConvLSTM] = []
        channels = input_channels
        for index, (filters, kernel_size) in enumerate(self.layers):
            recurrent_layers.append(
                ConvLSTM(
                    channels,
                    filters,
                    kernel_size,
                    return_sequences=index < len(self.layers) - 1,
                )
            )
            channels = filters
        self.recurrent_layers = nn.ModuleList(recurrent_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dropout = nn.Dropout(self.dropout)
        if hidden_classifier_width is None:
            self.hidden_classifier: nn.Linear | None = None
            self.hidden_dropout: nn.Dropout | None = None
            self.classifier = nn.Linear(channels, num_classes)
        else:
            self.hidden_classifier = nn.Linear(channels, hidden_classifier_width)
            self.hidden_dropout = nn.Dropout(self.dropout)
            self.classifier = nn.Linear(hidden_classifier_width, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final spatial feature map before pooling.

        Parameters:
            x: Videos shaped `(B, T, C, H, W)`.

        Returns:
            Final features shaped `(B, filters, H, W)`.
        """
        features = x
        for layer in self.recurrent_layers:
            features = layer(features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a video batch.

        Parameters:
            x: Videos shaped `(B, T, C, H, W)`.

        Returns:
            Class logits shaped `(B, num_classes)`.
        """
        features = self.forward_features(x)
        features = self.feature_dropout(self.pool(features).flatten(1))
        if self.hidden_classifier is not None:
            features = F.relu(self.hidden_classifier(features))
            if self.hidden_dropout is not None:
                features = self.hidden_dropout(features)
        return self.classifier(features)

    def configuration(self) -> dict[str, object]:
        """Return JSON-friendly settings that reconstruct this model.

        Parameters:
            None.

        Returns:
            The complete model configuration.
        """
        return {
            "model_name": self.__class__.__name__,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "layers": [
                {"filters": filters, "kernel_size": list(kernel_size)}
                for filters, kernel_size in self.layers
            ],
            "dropout": self.dropout,
            "hidden_classifier_width": self.hidden_classifier_width,
        }

    @classmethod
    def from_configuration(cls, configuration: dict[str, object]) -> CustomConvLSTM:
        """Rebuild a custom model from saved configuration data.

        Parameters:
            configuration: Configuration returned by `configuration()`.

        Returns:
            A reconstructed `CustomConvLSTM` model.
        """
        if configuration.get("model_name") != cls.__name__:
            raise ValueError("legacy or unsupported checkpoint model configuration")
        raw_layers = configuration.get("layers")
        if not isinstance(raw_layers, list):
            raise ValueError("checkpoint model configuration is missing layers")
        layers: list[tuple[int, tuple[int, int]]] = []
        for index, layer in enumerate(raw_layers):
            if not isinstance(layer, dict):
                raise ValueError(f"checkpoint layers[{index}] must be an object")
            filters = layer.get("filters")
            kernel = layer.get("kernel_size")
            if not isinstance(filters, int) or not isinstance(kernel, list):
                raise ValueError(f"checkpoint layers[{index}] is malformed")
            if len(kernel) != 2 or not all(isinstance(value, int) for value in kernel):
                raise ValueError(f"checkpoint layers[{index}].kernel_size is malformed")
            layers.append((filters, (kernel[0], kernel[1])))
        num_classes = configuration.get("num_classes")
        input_channels = configuration.get("input_channels", 3)
        dropout = configuration.get("dropout", 0.5)
        hidden_width = configuration.get("hidden_classifier_width")
        if not isinstance(num_classes, int) or not isinstance(input_channels, int):
            raise ValueError("checkpoint class or channel count is malformed")
        if not isinstance(dropout, (int, float)):
            raise ValueError("checkpoint dropout is malformed")
        if hidden_width is not None and not isinstance(hidden_width, int):
            raise ValueError("checkpoint hidden classifier width is malformed")
        return cls(
            num_classes=num_classes,
            layers=layers,
            input_channels=input_channels,
            dropout=float(dropout),
            hidden_classifier_width=hidden_width,
        )


def custom_model_from_checkpoint(checkpoint: dict[str, object]) -> CustomConvLSTM:
    """Reconstruct a custom model and load its saved weights.

    Parameters:
        checkpoint: Loaded checkpoint containing model configuration and weights.

    Returns:
        A configured model populated with saved weights.
    """
    configuration = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(configuration, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            "legacy checkpoint is incompatible: model_config and model_state_dict "
            "are required"
        )
    model = CustomConvLSTM.from_configuration(configuration)
    model.load_state_dict(state_dict)
    return model


def _safe_name(value: str) -> str:
    """Convert a label into a safe filename component.

    Parameters:
        value: Label or source name.

    Returns:
        A filesystem-safe name.
    """
    return "_".join(value.strip().replace("/", "_").split())


def _clear_model_samples(output_root: Path) -> None:
    """Remove only the fixed disposable model-sample directory.

    Parameters:
        output_root: Expected `outputs/model_samples` directory.

    Returns:
        None.
    """
    expected = (Path.cwd() / "outputs" / "model_samples").resolve()
    if output_root.resolve() != expected:
        raise ValueError(f"refusing to clear unexpected output path: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)


def _save_viewable_video(frames: torch.Tensor, path: Path, fps: int) -> None:
    """Upscale model frames and save them as a viewable MP4.

    Parameters:
        frames: Model input frames shaped `(T, C, H, W)` in `[0, 1]`.
        path: Destination MP4 path.
        fps: Output video frame rate.

    Returns:
        None.
    """
    view_frames = F.interpolate(
        frames, size=(256, 256), mode="bilinear", align_corners=False
    )
    write_video_torchvision(view_frames, path, fps)


def _build_parser() -> argparse.ArgumentParser:
    """Create the prediction-smoke command parser.

    Parameters:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Smoke-test both ConvLSTM models")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--convlstm-layer",
        action="append",
        nargs=3,
        type=int,
        metavar=("FILTERS", "KERNEL_HEIGHT", "KERNEL_WIDTH"),
    )
    parser.add_argument("--hidden-classifier-width", type=int, default=None)
    return parser


def main() -> None:
    """Run both random-weight models on the same real video samples.

    Parameters:
        None.

    Returns:
        None.
    """
    args = _build_parser().parse_args()
    if args.sequence_length != 64:
        raise ValueError("the ticket-011 smoke command requires T=64")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    dataset = AHARDataset(
        args.dataset_dir,
        sequence_length=args.sequence_length,
        frame_size=(args.height, args.width),
    )
    if not dataset.samples:
        raise ValueError("dataset contains no usable samples")
    sample_count = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), sample_count)
    samples = [(index, *dataset[index]) for index in indices]
    custom_layers = parse_layer_arguments(args.convlstm_layer)
    custom_model = CustomConvLSTM(
        dataset.num_classes,
        layers=custom_layers,
        hidden_classifier_width=args.hidden_classifier_width,
    )
    models: list[tuple[str, nn.Module, dict[str, object]]] = [
        (
            "paper_t64_topology_smoke",
            PaperConvLSTM(
                dataset.num_classes,
                input_shape=(3, args.height, args.width),
                sequence_length=args.sequence_length,
            ),
            {
                "model_name": "PaperConvLSTM",
                "input_shape": [3, args.height, args.width],
                "sequence_length": args.sequence_length,
                "faithful_default_input": False,
            },
        ),
        ("custom_convlstm", custom_model, custom_model.configuration()),
    ]
    output_root = Path("outputs") / "model_samples"
    _clear_model_samples(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = _safe_name(Path(args.dataset_dir).resolve().name)
    run_dir = output_root / timestamp / dataset_name
    run_dir.mkdir(parents=True)
    summary: dict[str, object] = {
        "warning": "Random-weight smoke predictions; not experimental evidence.",
        "dataset": str(Path(args.dataset_dir).resolve()),
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "seed": args.seed,
        "input": {
            "sequence_length": args.sequence_length,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
        },
        "class_names": dataset.class_names,
        "sample_indices": indices,
        "models": {},
    }
    model_summaries = summary["models"]
    if not isinstance(model_summaries, dict):
        raise RuntimeError("internal summary structure is invalid")
    print("Random-weight smoke predictions; not experimental evidence.")
    print(
        f"Input: ({sample_count}, {args.sequence_length}, 3, {args.height}, {args.width})"
    )
    for model_name, model, configuration in models:
        model = model.to(device).eval()
        model_dir = run_dir / model_name
        (model_dir / "correct").mkdir(parents=True)
        (model_dir / "incorrect").mkdir()
        true_labels: list[int] = []
        predicted_labels: list[int] = []
        prediction_records: list[dict[str, object]] = []
        with torch.inference_mode():
            for index, frames, true_label in samples:
                logits = model(frames.unsqueeze(0).to(device))
                predicted_label = int(logits.argmax(dim=1).item())
                is_correct = predicted_label == true_label
                true_name = dataset.class_names[true_label]
                predicted_name = dataset.class_names[predicted_label]
                source_stem = _safe_name(Path(dataset.samples[index][0]).stem)
                filename = (
                    f"{source_stem}_true-{_safe_name(true_name)}_"
                    f"pred-{_safe_name(predicted_name)}.mp4"
                )
                category = "correct" if is_correct else "incorrect"
                video_path = model_dir / category / filename
                _save_viewable_video(frames, video_path, args.fps)
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                prediction_records.append(
                    {
                        "sample_index": index,
                        "source": str(dataset.samples[index][0]),
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "correct": is_correct,
                        "logits_shape": list(logits.shape),
                        "video": str(video_path),
                    }
                )
                print(
                    f"{model_name}: {source_stem} -> {predicted_name} "
                    f"(true: {true_name}, logits: {tuple(logits.shape)})"
                )
        matrix = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=list(range(dataset.num_classes)),
        ).tolist()
        plot_confusion_matrix(
            true_labels,
            predicted_labels,
            dataset.class_names,
            dataset_name=model_name,
            save_path=str(model_dir / "confusion_matrix.png"),
        )
        parameter_count = count_trainable_parameters(model)
        model_summaries[model_name] = {
            "configuration": configuration,
            "trainable_parameters": parameter_count,
            "confusion_matrix": matrix,
            "predictions": prediction_records,
        }
        print(f"{model_name}: {parameter_count:,} trainable parameters")
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved smoke artifacts to {run_dir}")


if __name__ == "__main__":
    main()
