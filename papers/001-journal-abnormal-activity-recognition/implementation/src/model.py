"""
model.py

ConvLSTM model for AHAR. CNN extracts spatial features, LSTM models temporal dynamics.

Author: Sanele Hlabisa

python -m src.model \
    --dataset_dir "datasets/processed/videos_abnormal_activities" \
    --model_dir "models"
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .dataset import AHARDataset


KernelSize: TypeAlias = int | tuple[int, int]
LayerSpec: TypeAlias = tuple[int, tuple[int, int]]


def _normalise_kernel_size(
    kernel_size: KernelSize, *, field_name: str = "kernel_size"
) -> tuple[int, int]:
    """Validate a 2D odd kernel and return it as ``(height, width)``."""
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
        raise ValueError(
            f"{field_name} dimensions must be odd so same padding preserves spatial size"
        )
    return kernel


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``model``."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class ConvLSTM2DCell(nn.Module):
    """
    Implements a single ConvLSTM2D cell for processing one timestep.
    """

    def __init__(
        self, in_channels: int, filters: int, kernel_size: KernelSize = 3
    ) -> None:
        """
        Initializes the ConvLSTM2D cell with combined gate convolutions.

        Parameters:
            in_channels (int): Number of channels in the input tensor.
            filters (int): Number of output filters for the hidden state.
            kernel_size (int | tuple[int, int]): Size of the convolutional kernel.

        Returns:
            None
        """
        super().__init__()
        kernel = _normalise_kernel_size(kernel_size)
        pad = tuple(value // 2 for value in kernel)

        # Gates: input, forget, cell, output - all in one conv for efficiency
        # Input comes from x_t and h_{t-1} concatenated on channel dim
        self.conv = nn.Conv2d(
            in_channels + filters,
            filters * 4,  # i, f, g, o gates
            kernel_size=kernel,
            padding=pad,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, C, H, W)
        h: torch.Tensor,  # (B, filters, H, W)
        c: torch.Tensor,  # (B, filters, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the next hidden and cell states for the current timestep.

        Parameters:
            x (torch.Tensor): Input tensor for the current timestep of shape (B, C, H, W).
            h (torch.Tensor): Previous hidden state tensor of shape (B, filters, H, W).
            c (torch.Tensor): Previous cell state tensor of shape (B, filters, H, W).

        Returns:
            states (tuple[torch.Tensor, torch.Tensor]): A tuple containing the new hidden state and new cell state.
        """

        combined = torch.cat([x, h], dim=1)  # (B, C+filters, H, W)
        gates: torch.Tensor = self.conv(combined)  # (B, filters*4, H, W)

        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM2D(nn.Module):
    """
    Applies a ConvLSTM2D cell sequentially over an entire temporal dimension.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: KernelSize = 3,
        return_sequences: bool = False,
    ) -> None:
        """
        Initializes the sequential ConvLSTM module.

        Parameters:
            in_channels (int): Number of channels in the input frames.
            filters (int): Number of output filters for the hidden state.
            kernel_size (int | tuple[int, int]): Size of the convolutional kernel.
            return_sequences (bool): Return every hidden state when true; otherwise
                return only the final hidden state.

        Returns:
            None
        """
        super().__init__()
        self.filters = filters
        self.return_sequences = return_sequences
        self.cell = ConvLSTM2DCell(in_channels, filters, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence and returns all or only the final hidden state.

        Parameters:
            x (torch.Tensor): Input sequence tensor of shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Hidden states shaped (B, T, filters, H, W) when
                ``return_sequences`` is true, otherwise (B, filters, H, W).
        """
        if x.ndim != 5:
            raise ValueError("x must have shape (B, T, C, H, W)")

        B, T, C, H, W = x.shape
        if T == 0:
            raise ValueError("x must contain at least one timestep")

        h = x.new_zeros(B, self.filters, H, W)
        c = x.new_zeros(B, self.filters, H, W)
        outputs: list[torch.Tensor] = []

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        return h


# ============================================================
# Models
# ============================================================


class ConvLSTMOriginal(nn.Module):
    """
    Legacy approximation of the source paper's ConvLSTM architecture.

    This class collapses time before its post-convolution and dense head. It is
    retained for checkpoint and import compatibility; use
    :class:`OriginalPaperConvLSTM` for the faithful structural baseline.
    """

    def __init__(self, num_classes: int, input_shape: tuple = (3, 64, 64)) -> None:
        """
        Initializes the large baseline model and prints its parameter count.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).

        Returns:
            None
        """
        super().__init__()
        C, H, W = input_shape
        self.td_conv = nn.Conv2d(C, 16, 3, padding=1)
        self.convlstm = ConvLSTM2D(16, 64)
        self.bn = nn.BatchNorm2d(64)
        self.conv_post = nn.Conv2d(64, 16, 3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * H * W, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape
        x = F.relu(self.td_conv(x.view(B * T, C, H, W))).view(B, T, 16, H, W)
        x = self.bn(self.convlstm(x))
        x = self.dropout1(F.relu(self.conv_post(x)))
        return self.fc2(self.dropout2(F.relu(self.fc1(self.flatten(x)))))


class OriginalPaperConvLSTM(nn.Module):
    """Structural PyTorch reproduction of Vršková et al.'s proposed model.

    The published tensor shapes require ``padding="same"`` and a ConvLSTM that
    returns all 50 timesteps, even though those settings are not fully stated in
    the prose. The paper also does not completely specify convolution
    activations, ConvLSTM gate initialisation, or how Keras batch normalisation
    should translate to channel-first PyTorch. This implementation therefore
    adds no unreported ReLU layers, retains this project's existing ConvLSTM
    gate equations, and applies ``BatchNorm2d`` over the combined batch/time
    dimension. It returns logits; the training loss is responsible for the
    classification normalisation rather than embedding a softmax in the model.

    The default ``50 x 50 x 50`` flattened feature volume makes this baseline
    intentionally large. Instantiate it on PyTorch's ``meta`` device when only
    structural metadata or parameter counts are required.
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
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.td_conv = nn.Conv2d(channels, 16, (3, 3), padding=(1, 1))
        self.convlstm = ConvLSTM2D(16, 64, kernel_size=(3, 3), return_sequences=True)
        self.bn = nn.BatchNorm2d(64)
        self.conv_post = nn.Conv2d(64, 16, (3, 3), padding=(1, 1))
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(sequence_length * 16 * height * width, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for an input shaped ``(B, T, C, H, W)``."""
        if x.ndim != 5:
            raise ValueError("x must have shape (B, T, C, H, W)")

        batch, timesteps, channels, height, width = x.shape
        expected = (self.sequence_length, *self.input_shape)
        actual = (timesteps, channels, height, width)
        if actual != expected:
            raise ValueError(
                "x has sequence/frame shape "
                f"{actual}; expected {expected} for the configured dense head"
            )

        x = self.td_conv(x.reshape(batch * timesteps, channels, height, width))
        x = x.reshape(batch, timesteps, 16, height, width)
        x = self.convlstm(x)
        x = self.bn(x.reshape(batch * timesteps, 64, height, width))
        x = x.reshape(batch, timesteps, 64, height, width)
        x = self.conv_post(x.reshape(batch * timesteps, 64, height, width))
        x = x.reshape(batch, timesteps, 16, height, width)
        x = self.dropout1(x)
        x = self.fc1(self.flatten(x))
        return self.fc2(self.dropout2(x))


def _validate_layer_specs(convlstm_layers: object) -> tuple[LayerSpec, ...]:
    """Validate and normalise explicit stacked-ConvLSTM layer specifications."""
    if not isinstance(convlstm_layers, (list, tuple)) or not convlstm_layers:
        raise ValueError("convlstm_layers must contain at least one layer")

    validated: list[LayerSpec] = []
    for index, spec in enumerate(convlstm_layers):
        field = f"convlstm_layers[{index}]"
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError(
                f"{field} must be (filters, (kernel_height, kernel_width))"
            )

        filters, kernel_size = spec
        if not isinstance(filters, int) or isinstance(filters, bool) or filters <= 0:
            raise ValueError(f"{field}.filters must be a positive integer")
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError(f"{field}.kernel_size must be a pair of integers")

        kernel = _normalise_kernel_size(kernel_size, field_name=f"{field}.kernel_size")
        validated.append((filters, kernel))
    return tuple(validated)


class StackedConvLSTM(nn.Module):
    """Configurable ConvLSTM stack with a resolution-independent pooled head."""

    def __init__(
        self,
        num_classes: int,
        convlstm_layers: list[LayerSpec] | tuple[LayerSpec, ...],
        input_shape: tuple[int, int, int] = (3, 64, 64),
        dropout: float = 0.5,
        hidden_classifier_width: int | None = None,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in input_shape
        ):
            raise ValueError("input_shape must contain positive integers (C, H, W)")
        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError("dropout must be a number in [0, 1)")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if hidden_classifier_width is not None and (
            not isinstance(hidden_classifier_width, int)
            or isinstance(hidden_classifier_width, bool)
            or hidden_classifier_width <= 0
        ):
            raise ValueError(
                "hidden_classifier_width must be a positive integer or None"
            )

        self.layer_specs = _validate_layer_specs(convlstm_layers)
        recurrent_layers: list[ConvLSTM2D] = []
        in_channels = input_shape[0]
        for index, (filters, kernel_size) in enumerate(self.layer_specs):
            recurrent_layers.append(
                ConvLSTM2D(
                    in_channels,
                    filters,
                    kernel_size,
                    return_sequences=index < len(self.layer_specs) - 1,
                )
            )
            in_channels = filters

        self.convlstm_layers = nn.ModuleList(recurrent_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dropout = nn.Dropout(float(dropout))
        self.hidden_classifier: nn.Module | None
        self.classifier_dropout: nn.Module | None
        if hidden_classifier_width is None:
            self.hidden_classifier = None
            self.classifier_dropout = None
            self.classifier = nn.Linear(in_channels, num_classes)
        else:
            self.hidden_classifier = nn.Linear(in_channels, hidden_classifier_width)
            self.classifier_dropout = nn.Dropout(float(dropout))
            self.classifier = nn.Linear(hidden_classifier_width, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final layer's spatial representation before pooling."""
        features = x
        for layer in self.convlstm_layers:
            features = layer(features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits shaped ``(B, num_classes)``."""
        features = self.forward_features(x)
        features = self.feature_dropout(self.pool(features).flatten(1))
        if self.hidden_classifier is not None:
            features = F.relu(self.hidden_classifier(features))
            if self.classifier_dropout is not None:
                features = self.classifier_dropout(features)
        return self.classifier(features)


class ConvLSTMPooledModel(nn.Module):
    """
    Lightweight ConvLSTM variant using adaptive pooling and dropout to reduce overfitting.
    """

    def __init__(self, num_classes: int, input_shape: tuple = (3, 64, 64)) -> None:
        """
        Initializes the pooled model architecture with dropout regularization.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).

        Returns:
            None
        """
        super().__init__()

        C, H, W = input_shape

        self.td_conv = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM2D(32, 32)
        self.bn = nn.BatchNorm2d(32)

        self.conv_post = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

        total = count_trainable_parameters(self)

        print(f"ConvLSTMPooledModel initialized with {total:,} parameters.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape

        x = self.td_conv(x.view(B * T, C, H, W)).view(B, T, 32, H, W)
        x = self.convlstm(x)
        x = self.bn(x)
        x = self.conv_post(x)
        x = self.pool(x)

        x = x.view(B, -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class ConvLSTMPooledModelV1(nn.Module):
    """
    Higher-capacity variant of the pooled ConvLSTM with increased filter sizes and neurons.
    """

    def __init__(self, num_classes: int, input_shape: tuple = (3, 64, 64)) -> None:
        """
        Initializes the higher-capacity pooled model architecture.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).

        Returns:
            None
        """
        super().__init__()

        C, H, W = input_shape

        self.td_conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM2D(64, 64)
        self.bn = nn.BatchNorm2d(64)

        self.conv_post = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        total = count_trainable_parameters(self)

        print(f"ConvLSTMPooledModelV1 initialized with {total:,} parameters.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape

        x = self.td_conv(x.view(B * T, C, H, W)).view(B, T, 64, H, W)
        x = self.convlstm(x)
        x = self.bn(x)
        x = self.conv_post(x)
        x = self.pool(x)

        x = x.view(B, -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class ConvLSTMModel(nn.Module):
    """
    Ultra-lightweight ConvLSTM variant with reduced filter sizes.
    """

    def __init__(self, num_classes: int, input_shape: tuple = (3, 64, 64)) -> None:
        """
        Initializes the lightweight model architecture and prints its parameter count.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).

        Returns:
            None
        """
        super().__init__()
        C, H, W = input_shape
        self.td_conv = nn.Conv2d(C, 2, 3, padding=1)
        self.convlstm = ConvLSTM2D(2, 4)
        self.bn = nn.BatchNorm2d(4)
        self.conv_post = nn.Conv2d(4, 2, 3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * H * W, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)

        total = count_trainable_parameters(self)
        print(f"🧠 ConvLSTMModel (light) | params={total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape
        x = F.relu(self.td_conv(x.view(B * T, C, H, W))).view(B, T, 2, H, W)
        x = self.bn(self.convlstm(x))
        x = self.dropout1(F.relu(self.conv_post(x)))
        return self.fc2(self.dropout2(F.relu(self.fc1(self.flatten(x)))))


class ConvLSTMCustom(nn.Module):
    """
    Customizable ConvLSTM architecture parameterized by a list of filter sizes.
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple = (3, 64, 64),
        filters: list[int] = [16, 64, 16, 256],
    ) -> None:
        """
        Initializes the custom model using the provided network dimensionalities.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).
            filters (list[int]): Four integers representing sizes for [td_conv, convlstm, conv_post, fc1].

        Returns:
            None
        """
        super().__init__()
        C, H, W = input_shape
        f_td, f_lstm, f_post, f_fc = filters

        self.td_conv = nn.Conv2d(C, f_td, 3, padding=1)
        self.convlstm = ConvLSTM2D(f_td, f_lstm)
        self.bn = nn.BatchNorm2d(f_lstm)
        self.conv_post = nn.Conv2d(f_lstm, f_post, 3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(f_post * H * W, f_fc)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(f_fc, num_classes)

        self._f_td = f_td  # Save for view reshaping in forward

        total = count_trainable_parameters(self)
        print(f"ConvLSTMCustom initialized with {total:,} parameters.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape
        x = F.relu(self.td_conv(x.view(B * T, C, H, W))).view(B, T, self._f_td, H, W)
        x = self.bn(self.convlstm(x))
        x = self.dropout1(F.relu(self.conv_post(x)))
        return self.fc2(self.dropout2(F.relu(self.fc1(self.flatten(x)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default="datasets/abnormal_activities"
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--original", action="store_true", help="Use original large model"
    )
    args = parser.parse_args()

    out_dir = Path("outputs") / "model_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.height, args.width)
    )

    ModelClass = ConvLSTMOriginal if args.original else ConvLSTMModel
    model = ModelClass(
        dataset.num_classes, input_shape=(3, args.height, args.width)
    ).to(device)

    total_params = count_trainable_parameters(model)
    print(f"🧠 Model: {ModelClass.__name__} | Params: {total_params:,}")

    if args.model_dir:
        from .utils import load_model

        model, _, epoch, loss = load_model(
            model,
            checkpoint_path=f"{args.model_dir}/best_model.pth",
            map_location=device,
        )
        print(f"📂 Loaded - epoch={epoch}, loss={loss:.4f}")
    else:
        print("⚠️  No model_dir - random weights (shape check only)")

    idx = random.randint(0, len(dataset) - 1)
    frames, true_label = dataset[idx]

    model.eval()
    with torch.no_grad():
        logits = model(frames.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = probs.argmax().item()
        pred_conf = probs[pred_label].item()

    true_name = dataset.class_names[true_label]
    pred_name = dataset.class_names[pred_label]

    print(f"\n✅ Input : {frames.shape}  Logits: {logits.shape}")
    print(f"🔹 True  : {true_name}")
    print(f"🔹 Pred  : {pred_name}  ({pred_conf:.1%})")
    print(f"🔹 Probs : {np.round(probs.cpu().numpy(), 2)}")

    stem = Path(dataset.samples[idx][0]).stem
    correct = "correct" if pred_label == true_label else "wrong"
    fname = f"{stem}_true-{true_name}_pred-{pred_name}_{correct}.mp4"
    clip = (frames * 255).byte().permute(0, 2, 3, 1).cpu()
    torchvision.io.write_video(str(out_dir / fname), clip, fps=args.fps)
    print(f"\n🎬 Saved - {out_dir / fname}")


if __name__ == "__main__":
    main()
