"""Validated configuration shared by training and comparison runners."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Store the reproducible controls used by one experiment run."""

    dataset_dir: str = "datasets/abnormal_activities"
    runs_dir: str = "runs"
    split_manifest: str | None = None
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    sequence_length: int = 32
    height: int = 128
    width: int = 128
    convlstm_layers: tuple[tuple[int, tuple[int, int]], ...] = ((8, (3, 3)),)
    hidden_classifier_width: int | None = None
    epochs: int = 16
    early_stopping_patience: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    augment: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    optimizer: str = "adam"
    loss: str = "cross_entropy"
    scheduler: str = "reduce_on_plateau"

    def __post_init__(self) -> None:
        """Validate every field immediately after construction."""
        self.validate()

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any],
        defaults: ExperimentConfig | None = None,
    ) -> ExperimentConfig:
        """Build a validated configuration from a complete or partial mapping."""
        known_fields = {item.name for item in fields(cls)}
        unknown_fields = sorted(set(values) - known_fields)
        if unknown_fields:
            raise ValueError(
                "unknown configuration field(s): " + ", ".join(unknown_fields)
            )

        resolved = defaults.to_dict() if defaults is not None else {}
        resolved.update(values)
        if "convlstm_layers" in resolved:
            resolved["convlstm_layers"] = _normalise_layers(resolved["convlstm_layers"])
        return cls(**resolved)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        defaults: ExperimentConfig | None = None,
    ) -> ExperimentConfig:
        """Load and validate a JSON configuration object."""
        config_path = Path(path)
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {config_path}: {error.msg}") from error
        if not isinstance(values, dict):
            raise ValueError(f"configuration in {config_path} must be a JSON object")
        return cls.from_mapping(values, defaults=defaults)

    def to_dict(self) -> dict[str, object]:
        """Return the complete configuration as JSON-compatible values."""
        values = asdict(self)
        values["convlstm_layers"] = [
            [filters, [kernel_height, kernel_width]]
            for filters, (kernel_height, kernel_width) in self.convlstm_layers
        ]
        return values

    def to_json(self) -> str:
        """Return deterministic, human-readable JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save_json(self, path: str | Path) -> Path:
        """Save deterministic JSON and return its path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{self.to_json()}\n", encoding="utf-8")
        return destination

    def validate(self) -> None:
        """Reject invalid or unsupported experiment settings."""
        for field_name in ("dataset_dir", "runs_dir"):
            _require_non_empty_string(field_name, getattr(self, field_name))
        if self.split_manifest is not None:
            _require_non_empty_string("split_manifest", self.split_manifest)

        for field_name in ("split_seed", "seed", "num_workers"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.split_seed != 42:
            raise ValueError("split_seed must be 42 for the committed AAD split")

        for field_name in ("train_ratio", "val_ratio", "test_ratio"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
                or value >= 1.0
            ):
                raise ValueError(f"{field_name} must be between 0 and 1")
        expected_ratios = (0.7, 0.15, 0.15)
        actual_ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if not all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in zip(actual_ratios, expected_ratios)
        ):
            raise ValueError(
                "train_ratio, val_ratio, and test_ratio must be 0.7, 0.15, and "
                "0.15 for the committed split"
            )

        positive_integers = (
            "sequence_length",
            "height",
            "width",
            "epochs",
            "early_stopping_patience",
            "batch_size",
        )
        for field_name in positive_integers:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.hidden_classifier_width is not None and (
            not isinstance(self.hidden_classifier_width, int)
            or isinstance(self.hidden_classifier_width, bool)
            or self.hidden_classifier_width <= 0
        ):
            raise ValueError(
                "hidden_classifier_width must be null or a positive integer"
            )

        _validate_layers(self.convlstm_layers)
        for field_name in ("augment", "pin_memory"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")

        if (
            not isinstance(self.learning_rate, (int, float))
            or isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be greater than 0")
        if (
            not isinstance(self.weight_decay, (int, float))
            or isinstance(self.weight_decay, bool)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError("weight_decay must be at least 0")

        fixed_protocol = {
            "optimizer": (self.optimizer, "adam"),
            "loss": (self.loss, "cross_entropy"),
        }
        for field_name, (actual, expected) in fixed_protocol.items():
            if actual != expected:
                raise ValueError(f"{field_name} must be {expected}")
        if self.scheduler not in {"none", "reduce_on_plateau"}:
            raise ValueError("scheduler must be none or reduce_on_plateau")


def _require_non_empty_string(field_name: str, value: object) -> None:
    """Validate one required string field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _normalise_layers(value: object) -> tuple[tuple[int, tuple[int, int]], ...]:
    """Convert JSON or CLI layer values into immutable layer tuples."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("convlstm_layers must contain at least one layer")
    normalised: list[tuple[int, tuple[int, int]]] = []
    for index, raw_layer in enumerate(value):
        field_name = f"convlstm_layers[{index}]"
        if not isinstance(raw_layer, (list, tuple)) or len(raw_layer) != 2:
            raise ValueError(
                f"{field_name} must be [filters, [kernel_height, kernel_width]]"
            )
        filters, raw_kernel = raw_layer
        if not isinstance(raw_kernel, (list, tuple)) or len(raw_kernel) != 2:
            raise ValueError(f"{field_name}.kernel_size must contain height and width")
        normalised.append((filters, (raw_kernel[0], raw_kernel[1])))
    return tuple(normalised)


def _validate_layers(
    layers: tuple[tuple[int, tuple[int, int]], ...],
) -> None:
    """Validate all configured ConvLSTM layers."""
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("convlstm_layers must contain at least one layer")
    for index, layer in enumerate(layers):
        field_name = f"convlstm_layers[{index}]"
        if not isinstance(layer, tuple) or len(layer) != 2:
            raise ValueError(
                f"{field_name} must be (filters, (kernel_height, kernel_width))"
            )
        filters, kernel = layer
        if not isinstance(filters, int) or isinstance(filters, bool) or filters <= 0:
            raise ValueError(f"{field_name}.filters must be a positive integer")
        if not isinstance(kernel, tuple) or len(kernel) != 2:
            raise ValueError(f"{field_name}.kernel_size must contain two integers")
        for dimension_name, dimension in zip(("height", "width"), kernel):
            if (
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                or dimension % 2 == 0
            ):
                raise ValueError(
                    f"{field_name}.kernel_{dimension_name} must be a positive odd "
                    "integer"
                )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared explicit configuration options to a runner parser."""
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--print-config",
        "--print_config",
        dest="print_config",
        action="store_true",
        help="Print the resolved configuration without starting a run",
    )
    for field_name in ("dataset_dir", "runs_dir", "split_manifest"):
        parser.add_argument(f"--{field_name}", type=str)
    for field_name in (
        "split_seed",
        "seed",
        "sequence_length",
        "height",
        "width",
        "epochs",
        "early_stopping_patience",
        "batch_size",
        "num_workers",
    ):
        parser.add_argument(f"--{field_name}", type=int)
    parser.add_argument(
        "--hidden-classifier-width",
        "--hidden_classifier_width",
        dest="hidden_classifier_width",
        type=int,
    )
    for field_name in (
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "learning_rate",
        "weight_decay",
    ):
        parser.add_argument(f"--{field_name}", type=float)
    parser.add_argument(
        "--convlstm-layer",
        dest="convlstm_layers",
        action="append",
        nargs=3,
        type=int,
        metavar=("FILTERS", "KERNEL_HEIGHT", "KERNEL_WIDTH"),
        help="Repeat to replace the complete configured CustomConvLSTM stack",
    )
    _add_boolean_pair(parser, "augment", "--augment", "--no-augment")
    _add_boolean_pair(
        parser,
        "pin_memory",
        "--pin_memory",
        "--pin-memory",
        "--no-pin-memory",
    )


def _add_boolean_pair(
    parser: argparse.ArgumentParser,
    destination: str,
    positive_option: str,
    *other_options: str,
) -> None:
    """Add explicit true and false flags without parser defaults."""
    negative_options = tuple(
        option for option in other_options if option.startswith("--no-")
    )
    positive_options = (positive_option,) + tuple(
        option for option in other_options if not option.startswith("--no-")
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        *positive_options,
        dest=destination,
        action="store_true",
        default=argparse.SUPPRESS,
    )
    group.add_argument(
        *negative_options,
        dest=destination,
        action="store_false",
        default=argparse.SUPPRESS,
    )


def resolve_config_arguments(
    parsed_values: dict[str, object],
    defaults: ExperimentConfig,
) -> tuple[ExperimentConfig, dict[str, object], bool, bool]:
    """Resolve defaults, optional JSON, and only explicit CLI overrides."""
    values = dict(parsed_values)
    config_path = values.pop("config", None)
    print_config = bool(values.pop("print_config", False))
    known_fields = {item.name for item in fields(ExperimentConfig)}
    overrides = {
        field_name: values.pop(field_name)
        for field_name in tuple(values)
        if field_name in known_fields
    }
    if "convlstm_layers" in overrides:
        overrides["convlstm_layers"] = [
            [layer[0], [layer[1], layer[2]]] for layer in overrides["convlstm_layers"]
        ]
    base = (
        ExperimentConfig.from_json(config_path, defaults=defaults)
        if config_path is not None
        else defaults
    )
    resolved_values = base.to_dict()
    resolved_values.update(overrides)
    resolved = ExperimentConfig.from_mapping(resolved_values)
    return resolved, values, print_config, config_path is not None
