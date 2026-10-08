"""Shared configuration for the modular Paper 001 notebooks."""

from __future__ import annotations

from dataclasses import dataclass

from src.dataset_source import AAD_CLASS_NAMES, AAD_DATASET_NAME, AAD_KAGGLE_HANDLE


@dataclass(frozen=True)
class DiagnosticDataset:
    """One explicitly declared diagnostic dataset."""

    key: str
    kaggle_handle: str
    accepted_classes: tuple[str, ...]
    enabled: bool = True
    source_url: str = ""


# Change this one value to switch every modular diagnostic notebook.
SELECTED_DIAGNOSTIC_DATASET = "kinetics600-subset"

KINETICS600_SOURCE_URL = "https://s3.amazonaws.com/kinetics/600/train"
KINETICS600_CLASSES = (
    "headbutting",
    "slapping",
    "punching person (boxing)",
    "hugging (not baby)",
    "shaking hands",
)

# Exact folder names across our datasets. Only Kinetics uses this intersection;
# absent interests are reported and skipped, never mapped to another activity.
CLASSES_OF_INTEREST = (
    "Begging",
    "Drunkenness",
    "Fight",
    "Harassment",
    "Hijack",
    "Knife Hazard",
    "Normal Videos",
    "Pollution",
    "Property Damage",
    "Robbery",
    "Terrorism",
    "headbutting",
    "slapping",
    "punching_person__boxing_",
    "hugging",
    "shaking_hands",
)

DIAGNOSTIC_DATASETS = {
    AAD_DATASET_NAME: DiagnosticDataset(
        key=AAD_DATASET_NAME,
        kaggle_handle=AAD_KAGGLE_HANDLE,
        accepted_classes=AAD_CLASS_NAMES,
    ),
    "kinetics600-subset": DiagnosticDataset(
        key="kinetics600-subset",
        kaggle_handle="",  # This release uses direct class archives, not Kaggle.
        accepted_classes=KINETICS600_CLASSES,
        source_url=KINETICS600_SOURCE_URL,
    ),
    "vdd": DiagnosticDataset(
        key="vdd",
        kaggle_handle="sanelehlabisa/violence-detection-dataset",
        accepted_classes=(),
    ),
    "kinetics-subset": DiagnosticDataset(
        key="kinetics-subset",
        kaggle_handle="sanelehlabisa/kinetics-400-dataset/versions/1",
        # Resolved from CLASSES_OF_INTEREST and the actual folder inventory.
        accepted_classes=(),
    ),
}

CONTROLLED_AAD_PLAN = "configs/experiments/aad_controlled_experiment_plan.json"
CONTROLLED_AAD_DATASET_HANDLE = "sanelehlabisa/abnormal-activities-dataset"

# Bounded exploratory budgets, not the paper's controlled AAD protocol.
SEQUENCE_LENGTH = 16
TARGET_FPS = 16
FRAME_SIZE = 32
# Notebook 03 only; never mutate the preview or SUITE_* settings to train it.
TRAIN_EPOCHS = 200
TRAIN_HOURS = 8
TRAIN_SEQUENCE_LENGTH = 32
TRAIN_TARGET_FPS = 8
TRAIN_FRAME_SIZE = 96
TRAIN_BATCH_SIZE = 8
TRAIN_LAYERS = ((32, (3, 3)), (64, (3, 3)), (64, (3, 3)))
SCREEN_EPOCHS = 4
FINAL_EPOCHS = 8
# Retained legacy quick-grid API only; notebook 04 now uses SUITE_* below.
SCREEN_BATCH_SIZE = 4
# Every architecture is tried with every combination of these finite lists.
# These lists do not configure the expanded notebook 04 suite.
SCREEN_FRAME_SIZES = [64, 128]
SCREEN_SEQUENCE_LENGTHS = [16]
SCREEN_WEIGHT_DECAYS = [0.0, 0.0001]
SCREEN_AUGMENT_OPTIONS = [True]
BATCH_SIZE = 8
SEED = 42
DEFAULT_LAYERS = ((8, (3, 3)), (16, (3, 3)))
SCREEN_CANDIDATES = {
    "width_8": ((8, (3, 3)),),
    "width_16": ((16, (3, 3)),),
    "width_32": ((32, (3, 3)),),
    "depth_8_16": DEFAULT_LAYERS,
    "depth_16_8": ((16, (3, 3)), (8, (3, 3))),
    "depth_32_8": ((32, (3, 3)), (8, (3, 3))),
    "depth_64_8": ((64, (3, 3)), (8, (3, 3))),
    "depth_8_8_8": ((8, (3, 3)), (8, (3, 3)), (8, (3, 3))),
    "wide_beginning": ((32, (3, 3)), (8, (3, 3)), (8, (3, 3))),
    "wide_middle": ((8, (3, 3)), (32, (3, 3)), (8, (3, 3))),
    "wide_end": ((8, (3, 3)), (8, (3, 3)), (32, (3, 3))),
    "depth_64_16_8": ((64, (3, 3)), (16, (3, 3)), (8, (3, 3))),
    "depth_8_16_32": ((8, (3, 3)), (16, (3, 3)), (32, (3, 3))),
    "depth_32_16_8": ((32, (3, 3)), (16, (3, 3)), (8, (3, 3))),
    "depth_64_32_16": ((64, (3, 3)), (32, (3, 3)), (16, (3, 3))),
    "depth_8_8_8_8": ((8, (3, 3)), (8, (3, 3)), (8, (3, 3)), (8, (3, 3))),
}
# Keep the full model catalogue above; choose a small list for this quick check.
SCREEN_MODEL_NAMES = ["width_8", "depth_8_16", "depth_16_8"]

# Active notebook 04 suite. Preview/single-model and controlled AAD stay separate.
SUITE_SEQUENCE_LENGTH = 32
SUITE_TARGET_FPS = 8
SUITE_FRAME_SIZE = 64
SUITE_SPATIAL_SIZES = (64, 96, 128)
SUITE_SCREEN_EPOCHS = 24
SUITE_FINAL_EPOCHS = 64
SUITE_FINAL_FRAME_SIZE = 96  # minimum; never downsize a selected larger input
NOTEBOOK_DROPOUT = 0.1
LR_FACTOR = 0.5
LR_PATIENCE = 5  # reduction after six consecutive non-improving epochs
MIN_LR = 1e-5
SUITE_HOURS = 8
PAPER_MAX_MINUTES = 30
CONFIRMATION_SEEDS = (42, 2026)
TEMPORAL_FPS = (4, 8, 16)
TINY_PER_CLASS = 2
TINY_STEPS = 512


def selected_diagnostic_dataset() -> DiagnosticDataset:
    """Return the selected, runnable diagnostic dataset declaration."""
    try:
        dataset = DIAGNOSTIC_DATASETS[SELECTED_DIAGNOSTIC_DATASET]
    except KeyError as error:
        raise ValueError(
            f"unknown diagnostic dataset: {SELECTED_DIAGNOSTIC_DATASET}"
        ) from error
    requested = (
        CLASSES_OF_INTEREST
        if dataset.key == "kinetics-subset"
        else dataset.accepted_classes
    )
    if not dataset.enabled or (requested and len(set(requested)) < 2):
        raise ValueError(
            f"diagnostic dataset {dataset.key!r} is not ready; declare at least two "
            "exact accepted classes when filtering, and enable it first"
        )
    return dataset
