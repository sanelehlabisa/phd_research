"""Shared configuration for the modular Paper 001 notebooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticDataset:
    """One explicitly declared diagnostic dataset."""

    key: str
    kaggle_handle: str
    accepted_classes: tuple[str, ...]
    enabled: bool = True


# Change this one value to switch every modular diagnostic notebook.
SELECTED_DIAGNOSTIC_DATASET = "kinetics-subset"

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
    "non-violent",
    "violent",
    "headbutting",
    "slapping",
    "punching_person__boxing_",
    "hugging",
    "shaking_hands",
)

DIAGNOSTIC_DATASETS = {
    "vdd": DiagnosticDataset(
        key="vdd",
        kaggle_handle="sanelehlabisa/violence-detection-dataset",
        accepted_classes=("non-violent", "violent"),
    ),
    "kinetics-subset": DiagnosticDataset(
        key="kinetics-subset",
        kaggle_handle="sanelehlabisa/kinetics-400-dataset/versions/1",
        # Resolved from CLASSES_OF_INTEREST and the actual folder inventory.
        accepted_classes=(),
    ),
}

CONTROLLED_AAD_PLAN = "configs/aad_controlled_experiment_plan.json"
CONTROLLED_AAD_DATASET_HANDLE = "sanelehlabisa/abnormal-activities-dataset"

# Bounded exploratory budgets, not the paper's controlled AAD protocol.
SEQUENCE_LENGTH = 32
TARGET_FPS = 8
FRAME_SIZE = 64
FINAL_FRAME_SIZE = 96
TRAIN_EPOCHS = 32
SCREEN_EPOCHS = 24
FINAL_EPOCHS = 64
BATCH_SIZE = 8
SEED = 42
DEFAULT_LAYERS = ((8, (3, 3)), (16, (3, 3)))
SCREEN_CANDIDATES = {
    "width_8": ((8, (3, 3)),),
    "depth_8_16": DEFAULT_LAYERS,
    "depth_8_8_8": ((8, (3, 3)), (8, (3, 3)), (8, (3, 3))),
}

# Diagnostic-only controls. Controlled AAD configuration remains unchanged.
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
    if not dataset.enabled or len(set(requested)) < 2:
        raise ValueError(
            f"diagnostic dataset {dataset.key!r} is not ready; declare at least two "
            "exact accepted classes and enable it first"
        )
    return dataset
