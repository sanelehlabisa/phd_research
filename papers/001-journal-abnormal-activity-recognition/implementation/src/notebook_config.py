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
SELECTED_DIAGNOSTIC_DATASET = "vdd"

DIAGNOSTIC_DATASETS = {
    "vdd": DiagnosticDataset(
        key="vdd",
        kaggle_handle="sanelehlabisa/violence-detection-dataset",
        accepted_classes=("non-violent", "violent"),
    ),
    # Ticket 036 must replace these placeholders before this entry is enabled.
    "kinetics-subset": DiagnosticDataset(
        key="kinetics-subset",
        kaggle_handle="sanelehlabisa/kinetics-400-dataset",
        accepted_classes=(),
        enabled=False,
    ),
}

CONTROLLED_AAD_PLAN = "configs/aad_controlled_experiment_plan.json"
CONTROLLED_AAD_DATASET_HANDLE = "sanelehlabisa/abnormal-activities-dataset"


def selected_diagnostic_dataset() -> DiagnosticDataset:
    """Return the selected, runnable diagnostic dataset declaration."""
    try:
        dataset = DIAGNOSTIC_DATASETS[SELECTED_DIAGNOSTIC_DATASET]
    except KeyError as error:
        raise ValueError(
            f"unknown diagnostic dataset: {SELECTED_DIAGNOSTIC_DATASET}"
        ) from error
    if not dataset.enabled or len(dataset.accepted_classes) < 2:
        raise ValueError(
            f"diagnostic dataset {dataset.key!r} is not ready; declare at least two "
            "exact accepted classes and enable it first"
        )
    return dataset
