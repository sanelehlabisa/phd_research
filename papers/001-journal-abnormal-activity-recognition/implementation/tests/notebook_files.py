"""Canonical workflows only; leave local notebook copies out of contract tests."""

from pathlib import Path

NOTEBOOKS = tuple(
    Path(__file__).resolve().parents[1] / "notebooks" / name
    for name in (
        "01_dataset_setup.ipynb",
        "02_model_inspection.ipynb",
        "03_train_model.ipynb",
        "04_run_experiments.ipynb",
    )
)
