"""Tests for modular-notebook configuration and command delegation."""

from pathlib import Path

import pytest

from src import notebook_config
from src.notebook_utils import (
    controlled_plan_list_command,
    controlled_stage_command,
    diagnostic_manifest_path,
)


def test_vdd_is_the_default_diagnostic_dataset() -> None:
    selected = notebook_config.selected_diagnostic_dataset()
    assert selected.key == "vdd"
    assert selected.kaggle_handle == "sanelehlabisa/violence-detection-dataset"
    assert selected.accepted_classes == ("non-violent", "violent")


def test_disabled_dataset_cannot_be_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        notebook_config, "SELECTED_DIAGNOSTIC_DATASET", "kinetics-subset"
    )
    with pytest.raises(ValueError, match="not ready"):
        notebook_config.selected_diagnostic_dataset()


def test_manifest_is_dataset_specific_and_ignored_location(tmp_path: Path) -> None:
    assert diagnostic_manifest_path(tmp_path, "vdd") == (
        tmp_path / "runs/manifests/vdd_diagnostic_seed42.json"
    )


def test_controlled_plan_command_is_fixed_to_aad(tmp_path: Path) -> None:
    command = controlled_plan_list_command(tmp_path)
    assert command[-1] == "--list-plan"
    assert "aad_controlled_experiment_plan.json" in command[-2]
    assert "vdd" not in " ".join(command).lower()


def test_architecture_screen_preserves_ticket_021_guard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ticket 021"):
        controlled_stage_command(tmp_path, "architecture-screen", "pending")
