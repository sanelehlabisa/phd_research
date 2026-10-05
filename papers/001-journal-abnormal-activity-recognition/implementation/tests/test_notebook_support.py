"""Tests for modular-notebook configuration and command delegation."""

from pathlib import Path
from dataclasses import replace

import pytest

from notebook_files import NOTEBOOKS
from src import notebook_config
from src import notebook_utils
from src.notebook_utils import (
    controlled_plan_list_command,
    controlled_stage_command,
    deterministic_subset_indices,
    diagnostic_manifest_path,
    final_test_evaluation_command,
    final_test_report,
    validation_screen_winner,
)


@pytest.mark.parametrize("visible_gpu", [False, True])
def test_cuda_failure_identifies_kernel_or_package_problem(monkeypatch, visible_gpu):
    import subprocess

    monkeypatch.setattr(notebook_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        notebook_utils.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0 if visible_gpu else 1,
            stdout="NVIDIA A100" if visible_gpu else "",
            stderr="No GPU" if not visible_gpu else "",
        ),
    )
    message = "Restart this notebook" if visible_gpu else "Select Kernel"
    with pytest.raises(RuntimeError, match=message) as error:
        notebook_utils.validate_runtime(require_cuda=True)
    assert "Python:" in str(error.value)
    assert "CUDA build:" in str(error.value)


def test_kinetics_is_the_default_diagnostic_dataset() -> None:
    selected = notebook_config.selected_diagnostic_dataset()
    assert selected.key == "kinetics600-subset"
    assert selected.source_url == "https://s3.amazonaws.com/kinetics/600/train"
    assert selected.accepted_classes == (
        "headbutting",
        "slapping",
        "punching person (boxing)",
        "hugging (not baby)",
        "shaking hands",
    )


def test_vdd_switch_preserves_its_classes(monkeypatch) -> None:
    monkeypatch.setattr(notebook_config, "SELECTED_DIAGNOSTIC_DATASET", "vdd")
    monkeypatch.setattr(notebook_config, "CLASSES_OF_INTEREST", ("missing",))
    selected = notebook_config.selected_diagnostic_dataset()
    assert selected.key == "vdd"
    assert selected.kaggle_handle == "sanelehlabisa/violence-detection-dataset"
    assert selected.accepted_classes == ("non-violent", "violent")


def test_disabled_dataset_cannot_be_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        notebook_config.DIAGNOSTIC_DATASETS,
        "kinetics600-subset",
        replace(
            notebook_config.DIAGNOSTIC_DATASETS["kinetics600-subset"], enabled=False
        ),
    )
    with pytest.raises(ValueError, match="not ready"):
        notebook_config.selected_diagnostic_dataset()


def test_all_modular_notebooks_check_for_the_shared_kinetics_loader() -> None:
    import json

    assert len(NOTEBOOKS) == 4
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "kinetics600_subset.py" in code


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


def test_five_indices_are_deterministic() -> None:
    class SubsetStub:
        indices = list(range(20))

    first = deterministic_subset_indices(SubsetStub(), count=5, seed=42)
    second = deterministic_subset_indices(SubsetStub(), count=5, seed=42)
    assert first == second
    assert len(first) == 5


def test_screen_winner_comes_from_complete_validation_ranking(tmp_path: Path) -> None:
    import json

    (tmp_path / "run.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    winner = {
        "name": "candidate-a",
        "partition": "validation",
        "selected_checkpoint": "best.pth",
        "validation_metrics": {"macro_f1": 0.8},
    }
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "ranked": [winner],
                "all": [winner],
                "candidate_manifest": {
                    "content": {"candidates": [{"name": "candidate-a"}]},
                    "sha256": "test",
                },
            }
        ),
        encoding="utf-8",
    )
    assert validation_screen_winner(tmp_path)["name"] == "candidate-a"


def test_final_test_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="OPEN FINAL AAD TEST"):
        final_test_evaluation_command(tmp_path, tmp_path / "model.pth", "")


def test_final_report_requires_exactly_five_test_examples(tmp_path: Path) -> None:
    import json

    (tmp_path / "metrics").mkdir()
    (tmp_path / "run.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (tmp_path / "metrics" / "final.json").write_text(
        json.dumps(
            {
                "partition": "test",
                "metrics": {"accuracy": 0.7},
                "checkpoint": {"path": "model.pth"},
                "artifacts": {"prediction_clips": [{"path": str(i)} for i in range(5)]},
            }
        ),
        encoding="utf-8",
    )
    report = final_test_report(tmp_path)
    assert report["partition"] == "test"
    assert len(report["examples"]) == 5
