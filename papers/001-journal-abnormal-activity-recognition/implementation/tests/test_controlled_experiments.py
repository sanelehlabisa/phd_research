"""Focused checks for the controlled AAD experiment suite."""

import json
from pathlib import Path
import shutil

import pytest

from src.experiment_config import CandidateManifest, ExperimentConfig
from src.experiments import (
    build_plan_commands,
    load_controlled_plan,
    main,
    model_registry,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CANDIDATE_PATH = CONFIG_DIR / "aad_architecture_candidates.json"
PLAN_PATH = CONFIG_DIR / "aad_controlled_experiment_plan.json"


def test_candidate_manifest_has_exact_approved_order() -> None:
    """Keep the eleven approved custom stacks in their declared order."""
    manifest = CandidateManifest.from_json(CANDIDATE_PATH)
    expected = [
        ("custom_reference_8", (8,)),
        ("custom_two_layer_8_16", (8, 16)),
        ("custom_two_layer_16_8", (16, 8)),
        ("custom_two_layer_32_8", (32, 8)),
        ("custom_two_layer_64_8", (64, 8)),
        ("custom_depth_8_8_8", (8, 8, 8)),
        ("custom_capacity_early", (16, 8, 8)),
        ("custom_capacity_middle", (8, 16, 8)),
        ("custom_capacity_late", (8, 8, 16)),
        ("custom_funnel_64_16_8", (64, 16, 8)),
        ("custom_selected_64_32_16", (64, 32, 16)),
    ]
    actual = [
        (
            candidate.name,
            tuple(filters for filters, _ in candidate.convlstm_layers),
        )
        for candidate in manifest.candidates
    ]
    assert actual == expected
    assert all(
        kernel == (3, 3)
        for candidate in manifest.candidates
        for _, kernel in candidate.convlstm_layers
    )


def test_candidate_manifest_rejects_duplicate_question() -> None:
    """Do not allow two candidates to claim the same research question."""
    values = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    values["candidates"][1]["research_question"] = values["candidates"][0][
        "research_question"
    ]
    with pytest.raises(ValueError, match="research questions must be unique"):
        CandidateManifest.from_mapping(values)


def test_candidate_manifest_rejects_duplicate_name() -> None:
    """Do not allow two architecture entries to share one model name."""
    values = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    values["candidates"][1]["name"] = values["candidates"][0]["name"]
    with pytest.raises(ValueError, match="duplicate candidate name"):
        CandidateManifest.from_mapping(values)


def _copy_plan_files(tmp_path: Path) -> Path:
    """Copy controlled-plan inputs to one isolated implementation layout."""
    config_dir = tmp_path / "implementation" / "configs"
    config_dir.mkdir(parents=True)
    for filename in (
        "aad_architecture_candidates.json",
        "aad_screening_reference.json",
        "aad_confirmation_reference.json",
        "aad_paper_topology_reference.json",
        "aad_controlled_experiment_plan.json",
    ):
        shutil.copy(CONFIG_DIR / filename, config_dir / filename)
    return config_dir / "aad_controlled_experiment_plan.json"


def test_plan_has_one_factor_trials_and_fixed_learning_rate() -> None:
    """Validate approved factors, protocols, and exact generated run counts."""
    plan, manifest = load_controlled_plan(PLAN_PATH)
    assert plan["reference_candidate"] == "custom_depth_8_8_8"
    assert len(manifest.candidates) == 11
    all_commands = [
        *build_plan_commands(PLAN_PATH, "architecture-screen"),
        *build_plan_commands(PLAN_PATH, "baseline-confirmation"),
        *build_plan_commands(PLAN_PATH, "published-topology"),
    ]
    assert len(all_commands) == 4
    ablation_commands = build_plan_commands(PLAN_PATH, "focused-ablations")
    assert len(ablation_commands) == 12
    assert all(
        "src.experiments" in command and "src.evaluate" not in command
        for _, command in [*all_commands, *ablation_commands]
    )
    for _, command in ablation_commands:
        assert "--learning_rate" not in command
        assert "--epochs" not in command

    reference = ExperimentConfig.from_json(
        CONFIG_DIR / "aad_confirmation_reference.json"
    )
    assert reference.learning_rate == 0.001
    assert reference.epochs == 64


def test_plan_rejects_multi_factor_trial(tmp_path: Path) -> None:
    """Reject a trial that changes augmentation and weight decay together."""
    plan_path = _copy_plan_files(tmp_path)
    values = json.loads(plan_path.read_text(encoding="utf-8"))
    values["ablations"]["trials"][3]["overrides"] = {
        "augment": False,
        "weight_decay": 0.0,
    }
    plan_path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly its approved factor"):
        load_controlled_plan(plan_path)


def test_confirmation_registry_has_selected_custom_and_three_baselines() -> None:
    """Keep the comparable confirmation registry focused and explicit."""
    manifest = CandidateManifest.from_json(CANDIDATE_PATH)
    registry = model_registry(
        candidate_manifest=manifest,
        confirmation_candidate="custom_depth_8_8_8",
    )
    assert [entry["name"] for entry in registry] == [
        "custom_depth_8_8_8",
        "r3d_18",
        "mc3_18",
        "r2plus1d_18",
    ]


def test_published_model_rejects_reduced_input_before_dataset_access() -> None:
    """Never silently train the published topology on the screening input."""
    with pytest.raises(ValueError, match="native 50-frame 50x50 protocol"):
        main(
            [
                "--config",
                str(CONFIG_DIR / "aad_screening_reference.json"),
                "--model",
                "paper_convlstm_published",
            ]
        )
