"""Focused checks for the controlled AAD experiment suite."""

import json
from pathlib import Path
import shutil

import pytest

from src.experiment_config import CandidateManifest, ExperimentConfig
from src.experiments import (
    EXPERIMENT_DEFAULT_CONFIG,
    _resolved_arguments,
    build_experiment_criterion,
    build_plan_commands,
    controlled_plan_rows,
    execute_plan_stage,
    inspect_experiment_run_directories,
    load_controlled_plan,
    main,
    model_registry,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CANDIDATE_PATH = CONFIG_DIR / "experiments" / "aad_architecture_candidates.json"
PLAN_PATH = CONFIG_DIR / "experiments" / "aad_controlled_experiment_plan.json"


def test_candidate_manifest_has_exact_approved_order() -> None:
    """Keep the three approved custom stacks in their declared order."""
    manifest = CandidateManifest.from_json(CANDIDATE_PATH)
    expected = [
        ("custom_two_layer_32_16", (32, 16)),
        ("custom_two_layer_16_32", (16, 32)),
        ("custom_depth_32_16_8", (32, 16, 8)),
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


def test_runner_configs_are_grouped_by_owner() -> None:
    """Keep runner configs in clear subdirectories, not a flat config pile."""
    assert not list(CONFIG_DIR.glob("*.json"))
    assert {path.parent.name for path in CONFIG_DIR.glob("*/*.json")} == {
        "train",
        "evaluate",
        "experiments",
        "model",
    }


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
    for folder, filenames in {
        "experiments": (
            "aad_architecture_candidates.json",
            "aad_architecture_screen_reference.json",
            "aad_confirmation_reference.json",
            "aad_paper_topology_reference.json",
            "aad_controlled_experiment_plan.json",
        ),
        "train": ("aad_screening_reference.json",),
    }.items():
        target_dir = config_dir / folder
        target_dir.mkdir()
        for filename in filenames:
            shutil.copy(CONFIG_DIR / folder / filename, target_dir / filename)
    return config_dir / "experiments" / "aad_controlled_experiment_plan.json"


def test_plan_has_one_factor_trials_and_fixed_learning_rate() -> None:
    """Validate approved factors, protocols, and exact generated run counts."""
    plan, manifest = load_controlled_plan(PLAN_PATH)
    assert plan["reference_candidate"] == "custom_two_layer_32_16"
    assert plan["active_stage"] == "architecture-screen"
    assert len(manifest.candidates) == 3
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
        CONFIG_DIR / "experiments" / "aad_confirmation_reference.json"
    )
    assert reference.learning_rate == 0.001
    assert reference.epochs == 64


def test_plan_stage_runs_from_implementation_root(tmp_path, monkeypatch) -> None:
    """Child module commands need the implementation root on their import path."""
    plan_path = _copy_plan_files(tmp_path)
    implementation_root = plan_path.parents[2]
    calls = []
    monkeypatch.setattr(
        "src.experiments.subprocess.run",
        lambda command, cwd, check: calls.append((command, cwd, check)),
    )

    execute_plan_stage(plan_path, "architecture-screen", None, None, None)

    assert calls
    assert all(cwd == implementation_root for _, cwd, _ in calls)
    assert all(check for _, _, check in calls)


def test_architecture_screen_matches_historical_high_validation_profile() -> None:
    """Use the previously strongest validation run as the screen protocol."""
    plan, manifest = load_controlled_plan(PLAN_PATH)
    assert plan["screening"]["config"] == (
        "configs/experiments/aad_architecture_screen_reference.json"
    )
    screen_path = CONFIG_DIR / "experiments" / "aad_architecture_screen_reference.json"
    screen = ExperimentConfig.from_json(screen_path)

    assert screen_path.name == "aad_architecture_screen_reference.json"
    assert len(manifest.candidates) == 3
    assert screen.epochs == 160
    assert screen.batch_size == 16
    assert screen.sequence_length == 8
    assert (screen.height, screen.width) == (64, 64)
    assert screen.learning_rate == 0.01
    assert screen.weight_decay == 0.0
    assert screen.augment is False
    assert screen.optimizer == "adam"
    assert screen.scheduler == "reduce_on_plateau"
    assert screen.loss == "cross_entropy"
    assert screen.early_stopping_patience == 20
    assert screen.seed == screen.split_seed == 42
    assert (screen.train_ratio, screen.val_ratio, screen.test_ratio) == (
        0.7,
        0.15,
        0.15,
    )
    assert screen.split_manifest is None


def test_plan_config_alone_runs_its_active_stage(monkeypatch) -> None:
    """Use the stage in the plan without requiring extra execution flags."""
    commands: list[list[str]] = []

    def capture(command, cwd, check):
        commands.append(command)

    monkeypatch.setattr("src.experiments.subprocess.run", capture)
    main(["--plan-config", str(PLAN_PATH)])

    assert len(commands) == 1
    assert "--candidates-config" in commands[0]
    assert "aad_architecture_candidates.json" in commands[0][-1]
    assert "--model" not in commands[0]
    assert "--run-plan-stage" not in commands[0]


def test_plan_rows_expose_every_controlled_factor() -> None:
    """Provide a complete notebook table without allocating data or models."""
    rows = controlled_plan_rows(PLAN_PATH)

    assert len(rows) == 9
    assert sum(int(row["expected_runs"]) for row in rows) == 16
    assert rows[0]["stage"] == "architecture-screen"
    assert rows[0]["model_role"] == "3 custom candidates"
    by_trial = {str(row["trial"]): row for row in rows}
    assert by_trial["augmentation_off"]["augmentation"] is False
    assert by_trial["spatial_64"]["input_size"] == "T=16, 64x64"
    assert by_trial["sequence_32"]["input_size"] == "T=32, 32x32"
    assert by_trial["weight_decay_0"]["weight_decay"] == 0.0
    assert by_trial["weight_decay_0_0001"]["weight_decay"] == 0.0001


def test_experiment_loss_matches_plain_cross_entropy_config() -> None:
    """Do not silently add label smoothing to the controlled protocol."""
    criterion = build_experiment_criterion()

    assert criterion.label_smoothing == 0.0


def test_run_inspection_distinguishes_missing_partial_and_complete(
    tmp_path: Path,
) -> None:
    """Describe exact requested runs without treating partial output as evidence."""
    missing = tmp_path / "missing"
    partial = tmp_path / "partial"
    complete = tmp_path / "complete"
    partial.mkdir()
    complete.mkdir()
    (partial / "run.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    (complete / "run.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (complete / "summary.json").write_text(
        json.dumps({"ranked": [{"partition": "validation"}]}),
        encoding="utf-8",
    )

    records = inspect_experiment_run_directories([missing, partial, complete])

    assert [record["evidence_state"] for record in records] == [
        "missing",
        "partial",
        "complete",
    ]


def test_run_inspection_rejects_test_rankings(tmp_path: Path) -> None:
    """Never present test-ranked output as controlled selection evidence."""
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "run.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (invalid / "summary.json").write_text(
        json.dumps({"ranked": [{"partition": "test"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must rank validation results"):
        inspect_experiment_run_directories([invalid])


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
        confirmation_candidate="custom_two_layer_32_16",
    )
    assert [entry["name"] for entry in registry] == [
        "custom_two_layer_32_16",
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
                str(CONFIG_DIR / "train" / "aad_screening_reference.json"),
                "--model",
                "paper_convlstm_published",
            ]
        )


def test_swin_leaf_json_selects_one_model_without_default_baselines(capsys) -> None:
    """A single JSON command selects Swin only before loading any dataset."""
    path = CONFIG_DIR / "experiments" / "aad_swin3d_t_smoke.json"
    configured_rate = json.loads(path.read_text())["learning_rate"]
    args, config, candidates, print_only = _resolved_arguments({"config": path})
    entries = model_registry(config, candidates, selected_models=args.models)
    assert [entry["name"] for entry in entries] == ["swin3d_t"]
    assert candidates is None and not print_only
    assert config.dataset_name == "aad"
    assert config.sampling_version == "timestamps_v1"
    assert config.sequence_length == 8
    assert (config.height, config.width, config.target_fps) == (64, 64, 16)
    assert config.batch_size == 2 and config.epochs == 8
    assert config.learning_rate == configured_rate and config.weight_decay == 0
    assert config.scheduler == "none" and not config.augment
    assert config.seed == config.split_seed == 42
    main(["--config", str(path), "--list-models"])
    output = capsys.readouterr().out
    assert "swin3d_t" in output
    assert "swin3d_s" not in output
    assert "paper_convlstm_published" not in output
    assert "r3d_18" not in output


def test_leaf_json_models_allow_explicit_cli_model_and_training_overrides() -> None:
    """Explicit CLI values retain precedence over the JSON model selection."""
    path = CONFIG_DIR / "experiments" / "aad_swin3d_t_smoke.json"
    configured_rate = json.loads(path.read_text())["learning_rate"]
    args, config, _, _ = _resolved_arguments(
        {"config": path, "models": ["swin3d_s"], "epochs": 2}
    )
    assert args.models == ["swin3d_s"]
    assert config.epochs == 2
    assert config.batch_size == 2 and config.learning_rate == configured_rate
    assert json.loads(path.read_text())["models"] == ["swin3d_t"]


@pytest.mark.parametrize(
    "models",
    [[], "swin3d_t", None, ["swin3d_t", "swin3d_t"], [1], [{"name": "swin3d_t"}]],
)
def test_leaf_json_rejects_invalid_model_lists_before_training(
    models, tmp_path
) -> None:
    """Model lists must be nonempty, unique and contain only registry names."""
    path = tmp_path / "invalid-models.json"
    path.write_text(json.dumps({"models": models}))
    with pytest.raises(ValueError, match="non-empty list of unique model names"):
        _resolved_arguments({"config": path})


def test_leaf_json_models_do_not_hide_unknown_config_keys_or_model_names(
    tmp_path,
) -> None:
    """Optional models never bypass configuration or registry validation."""
    path = tmp_path / "unknown-fields.json"
    path.write_text(json.dumps({"models": ["swin3d_t"], "learning_rtae": 0.0001}))
    with pytest.raises(ValueError, match="unknown configuration field.*learning_rtae"):
        _resolved_arguments({"config": path})
    path.write_text(json.dumps({"models": ["not_registered"]}))
    args, config, _, _ = _resolved_arguments({"config": path})
    with pytest.raises(ValueError, match="unknown selected model.*not_registered"):
        model_registry(config, selected_models=args.models)


def test_legacy_leaf_configs_preserve_settings_and_default_model_registry() -> None:
    """Existing leaf JSONs without models keep their original complete registry."""
    path = CONFIG_DIR / "experiments" / "aad_architecture_screen_reference.json"
    args, config, candidates, print_only = _resolved_arguments({"config": path})
    expected = ExperimentConfig.from_json(path, defaults=EXPERIMENT_DEFAULT_CONFIG)
    assert config == expected
    assert not hasattr(args, "models")
    assert candidates is None and not print_only
    assert [entry["name"] for entry in model_registry(config)] == [
        entry["name"] for entry in model_registry(expected)
    ]
    assert len(model_registry(config)) == 7
