"""Single-JSON AAD study validation, model smoke and sequential runner checks."""

import copy
from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from src import experiments, study_config
from src.experiment_config import ExperimentConfig
from src.model import count_trainable_parameters
from src.utils import write_json, write_video_torchvision

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/aad_staged_experiments.json"
)
SMOKE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/experiments/aad_local_smoke.json"
)
COLAB_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/experiments/aad_colab_a100.json"
)
CUSTOM_SEARCH_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/aad_custom_search_colab.json"
)
MODEL_COMPARISON_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/aad_model_comparison_colab.json"
)


def resolved_comparison_config(tmp_path):
    """Build a comparison config from explicit validation-search evidence."""
    selected = {
        "candidate": {
            "name": "custom_width_late_16_16_32",
            "convlstm_layers": [[16, [3, 3]], [16, [3, 3]], [32, [3, 3]]],
            "hidden_classifier_width": None,
        },
        "input": {"sequence_length": 8, "height": 64, "width": 64},
        "seed": 42,
        "validation_metrics": {"accuracy": 0.8, "loss": 0.5},
        "num_params": 1000,
        "source_run": "runs/experiments/example",
        "split": {"manifest_hash": "fixed-split"},
        "selection_partition": "validation",
        "test_access": "locked",
    }
    selected_path = write_json(tmp_path / "selected_config.json", selected)
    values = json.loads(MODEL_COMPARISON_CONFIG.read_text(encoding="utf-8"))
    # Legacy one-winner studies remain readable; the active top-three mode has
    # separate provenance/guard tests in test_multisize_study.py.
    values.pop("mode")
    values["models"] = ["custom_selected", *values["models"][3:]]
    values["training"]["epochs"] = 64
    values["max_runs"] = 6
    values["selected_config"] = str(selected_path)
    values["custom_candidates"] = [
        {
            "name": "custom_selected",
            "research_question": "Read from validation-selected config",
            "convlstm_layers": selected["candidate"]["convlstm_layers"],
            "hidden_classifier_width": None,
        }
    ]
    return write_json(tmp_path / "comparison.json", values)


def test_custom_search_config_has_complete_resolution_matrix():
    study, config, manifest = study_config.load_study(CUSTOM_SEARCH_CONFIG)
    rows = study_config.study_rows(study, config)
    names = [candidate.name for candidate in manifest.candidates]

    assert len(rows) == study["max_runs"] == 36
    assert names == [
        "custom_flat_16_16",
        "custom_flat_16_16_16",
        "custom_flat_24_24_24",
        "custom_flat_32_32_32",
        "custom_reference_32_16",
        "custom_width_early_32_16_16",
        "custom_width_middle_16_32_16",
        "custom_width_late_16_16_32",
        "custom_narrow_early_8_16",
        "custom_wide_early_24_16",
        "custom_wide_late_16_24",
        "custom_flat_16_16_16_16",
    ]
    assert study["models"] == names
    assert [candidate.convlstm_layers for candidate in manifest.candidates] == [
        ((16, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (16, (3, 3)), (16, (3, 3))),
        ((24, (3, 3)), (24, (3, 3)), (24, (3, 3))),
        ((32, (3, 3)), (32, (3, 3)), (32, (3, 3))),
        ((32, (3, 3)), (16, (3, 3))),
        ((32, (3, 3)), (16, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (32, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (16, (3, 3)), (32, (3, 3))),
        ((8, (3, 3)), (16, (3, 3))),
        ((24, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (24, (3, 3))),
        ((16, (3, 3)), (16, (3, 3)), (16, (3, 3)), (16, (3, 3))),
    ]
    assert study["seeds"] == [42]
    assert (config.sequence_length, config.height, config.width) == (8, 32, 32)
    assert (
        config.epochs,
        config.early_stopping_patience,
        config.batch_size,
        config.learning_rate,
        config.weight_decay,
    ) == (128, 16, 16, 0.01, 0.0)
    assert study["factors"] == {
        "frame_sizes": [32, 48, 64],
        "sequence_lengths": [8],
        "weight_decays": [0.0],
    }
    assert config.cache_dataset is True
    assert {row["stage"] for row in rows} == {"resolution_search"}
    assert {row["seed"] for row in rows} == {42}
    assert all(row["config"]["sequence_length"] == 8 for row in rows)
    assert all(
        row["config"]["augment"] is False and row["config"]["weight_decay"] == 0.0
        for row in rows
    )
    assert {(row["model"], row["config"]["height"]) for row in rows} == {
        (name, size) for name in names for size in [32, 48, 64]
    }
    assert "confirmation_seed" not in study
    assert all(
        row["model"] in names or row["model"] == "VALIDATION_WINNER" for row in rows
    )
    assert all(row["partition"] == "validation" for row in rows)
    assert all(row["test_access"] == "locked" for row in rows)


def test_custom_search_rejects_candidate_and_confirmation_seed_overflow(tmp_path):
    values = json.loads(CUSTOM_SEARCH_CONFIG.read_text(encoding="utf-8"))
    extra = copy.deepcopy(values["custom_candidates"][0])
    extra["name"] = "custom_candidate_13"
    extra["research_question"] = "Out-of-bound candidate"
    values["custom_candidates"].append(extra)
    values["models"].append(extra["name"])
    path = write_json(tmp_path / "too_many_candidates.json", values)
    with pytest.raises(ValueError, match="at most twelve"):
        study_config.load_study(path)

    values = json.loads(CUSTOM_SEARCH_CONFIG.read_text(encoding="utf-8"))
    values["confirmation_seed"] = 42
    path = write_json(tmp_path / "duplicate_confirmation_seed.json", values)
    with pytest.raises(ValueError, match="confirmation"):
        study_config.load_study(path)


def test_model_comparison_config_fixes_protocol_and_includes_model_families(tmp_path):
    comparison_path = resolved_comparison_config(tmp_path)
    study, config, manifest = study_config.load_study(comparison_path)
    rows = study_config.study_rows(study, config)
    registry = experiments.model_registry(
        config, manifest, selected_models=study["models"]
    )

    assert len(rows) == study["max_runs"] == 6
    assert study["models"] == [
        "custom_selected",
        "paper_convlstm_published",
        "r3d_18",
        "mc3_18",
        "swin3d_t",
        "swin3d_s",
    ]
    assert [entry["name"] for entry in registry] == study["models"]
    assert {row["seed"] for row in rows} == {42}
    assert all(row["minimum_epochs"] == 32 for row in rows)
    assert config.epochs == 64 and config.early_stopping_patience == 12
    assert all(row["stage"] == "screen" for row in rows)
    assert all(
        (
            row["config"]["sequence_length"],
            row["config"]["height"],
            row["config"]["width"],
            row["config"]["batch_size"],
        )
        == (50, 50, 50, 1)
        for row in rows
    )
    assert all(row["config"]["weight_decay"] == 0.0001 for row in rows)
    assert all(row["test_access"] == "locked" for row in rows)

    family_by_name = {entry["name"]: entry["family"] for entry in registry}
    records = [
        {
            "name": row["model"],
            "seed": row["seed"],
            "family": family_by_name[row["model"]],
            "model_class": next(
                entry["model_class"]
                for entry in registry
                if entry["name"] == row["model"]
            ),
            "partition": "validation",
            "test_access": "locked",
            "split": {"manifest_hash": "same-split"},
            "dataset_dir": "aad",
            "num_params": 123,
            "experiment_config": row["config"],
            "validation_metrics": {
                "loss": 1.0,
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
            },
        }
        for row in rows
    ]
    table_rows = study_config.aggregate_screen(records, study["models"], study["seeds"])
    assert {row["family"] for row in table_rows} == {
        "ConvLSTM",
        "3D-CNN",
        "Video-Transformer",
    }


def test_comparison_template_requires_search_selection():
    with pytest.raises(ValueError, match="requires selected_config"):
        study_config.load_study(MODEL_COMPARISON_CONFIG)


def test_search_writes_exact_validation_selected_architecture(tmp_path, monkeypatch):
    from src import dataset_source

    values = json.loads(CUSTOM_SEARCH_CONFIG.read_text(encoding="utf-8"))
    # Retain coverage of historical winner-only/second-seed evidence.
    values["mode"] = "staged"
    values["confirmation_seed"] = 2026
    values["factors"]["sequence_lengths"] = [8, 16]
    values["max_runs"] = 16
    values["dataset"]["path"] = str(tmp_path / "aad")
    values["runs_dir"] = str(tmp_path / "runs")
    config_path = write_json(tmp_path / "search.json", values)
    study, config, manifest = study_config.load_study(config_path)
    dataset_path = tmp_path / "aad"
    dataset_path.mkdir()
    monkeypatch.setattr(dataset_source, "resolve_dataset", lambda *a, **k: dataset_path)
    target = "custom_width_late_16_16_32"
    fake_runs = []

    def fake_run(args):
        leaf_config_path = Path(args[args.index("--config") + 1])
        model_name = args[args.index("--model") + 1]
        run_dir = tmp_path / f"screen-{len(fake_runs):02d}"
        run_dir.mkdir()
        leaf_config = ExperimentConfig.from_json(leaf_config_path).to_dict()
        seed = leaf_config["seed"]
        if (
            model_name == target
            and leaf_config["height"] == 48
            and leaf_config["sequence_length"] == 8
        ):
            accuracy = 0.96 if seed == 42 else 0.91
        elif (
            model_name == target
            and leaf_config["height"] == 32
            and leaf_config["sequence_length"] == 8
        ):
            accuracy = 0.95 if seed == 42 else 0.90
        else:
            accuracy = 0.4
        write_json(
            run_dir / "summary.json",
            {
                "all": [
                    {
                        "name": model_name,
                        "seed": seed,
                        "partition": "validation",
                        "test_access": "locked",
                        "split": {"manifest_hash": "fixed-split"},
                        "dataset_dir": str(dataset_path),
                        "num_params": 5000,
                        "input_dimensions": {
                            "sequence_length": leaf_config["sequence_length"],
                            "height": leaf_config["height"],
                            "width": leaf_config["width"],
                        },
                        "validation_metrics": {
                            "accuracy": accuracy,
                            "loss": 1 - accuracy,
                            "precision": accuracy,
                            "recall": accuracy,
                            "f1": accuracy,
                        },
                        "experiment_config": leaf_config,
                    }
                ]
            },
        )
        fake_runs.append(run_dir)
        return run_dir

    result_dir = study_config.execute_study(
        config_path, study, config, manifest, fake_run
    )
    selected = json.loads((result_dir / "selected_config.json").read_text())
    assert selected["candidate"]["name"] == target
    assert selected["candidate"]["convlstm_layers"] == [
        [16, [3, 3]],
        [16, [3, 3]],
        [32, [3, 3]],
    ]
    assert selected["input"] == {"sequence_length": 8, "height": 48, "width": 48}
    assert selected["seed"] == 42
    assert selected["validation_metrics"]["accuracy"] == 0.96
    assert selected["seed_confirmation"]["seed"] == 2026
    assert selected["seed_confirmation"]["validation_metrics"]["accuracy"] == 0.91
    assert selected["seed_confirmation"]["split"]["manifest_hash"] == "fixed-split"
    assert selected["seed_confirmation"]["same_selected_input"] is True
    assert selected["single_seed_evidence"] is False
    assert selected["num_params"] == 5000
    assert Path(selected["source_run"]).is_dir()
    assert selected["selection_partition"] == "validation"
    assert selected["test_access"] == "locked"
    assert selected["split"]["manifest_hash"] == "fixed-split"
    summary = json.loads((result_dir / "summary.json").read_text())
    confirmation = next(
        job for job in summary["jobs"] if job["stage"] == "seed_confirmation"
    )
    assert confirmation["result"]["seed"] == 2026
    assert confirmation["result"]["experiment_config"]["split_seed"] == 42
    assert confirmation["result"]["experiment_config"]["height"] == 48
    assert confirmation["result"]["experiment_config"]["sequence_length"] == 8
    assert summary["validation_metrics_by_seed"] == {
        "42": selected["validation_metrics"],
        "2026": selected["seed_confirmation"]["validation_metrics"],
    }


def test_new_colab_profiles_list_without_training(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        experiments,
        "build_registered_model",
        lambda *a, **k: pytest.fail("model allocated while listing"),
    )
    experiments.main(["--config", str(CUSTOM_SEARCH_CONFIG), "--list-plan"])
    search_output = capsys.readouterr().out
    assert "custom architecture search: 36 declared runs" in search_output
    assert "custom_flat_24_24_24" in search_output

    comparison_config = resolved_comparison_config(tmp_path)
    experiments.main(["--config", str(comparison_config), "--list-plan"])
    comparison_output = capsys.readouterr().out
    assert "model-family comparison: 6 declared runs" in comparison_output
    assert "paper_convlstm_published" in comparison_output
    assert "swin3d_s" in comparison_output


def test_comparison_models_are_feasible_on_native_paper_input(tmp_path):
    comparison_config = resolved_comparison_config(tmp_path)
    study, config, manifest = study_config.load_study(comparison_config)
    with torch.device("meta"), torch.inference_mode():
        for name in study["models"]:
            model = experiments.build_registered_model(
                name,
                11,
                (3, 50, 50),
                50,
                config,
                manifest,
            ).eval()
            logits = model(torch.zeros(1, 50, 3, 50, 50))
            assert logits.shape == (1, 11), name
            assert count_trainable_parameters(model) > 0


def test_local_smoke_is_one_tiny_training_pass():
    study, config, _ = study_config.load_study(SMOKE_CONFIG)
    rows = study_config.study_rows(study, config)
    assert len(rows) == study["max_runs"] == 1
    assert len(study["models"]) == 1
    assert study["profile"] == "local_smoke"
    assert set(study["models"]) == {"local_tiny_4_4"}
    assert study["frame_sizes"] == [16]
    assert study["num_frames"] == [2]
    assert study["data_sizes"] == [[2, 16]]
    assert study["epoch_values"] == [1]
    assert study["factors"]["weight_decays"] == [0.0]
    assert all(row["config"]["epochs"] == 1 for row in rows)
    assert all(row["config"]["early_stopping_patience"] == 1 for row in rows)
    assert all(row["config"]["batch_size"] == 16 for row in rows)
    assert len({row["trial_id"] for row in rows}) == 1
    assert all(row["partition"] == "validation" for row in rows)
    assert all(row["test_access"] == "locked" for row in rows)


def test_legacy_colab_grid_remains_readable_for_historical_reproduction():
    study, config, _ = study_config.load_study(COLAB_CONFIG)
    rows = study_config.study_rows(study, config)
    assert study["profile"] == "colab_a100"
    assert len(rows) == study["max_runs"] == 32
    assert study["frame_sizes"] == [32, 48]
    assert study["num_frames"] == [8, 16]
    assert study["epoch_values"] == [16]
    assert study["seeds"] == [42]
    assert all(row["test_access"] == "locked" for row in rows)


def test_grid_config_selects_local_dataset_and_uses_its_default_split(tmp_path):
    from src.dataset import resolve_split_manifest_path

    values = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    values["dataset_name"] = "vdd"
    values["dataset_dir"] = str(tmp_path / "vdd")
    values["split_manifest"] = None
    path = write_json(tmp_path / "vdd-grid.json", values)

    study, config, _ = study_config.load_study(path)

    assert study["dataset"]["name"] == "vdd"
    assert config.dataset_name == "vdd"
    assert config.dataset_dir == str((tmp_path / "vdd").resolve())
    assert config.split_manifest is None
    assert (
        resolve_split_manifest_path(
            config.dataset_dir, config.split_manifest, config.split_seed
        ).name
        == "vdd_seed42.json"
    )


def test_staged_config_selects_dataset_without_aad_specific_validation(tmp_path):
    values = json.loads(CONFIG.read_text(encoding="utf-8"))
    values["dataset"].update(
        name="vdd",
        path=str(tmp_path / "custom-vdd"),
        split_manifest=None,
    )
    path = write_json(tmp_path / "vdd-study.json", values)

    study, config, _ = study_config.load_study(path)

    assert study["dataset"]["name"] == "vdd"
    assert config.dataset_name == "vdd"
    assert config.dataset_dir == str((tmp_path / "custom-vdd").resolve())
    assert config.split_manifest is None


def test_smoke_selection_is_validation_only_and_requires_one_fixed_split():
    study, config, manifest = study_config.load_study(SMOKE_CONFIG)
    rows = study_config.study_rows(study, config)
    records = []
    for index, row in enumerate(rows):
        candidate = next(
            (item for item in manifest.candidates if item.name == row["model"]),
            None,
        )
        values = dict(row["config"])
        if candidate is not None:
            values["convlstm_layers"] = candidate.to_dict()["convlstm_layers"]
            values["hidden_classifier_width"] = candidate.hidden_classifier_width
        values = ExperimentConfig.from_mapping(values).to_dict()
        row["config"] = values
        records.append(
            {
                "name": row["model"],
                "seed": row["seed"],
                "partition": "validation",
                "test_access": "locked",
                "split": {"manifest_hash": "fixed-split"},
                "num_params": index + 1,
                "validation_metrics": {
                    "accuracy": 0.5 + index / 100,
                    "loss": 1.0,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                },
                "experiment_config": values,
            }
        )
    ranking = study_config.rank_smoke_grid(records, rows)
    assert ranking[0]["name"] == rows[-1]["trial_id"]
    records[0]["partition"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        study_config.rank_smoke_grid(records, rows)


def test_config_command_lists_smoke_without_training(monkeypatch, capsys):
    monkeypatch.setattr(
        experiments,
        "build_registered_model",
        lambda *a, **k: pytest.fail("model allocated while listing"),
    )
    experiments.main(["--config", str(SMOKE_CONFIG), "--list-plan"])
    listing = capsys.readouterr().out
    assert "1 Cartesian configurations" in listing
    assert "not paper evidence" in listing
    assert "--config" in listing
    assert "local_tiny_4_4" in listing


def test_smoke_writes_selected_train_config_and_keeps_test_locked(
    tmp_path, monkeypatch
):
    from src import dataset_source

    study_values = json.loads(SMOKE_CONFIG.read_text())
    study_values["runs_dir"] = str(tmp_path / "runs")
    config_path = write_json(tmp_path / "smoke.json", study_values)
    study, config, manifest = study_config.load_study(config_path)
    monkeypatch.setattr(dataset_source, "resolve_dataset", lambda *a, **k: tmp_path)

    def fake_run(args):
        config_path = Path(args[args.index("--config") + 1])
        model_name = args[args.index("--model") + 1]
        run_dir = tmp_path / f"fake-{len(list(tmp_path.glob('fake-*'))):02d}"
        run_dir.mkdir()
        leaf_config = ExperimentConfig.from_json(config_path).to_dict()
        write_json(
            run_dir / "summary.json",
            {
                "all": [
                    {
                        "name": model_name,
                        "seed": leaf_config["seed"],
                        "partition": "validation",
                        "test_access": "locked",
                        "split": {"manifest_hash": "fixed-split"},
                        "dataset_dir": str(tmp_path),
                        "num_params": 100,
                        "validation_metrics": {
                            "accuracy": 0.75,
                            "loss": 0.5,
                            "precision": 0.75,
                            "recall": 0.75,
                            "f1": 0.75,
                        },
                        "experiment_config": leaf_config,
                    }
                ]
            },
        )
        return run_dir

    result_dir = study_config.execute_study(
        config_path, study, config, manifest, fake_run
    )
    selection = json.loads((result_dir / "selection.json").read_text())
    selected_config = result_dir / "selected_config.json"
    assert selection["pipeline_smoke_only"] is True
    assert selection["selection_partition"] == "validation"
    assert selection["test_access"] == "locked"
    assert selection["selected_config"] == str(selected_config)
    restored = ExperimentConfig.from_json(selected_config)
    assert restored.sequence_length == 2 and restored.epochs == 1
    summary = json.loads((result_dir / "summary.json").read_text())
    assert len(summary["jobs"]) == 1
    assert all(job["result"]["partition"] == "validation" for job in summary["jobs"])


def test_selected_architecture_handoff_and_test_evaluation_after_freeze(
    tmp_path, monkeypatch
):
    from src import dataset_source

    comparison_config = resolved_comparison_config(tmp_path)
    values = json.loads(comparison_config.read_text(encoding="utf-8"))
    values["dataset"]["path"] = str(tmp_path / "aad")
    values["runs_dir"] = str(tmp_path / "runs")
    comparison_config = write_json(comparison_config, values)
    study, config, manifest = study_config.load_study(comparison_config)
    dataset_path = tmp_path / "aad"
    dataset_path.mkdir()
    monkeypatch.setattr(dataset_source, "resolve_dataset", lambda *a, **k: dataset_path)
    leaf_runs = []
    evaluation_calls = []

    def fake_run(args):
        config_path = Path(args[args.index("--config") + 1])
        model_name = args[args.index("--model") + 1]
        assert args[args.index("--minimum-epochs") + 1] == "32"
        run_dir = tmp_path / f"leaf-{len(leaf_runs):02d}"
        run_dir.mkdir()
        checkpoint = run_dir / "best_model.pth"
        checkpoint.write_bytes(b"checkpoint")
        leaf_config = ExperimentConfig.from_json(config_path).to_dict()
        score = 0.9 if model_name == "custom_selected" else 0.7
        write_json(
            run_dir / "summary.json",
            {
                "all": [
                    {
                        "name": model_name,
                        "seed": 42,
                        "partition": "validation",
                        "test_access": "locked",
                        "split": {"manifest_hash": "fixed-split"},
                        "dataset_dir": str(dataset_path),
                        "num_params": 100,
                        "validation_metrics": {
                            "accuracy": score,
                            "loss": 1 - score,
                            "precision": score,
                            "recall": score,
                            "f1": score,
                        },
                        "experiment_config": leaf_config,
                        "selected_checkpoint": str(checkpoint),
                    }
                ]
            },
        )
        leaf_runs.append(run_dir)
        return run_dir

    def fake_evaluate(config_path):
        assert list((tmp_path / "runs/studies").glob("*/validation_frozen.json"))
        eval_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        evaluation_calls.append(eval_config)
        run_dir = tmp_path / f"eval-{len(evaluation_calls):02d}"
        (run_dir / "metrics").mkdir(parents=True)
        write_json(
            run_dir / "metrics/final.json",
            {
                "partition": "test",
                "split": {"manifest_hash": "fixed-split"},
                "metrics": {
                    "loss": 0.5,
                    "accuracy": 0.8,
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1": 0.8,
                },
                "artifacts": {"confusion_matrix": {"json": "test.json"}},
                "prediction_examples": {"records": ["example.mp4"]},
            },
        )
        return run_dir

    result_dir = study_config.execute_study(
        comparison_config, study, config, manifest, fake_run, fake_evaluate
    )
    summary = json.loads((result_dir / "summary.json").read_text())
    assert len(leaf_runs) == len(evaluation_calls) == 6
    assert summary["selected_model"] == "custom_selected"
    assert summary["test_evaluation"]["frozen_before_test"] is True
    assert len(summary["test_evaluation"]["models"]) == 6
    assert all(
        item["partition"] == "test" for item in summary["test_evaluation"]["models"]
    )
    selected = json.loads(MODEL_COMPARISON_CONFIG.read_text())
    assert selected["custom_candidates"] == []
    custom_run = next(
        item for item in summary["jobs"] if item["result"]["name"] == "custom_selected"
    )
    assert custom_run["result"]["experiment_config"]["convlstm_layers"] == [
        [16, [3, 3]],
        [16, [3, 3]],
        [32, [3, 3]],
    ]


def test_additive_stages_and_fixed_protocol():
    study, config, manifest = study_config.load_study(CONFIG)
    rows = study_config.study_rows(study, config, "r3d_18")
    assert len(rows) == 22
    assert len([r for r in rows if r["stage"] == "screen"]) == 14
    assert len(manifest.candidates) == 3
    expected = {
        "frame_size": {"height", "width"},
        "sequence_length": {"sequence_length"},
        "weight_decay": {"weight_decay"},
    }
    for row in rows:
        baseline = {**config.to_dict(), "seed": row["seed"]}
        changed = {key for key in baseline if baseline[key] != row["config"][key]}
        assert changed == (
            expected[row["changed_factor"]] if row["stage"] == "ablation" else set()
        )
        assert row["partition"] == "validation" and row["test_access"] == "locked"
    study["published_topology"]["enabled"] = True
    native = [
        r for r in study_config.study_rows(study, config) if r["stage"] == "native"
    ]
    assert len(native) == 2
    assert all(
        r["config"]["sequence_length"] == r["config"]["height"] == 50 for r in native
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["models"].append("unknown"),
        lambda s: s["factors"]["frame_sizes"].append(-8),
        lambda s: s["factors"]["frame_sizes"].append(65),
        lambda s: s["factors"]["sequence_lengths"].append(True),
        lambda s: s["factors"]["weight_decays"].append(float("nan")),
        lambda s: s["factors"].update(cartesian=True),
        lambda s: s["training"].update(learning_rate=[0.001, 0.01]),
        lambda s: s.update(max_runs=10),
        lambda s: s.update(seeds=[42, 42]),
        lambda s: s["dataset"].update(name=""),
        lambda s: s["reference_input"].update(frame_size=63),
    ],
)
def test_invalid_study_fails_before_dataset_or_training(tmp_path, monkeypatch, change):
    values = json.loads(CONFIG.read_text())
    change(values)
    path = write_json(tmp_path / "bad.json", values)
    monkeypatch.setattr(
        experiments, "resolve_dataset", lambda *a, **k: pytest.fail("dataset accessed")
    )
    with pytest.raises((ValueError, TypeError)):
        experiments.main(["--study-config", str(path)])


def test_no_training_listing_and_override_rejection(monkeypatch, capsys):
    monkeypatch.setattr(
        experiments,
        "build_registered_model",
        lambda *a, **k: pytest.fail("model allocated"),
    )
    experiments.main(["--study-config", str(CONFIG), "--list-plan"])
    listing = capsys.readouterr().out
    assert "22 declared runs" in listing and "swin3d_t" in listing
    assert "VALIDATION_WINNER" in listing and "separate/unranked" in listing
    for override in (
        ["--seed", "0"],
        ["--print-config"],
        ["--run-plan-stage", "architecture-screen"],
    ):
        with pytest.raises(ValueError, match="study-config"):
            experiments.main(["--study-config", str(CONFIG), *override])


def test_all_declared_models_forward_without_weights_download(monkeypatch):
    torch.set_num_threads(1)
    study, config, manifest = study_config.load_study(CONFIG)
    original = experiments.video_models.swin3d_t
    calls = []

    def scratch_swin(*, weights):
        calls.append(weights)
        return original(weights=weights)

    monkeypatch.setattr(experiments.video_models, "swin3d_t", scratch_swin)
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *a, **k: pytest.fail("pretrained download"),
    )
    registry = experiments.model_registry(
        config, manifest, selected_models=study["models"]
    )
    assert [entry["name"] for entry in registry] == study["models"]
    for name in study["models"]:
        model = experiments.build_registered_model(
            name, 11, (3, 16, 16), 4, config, manifest
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 4, 3, 16, 16))
        assert output.shape == (1, 11) and torch.isfinite(output).all()
        assert count_trainable_parameters(model) > 0
        if name == "swin3d_t":
            assert model.model.head.out_features == 11
            assert 27_000_000 < count_trainable_parameters(model) < 29_000_000
        del model
    assert calls == [None]
    # Full faithful topology is too large for a cheap CPU smoke; no tensor storage.
    with torch.device("meta"), torch.inference_mode():
        paper = experiments.build_registered_model(
            "paper_convlstm_published", 11, (3, 50, 50), 50
        ).eval()
        assert count_trainable_parameters(paper) == 512_197_467
        assert paper(torch.zeros(1, 50, 3, 50, 50)).shape == (1, 11)


def _records():
    config = ExperimentConfig().to_dict()
    return [
        {
            "name": name,
            "seed": seed,
            "num_params": 10,
            "partition": "validation",
            "test_access": "locked",
            "split": {"manifest_hash": "split"},
            "dataset_dir": "aad",
            "experiment_config": {**config, "seed": seed},
            "validation_metrics": {
                "accuracy": score,
                "loss": 1 - score,
                "precision": score,
                "recall": score,
                "f1": score,
            },
            "test_metrics": {"accuracy": 1 - score},
        }
        for name, score in (("small", 0.5), ("large", 0.7))
        for seed in (42, 2026)
    ]


def test_reference_uses_all_validation_seeds_not_test():
    records = _records()
    records[2]["validation_metrics"]["accuracy"] = 0.6
    ranking = study_config.aggregate_screen(records, ["small", "large"], [42, 2026])
    assert ranking[0]["name"] == "large"
    assert ranking[0]["validation_metrics"]["accuracy"] == pytest.approx(0.65)
    assert ranking[0]["validation_std"]["accuracy"] > 0
    for mutation in ("missing", "test", "split", "budget"):
        bad = copy.deepcopy(records)
        if mutation == "missing":
            bad.pop()
        elif mutation == "test":
            bad[0]["partition"] = "test"
        elif mutation == "split":
            bad[0]["split"]["manifest_hash"] = "other"
        else:
            bad[0]["experiment_config"]["epochs"] += 1
        with pytest.raises(ValueError):
            study_config.aggregate_screen(bad, ["small", "large"], [42, 2026])


def test_end_to_end_study_reuses_runner_and_keeps_test_locked(tmp_path, monkeypatch):
    from src import dataset, dataset_source

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = tmp_path / "aad"
    for class_name in dataset_source.AAD_CLASS_NAMES:
        for number in range(10):
            write_video_torchvision(
                torch.rand(3, 3, 8, 8), root / class_name / f"{number}.mp4"
            )
    split = tmp_path / "split.json"
    source = dataset.AHARDataset(root, 2, (8, 8))
    _, _, test, _ = dataset.load_split_subsets(source, split)
    locked = {source.samples[index][0].resolve() for index in test.indices}
    original = dataset.AHARDataset._load_video
    decoded_paths = []

    def guarded(self, path):
        assert Path(path).resolve() not in locked
        decoded_paths.append(Path(path).resolve())
        return original(self, path)

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", guarded)
    values = json.loads(CONFIG.read_text())
    values["dataset"].update(path=str(root), split_manifest=str(split))
    values["runs_dir"] = str(tmp_path / "runs")
    values["models"] = ["tiny"]
    values["custom_candidates"] = [
        {
            "name": "tiny",
            "research_question": "Synthetic runner check",
            "convlstm_layers": [[2, [3, 3]]],
            "hidden_classifier_width": None,
        }
    ]
    values["training"].update(
        epochs=2,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        prediction_samples_per_category=0,
        cache_dataset=True,
    )
    values["reference_input"] = {
        "frame_size": 8,
        "sequence_length": 2,
        "weight_decay": 0.0,
    }
    values["factors"] = {
        "frame_sizes": [8],
        "sequence_lengths": [2, 3],
        "weight_decays": [0.0],
    }
    path = write_json(tmp_path / "study.json", values)
    result_dir = experiments.main(["--study-config", str(path)])
    summary = json.loads((result_dir / "summary.json").read_text())
    assert summary["selected_model"] == "tiny"
    assert summary["test_access"] == "locked"
    assert [job["stage"] for job in summary["jobs"]] == [
        "screen",
        "screen",
        "ablation",
        "ablation",
    ]
    assert len(list(result_dir.glob("jobs/*/command.json"))) == 4
    assert len(summary["reference_runs"]) == 2
    assert len(summary["ablation_comparisons"]) == 1
    assert "validation_std" in summary["ablation_comparisons"][0]["summary"]
    for job in summary["jobs"]:
        result = job["result"]
        assert result["num_params"] > 0 and result["train_time_s"] >= 0
        assert result["split"]["manifest_hash"] and result["dataset_name"] == "aad"
        assert result["input_dimensions"]["height"] == 8
        assert result["partition"] == "validation"
    assert json.loads((result_dir / "run.json").read_text())["status"] == "complete"
    # One bounded cache is reused across both seeds for each of two inputs.
    assert set(Counter(decoded_paths).values()) == {2}
