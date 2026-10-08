"""Single-JSON AAD study validation, model smoke and sequential runner checks."""

import copy
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
    Path(__file__).resolve().parents[1]
    / "configs/experiments/aad_local_smoke.json"
)
COLAB_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/aad_colab_a100.json"
)
CUSTOM_SEARCH_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/aad_custom_search_colab.json"
)
MODEL_COMPARISON_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/aad_model_comparison_colab.json"
)


def test_custom_search_config_has_reference_and_one_factor_variants():
    study, config, manifest = study_config.load_study(CUSTOM_SEARCH_CONFIG)
    rows = study_config.study_rows(study, config)
    names = [candidate.name for candidate in manifest.candidates]

    assert len(rows) == study["max_runs"] == 20
    assert names == [
        "custom_flat_16_16_16",
        "custom_depth_16_16",
        "custom_depth_16_16_16_16",
        "custom_width_early_32_16_16",
        "custom_width_middle_16_32_16",
        "custom_width_late_16_16_32",
    ]
    assert study["models"] == names
    assert [candidate.convlstm_layers for candidate in manifest.candidates] == [
        ((16, (3, 3)), (16, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (16, (3, 3)), (16, (3, 3)), (16, (3, 3))),
        ((32, (3, 3)), (16, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (32, (3, 3)), (16, (3, 3))),
        ((16, (3, 3)), (16, (3, 3)), (32, (3, 3))),
    ]
    assert study["seeds"] == [42, 2026]
    assert {row["stage"] for row in rows} == {"screen", "ablation"}
    assert all(
        row["model"] in names or row["model"] == "VALIDATION_WINNER"
        for row in rows
    )
    assert all(row["partition"] == "validation" for row in rows)
    assert all(row["test_access"] == "locked" for row in rows)


def test_model_comparison_config_fixes_protocol_and_includes_model_families():
    study, config, manifest = study_config.load_study(MODEL_COMPARISON_CONFIG)
    rows = study_config.study_rows(study, config)
    registry = experiments.model_registry(
        config, manifest, selected_models=study["models"]
    )

    assert len(rows) == study["max_runs"] == 12
    assert study["models"] == [
        "custom_selected",
        "paper_convlstm_published",
        "r3d_18",
        "mc3_18",
        "swin3d_t",
        "swin3d_s",
    ]
    assert [entry["name"] for entry in registry] == study["models"]
    assert {row["seed"] for row in rows} == {42, 2026}
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
    table_rows = study_config.aggregate_screen(
        records, study["models"], study["seeds"]
    )
    assert {row["family"] for row in table_rows} == {
        "ConvLSTM",
        "3D-CNN",
        "Video-Transformer",
    }


def test_new_colab_profiles_list_without_training(monkeypatch, capsys):
    monkeypatch.setattr(
        experiments,
        "build_registered_model",
        lambda *a, **k: pytest.fail("model allocated while listing"),
    )
    experiments.main(["--config", str(CUSTOM_SEARCH_CONFIG), "--list-plan"])
    search_output = capsys.readouterr().out
    assert "custom architecture search: 20 declared runs" in search_output
    assert "custom_flat_16_16_16" in search_output

    experiments.main(["--config", str(MODEL_COMPARISON_CONFIG), "--list-plan"])
    comparison_output = capsys.readouterr().out
    assert "model-family comparison: 12 declared runs" in comparison_output
    assert "paper_convlstm_published" in comparison_output
    assert "swin3d_s" in comparison_output


def test_comparison_models_are_feasible_on_native_paper_input():
    study, config, manifest = study_config.load_study(MODEL_COMPARISON_CONFIG)
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


def test_local_smoke_is_an_eight_run_tiny_custom_grid():
    study, config, _ = study_config.load_study(SMOKE_CONFIG)
    rows = study_config.study_rows(study, config)
    assert len(rows) == study["max_runs"] == 8
    assert len(study["models"]) == 2
    assert study["profile"] == "local_smoke"
    assert set(study["models"]) == {"local_tiny_8_4", "local_tiny_4_8"}
    assert study["frame_sizes"] == [32]
    assert study["num_frames"] == [4, 8]
    assert study["data_sizes"] == [[4, 32], [8, 32]]
    assert study["epoch_values"] == [8]
    assert study["factors"]["weight_decays"] == [0.0, 0.001]
    assert all(row["config"]["epochs"] == 8 for row in rows)
    assert all(row["config"]["early_stopping_patience"] == 4 for row in rows)
    assert all(row["config"]["batch_size"] == 8 for row in rows)
    assert len({row["trial_id"] for row in rows}) == 8
    assert all(row["partition"] == "validation" for row in rows)
    assert all(row["test_access"] == "locked" for row in rows)


def test_colab_profile_has_four_data_sizes_and_16_epoch_budget():
    study, config, _ = study_config.load_study(COLAB_CONFIG)
    rows = study_config.study_rows(study, config)
    assert study["profile"] == "colab_a100"
    assert len(rows) == study["max_runs"] == 32
    assert set(study["models"]) == {
        "custom_two_layer_32_16",
        "custom_two_layer_16_32",
        "custom_depth_32_16_8",
        "r3d_18",
    }
    assert study["frame_sizes"] == [32, 48]
    assert study["num_frames"] == [8, 16]
    assert {tuple(size) for size in study["data_sizes"]} == {
        (8, 32),
        (16, 32),
        (8, 48),
        (16, 48),
    }
    assert study["epoch_values"] == [16]
    assert study["factors"]["weight_decays"] == [0.0, 0.0001]
    assert {row["config"]["epochs"] for row in rows} == {16}
    assert all(row["config"]["early_stopping_patience"] == 4 for row in rows)
    assert all(row["partition"] == "validation" for row in rows)
    assert all(row["test_access"] == "locked" for row in rows)
    assert study["seeds"] == [42]
    assert {row["config"]["height"] for row in rows} == {32, 48}
    assert {row["config"]["width"] for row in rows} == {32, 48}
    assert {row["config"]["sequence_length"] for row in rows} == {8, 16}
    assert {row["config"]["weight_decay"] for row in rows} == {0.0, 0.0001}


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
    assert resolve_split_manifest_path(
        config.dataset_dir, config.split_manifest, config.split_seed
    ).name == "vdd_seed42.json"


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
    assert "8 Cartesian configurations" in listing
    assert "not paper evidence" in listing
    assert "--config" in listing
    assert "local_tiny_8_4" in listing


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
    assert restored.sequence_length in {4, 8} and restored.epochs == 8
    summary = json.loads((result_dir / "summary.json").read_text())
    assert len(summary["jobs"]) == 8
    assert all(job["result"]["partition"] == "validation" for job in summary["jobs"])


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

    def guarded(self, path):
        assert Path(path).resolve() not in locked
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
        epochs=1,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        prediction_samples_per_category=0,
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
