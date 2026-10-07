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
        lambda s: s["models"].append("paper_convlstm_published"),
        lambda s: s["factors"]["frame_sizes"].append(-8),
        lambda s: s["factors"]["frame_sizes"].append(65),
        lambda s: s["factors"]["sequence_lengths"].append(True),
        lambda s: s["factors"]["weight_decays"].append(float("nan")),
        lambda s: s["factors"].update(cartesian=True),
        lambda s: s["training"].update(learning_rate=[0.001, 0.01]),
        lambda s: s.update(max_runs=10),
        lambda s: s.update(seeds=[42, 42]),
        lambda s: s["dataset"].update(name="kinetics-subset"),
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
    assert "22 model runs" in listing and "swin3d_t" in listing
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
