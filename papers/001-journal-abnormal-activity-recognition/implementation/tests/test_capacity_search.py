"""Ticket 069: adaptive decisions, evidence guards, sampling and full reporting."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import capacity_config as plan, capacity_search as search
from src import dataset, experiments, study_cache, study_config, study_matrix
from src.experiment_config import ExperimentConfig
from src.model import CustomConvLSTM
from src.study_reporting import extended_metrics, result_row, write_full_table
from src.study_resources import inference_measurement
from src.temporal_sampling import timestamp_plan, video_plan
from src.utils import write_json, write_video_torchvision
from notebooks.utils import aad_study, study_archive

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/experiments/aad_capacity_search_colab.json"


def test_exact_plan_and_unmodified_legacy_protocol(capsys):
    values, config, manifest = study_config.load_study(PROFILE)
    assert len(manifest.candidates) == 21
    assert sum(plan.STAGE_LIMITS.values()) == 156
    assert len(plan.calibration_rows(values, config)) == 12
    rows = plan.matrix_rows(
        values, config, plan.flats(values), {"custom": 0.003}, "flat"
    )
    assert len(rows) == 63
    assert {len(r["candidate"]["convlstm_layers"]) for r in rows} == {1, 2, 3}
    assert {r["config"]["sequence_length"] for r in rows} == {8}
    study_config.print_study(PROFILE, values, config)
    output = capsys.readouterr().out
    assert "custom_4 | 32x32" in output and "swin3d_t | 64x64" in output
    assert "156" in output and "test locked" in output
    legacy, old_config, _ = study_config.load_study(
        ROOT / "configs/experiments/aad_custom_search_colab.json"
    )
    assert (
        old_config.epochs == 128
        and len(study_config.study_rows(legacy, old_config)) == 36
    )
    assert "sampling_version" not in old_config.to_dict()
    assert config.sampling_version == "timestamps_v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("widths", [4, 8]),
        ("depths", [1, 2]),
        ("seeds", [42]),
        ("max_runs", 999),
        ("learning_rates", [0.001, 0.01]),
    ],
)
def test_protocol_rejects_silent_expansion_or_reduction(field, value):
    values = study_matrix.read_json(PROFILE)
    values[field] = value
    with pytest.raises(ValueError):
        plan.load_capacity(values)


def test_refinement_dedup_and_one_factor_ablation_counts():
    values, config, _ = study_config.load_study(PROFILE)
    assert len(plan.refinement_candidates(4, plan.flats(values))) == 3
    assert len(plan.refinement_candidates(24, plan.flats(values))) == 6
    shortlist = [plan.candidate([4]), plan.candidate([8, 8]), plan.candidate([24] * 3)]
    lrs = {"custom": 0.003, "r3d_18": 0.001, "swin3d_t": 0.001}
    assert len(plan.ablation_rows(values, config, shortlist, lrs, "weight_decay")) == 18
    rows = plan.ablation_rows(values, config, shortlist, lrs, "temporal")
    assert len(rows) == 15
    for row in rows:
        cfg = row["config"]
        assert cfg["height"] == cfg["width"] == 48
        assert cfg["weight_decay"] == 0 and cfg["sequence_length"] <= 16
        assert (cfg["sequence_length"], cfg["target_fps"]) in {(8, 8), (8, 4), (16, 16)}
    assert len(plan.confirmation_specs(values, shortlist, 24)) <= 8


@pytest.mark.parametrize(
    "depth,width", [(1, 4), (1, 64), (2, 4), (2, 32), (3, 64), (4, 8)]
)
def test_custom_shallow_and_deep_shapes(depth, width):
    torch.set_num_threads(1)
    model = CustomConvLSTM(3, layers=[(width, (3, 3))] * depth)
    output = model(torch.zeros(2, 2, 3, 8, 8))
    assert output.shape == (2, 3) and torch.isfinite(output).all()


@pytest.mark.parametrize("name", ["r3d_18", "swin3d_t"])
def test_real_reference_shapes_and_cpu_measurement(name):
    torch.set_num_threads(1)
    config = ExperimentConfig(sequence_length=8, height=32, width=32)
    model = experiments.build_registered_model(name, 3, (3, 32, 32), 8, config)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 8, 3, 32, 32)).shape == (1, 3)
    measured = inference_measurement(model, config, torch.device("cpu"), repeats=1)
    assert measured["latency_ms_per_batch"] > 0
    assert measured["peak_cuda_memory_bytes"] is None
    assert measured["batch_size"] == 1


def test_timestamp_sampling_fixed_variable_short_and_low_fps():
    normal = timestamp_plan([i / 30 for i in range(60)], 16, 8, 30)
    assert normal["selected_indices"] == [0, 1, 3, 5, 7, 9, 11, 13]
    assert normal["padding_fraction"] == 0
    low = timestamp_plan([i / 4 for i in range(8)], 16, 8, 4)
    assert low["selected_indices"] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert low["unique_frames"] == 2 and low["padding_fraction"] == 0
    short = timestamp_plan([0, 0.1], 10, 8, 10)
    assert short["selected_indices"] == [0, 1, 1, 1, 1, 1, 1, 1]
    assert short["padding_fraction"] == 0.75
    variable = timestamp_plan([0, 0.02, 0.11, 0.3, 0.4], 8, 4, 10)
    assert variable["variable_frame_rate"]
    assert variable["selected_indices"] == [0, 2, 2, 3]
    assert variable["target_timestamps_seconds"] == [0, 0.125, 0.25, 0.375]
    for times in ([], [0, 0], [0, -1], [0, float("nan")]):
        with pytest.raises(ValueError):
            timestamp_plan(times, 16, 8, 30)


def test_macro_and_balanced_metrics_are_not_relabelled_micro():
    metrics = extended_metrics([0, 0, 0, 0], [0, 0, 0, 1], 3)
    assert metrics["micro_f1"] == 0.75
    assert metrics["macro_recall"] == pytest.approx(1 / 3)
    assert metrics["macro_precision"] == pytest.approx(0.25)
    assert metrics["macro_f1"] == pytest.approx((6 / 7) / 3)
    assert metrics["balanced_accuracy"] == 0.5


def test_calibration_uses_family_mean_then_loss_then_lower_lr():
    values, config, _ = study_config.load_study(PROFILE)
    jobs = []
    for row in plan.calibration_rows(values, config):
        name, lr = row["model"], row["config"]["learning_rate"]
        if name == "custom_24_24_24":
            accuracy, loss = (0.9 if lr == 0.001 else 0.8), 0.2
        elif name == "custom_64_64_64":
            accuracy, loss = (0.4 if lr == 0.001 else 0.8), 0.2
        elif name == "r3d_18":
            accuracy, loss = (0.7 if lr == 0.01 else 0.8), (0.3 if lr == 0.001 else 0.2)
        else:
            accuracy, loss = 0.5, 0.1
        jobs.append(
            {
                "result": {
                    "name": name,
                    "experiment_config": row["config"],
                    "validation_metrics": {"accuracy": accuracy, "loss": loss},
                }
            }
        )
    assert plan.select_learning_rates(values, jobs) == {
        "custom": 0.003,
        "r3d_18": 0.003,
        "swin3d_t": 0.001,
    }
    with pytest.raises(ValueError, match="exactly once"):
        plan.select_learning_rates(values, jobs[:-1])


def test_two_seed_sd_is_not_resolution_variation_and_bad_evidence_blocks():
    values, config, _ = study_config.load_study(PROFILE)
    specs = [plan.candidate([4]), plan.candidate([8, 8])]
    jobs = []
    for seed in values["seeds"]:
        for row in plan.matrix_rows(
            values, config, specs, {"custom": 0.001}, "confirmation", seed
        ):
            name, size = row["model"], row["config"]["height"]
            mean = (0.6 if seed == 42 else 0.9) if name == "custom_4" else 0.7
            jobs.append(
                {
                    "result": {
                        "name": name,
                        "seed": seed,
                        "experiment_config": row["config"],
                        "partition": "validation",
                        "test_access": "locked",
                        "dataset_dir": config.dataset_dir,
                        "split": {"manifest_hash": "fixed"},
                        "num_params": 400 if name == "custom_4" else 1600,
                        "validation_metrics": {
                            "accuracy": mean + (size - 48) / 320,
                            "loss": 0.2,
                        },
                    }
                }
            )
    names = [c["name"] for c in specs]
    ranked = plan.architecture_ranking(
        jobs, names, values["frame_sizes"], values["seeds"]
    )
    assert ranked[0]["name"] == "custom_4"
    assert ranked[0]["mean_validation_accuracy"] == pytest.approx(0.75)
    assert ranked[0]["seed_sample_sd"] == pytest.approx((0.15**2 * 2) ** 0.5)
    assert ranked[0]["worst_validation_accuracy"] == pytest.approx(0.55)
    assert ranked[1]["seed_sample_sd"] == 0
    assert all(len(r["per_resolution"]) == 3 for r in ranked[0]["per_seed"])
    with pytest.raises(ValueError, match="exactly once"):
        plan.architecture_ranking(
            jobs[:-1], names, values["frame_sizes"], values["seeds"]
        )
    invalid = copy.deepcopy(jobs)
    invalid[0]["result"]["validation_metrics"]["accuracy"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        plan.architecture_ranking(
            invalid, names, values["frame_sizes"], values["seeds"]
        )


def test_full_table_keeps_more_than_five_and_incomplete(tmp_path, capsys):
    rows = [
        {
            "model": f"custom_{i}",
            "status": "complete",
            "validation_accuracy": i / 10,
            "validation_loss": 1 - i / 10,
            "parameters": i + 1,
        }
        for i in range(8)
    ]
    rows += [
        {"model": "failed_model", "status": "failed"},
        {"model": "pending_model", "status": "pending"},
    ]
    written = write_full_table(tmp_path, rows)
    assert len(written) == 10
    output = capsys.readouterr().out
    assert "custom_0" in output and "custom_7" in output and "Top 5" not in output
    assert "failed_model" in output and "pending_model" in output
    assert len(study_matrix.read_json(tmp_path / "results.json")["all"]) == 10
    with (tmp_path / "results.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 10
    assert "custom_0" in (tmp_path / "results.md").read_text()


def test_ablation_recommendations_keep_single_factors_separate():
    rows = []
    for model in ("custom_4", "r3d_18", "custom_64"):
        for size in (32, 48, 64):
            rows.append(
                {
                    "model": model,
                    "status": "complete",
                    "stage": "flat" if model.startswith("custom") else "references",
                    "height": size,
                    "frames": 8,
                    "fps": 16,
                    "weight_decay": 0,
                    "validation_accuracy": 0.7,
                    "validation_loss": 0.4,
                }
            )
            if model == "custom_4":
                for wd, accuracy in ((1e-4, 0.8), (1e-3, 0.6)):
                    rows.append(
                        {
                            **rows[-1],
                            "stage": "weight_decay",
                            "weight_decay": wd,
                            "validation_accuracy": accuracy,
                        }
                    )
        if model != "custom_64":
            rows.append(
                {
                    **rows[-1],
                    "stage": "temporal",
                    "height": 48,
                    "weight_decay": 0,
                    "frames": 8,
                    "fps": 8,
                    "validation_accuracy": 0.9,
                }
            )
    report = search._ablation_recommendations(rows)
    assert set(report["models"]) == {"custom_4", "r3d_18"}
    custom = report["models"]["custom_4"]
    assert custom["weight_decay_recommendation"]["value"] == 1e-4
    assert custom["temporal_recommendation"]["fps"] == 8
    assert len(custom["weight_decay_options"]) == 3
    assert report["models"]["r3d_18"]["weight_decay_recommendation"] is None
    assert "No combined winner" in report["note"]


@pytest.fixture
def protocol(tmp_path):
    data = tmp_path / "aad"
    for label in ("a", "b"):
        (data / label).mkdir(parents=True)
        for i in range(10):
            (data / label / f"{i}.mp4").write_bytes(
                f"metadata fixture {label}/{i}".encode()
            )
    source = dataset.AHARDataset(data)
    split_path = tmp_path / "split.json"
    dataset.create_split_manifest(source, split_path)
    values = study_matrix.read_json(PROFILE)
    values["dataset"].update(path=str(data), split_manifest=str(split_path))
    values["runs_dir"] = str(tmp_path / "runs")
    path = write_json(tmp_path / "study.json", values)
    return SimpleNamespace(
        root=tmp_path, data=data, path=path, values=values, trained=[], archives=[]
    )


def fake_train(p):
    """Synthetic receipts; no claim that these metrics are real experiments."""

    def run(args):
        config = study_matrix.read_json(args[args.index("--config") + 1])
        name = args[args.index("--model") + 1]
        folder = p.root / "runs/experiments" / f"fixture_{len(p.trained):03d}"
        folder.mkdir(parents=True)
        split = study_matrix.read_json(config["split_manifest"])
        valid = [r for r in split["samples"] if r["split"] == "validation"]
        strong = name in {"custom_4", "custom_8_8", "custom_24_24_24", *plan.REFERENCES}
        predictions = []
        for i, row in enumerate(valid):
            target = row["class_index"]
            pred = target if strong or i % 2 == 0 else 1 - target
            predictions.append(
                {
                    "source": str((p.data / row["path"]).resolve()),
                    "partition": "validation",
                    "target": target,
                    "predicted": pred,
                    "loss": 0.2 + config["learning_rate"],
                    "correct": target == pred,
                }
            )
        metrics = extended_metrics(
            [r["predicted"] for r in predictions], [r["target"] for r in predictions], 2
        )
        accuracy = metrics["micro_f1"]
        metrics.update(
            loss=0.2 + config["learning_rate"],
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            f1=accuracy,
        )
        checkpoint = folder / "best.pth"
        checkpoint.write_bytes(b"synthetic test checkpoint")
        split_ref = {
            "manifest_hash": study_matrix.file_hash(config["split_manifest"]),
            "manifest_path": config["split_manifest"],
        }
        write_json(
            checkpoint.with_suffix(".json"),
            {
                "checkpoint_role": "validation_selected_lowest_loss",
                "selection_partition": "validation",
                "experiment_config": config,
                "split_manifest_hash": split_ref["manifest_hash"],
                "selected_epoch": 64,
                "seed": config["seed"],
                "model_registry_entry": {"name": name},
            },
        )
        history = write_json(folder / "history.json", {"epochs": [{"epoch": 64}]})
        prediction_path = write_json(folder / "predictions.json", predictions)
        params = (
            sum(layer[0] for layer in config["convlstm_layers"]) * 100
            if name.startswith("custom")
            else 100000
        )
        result = {
            "name": name,
            "seed": config["seed"],
            "partition": "validation",
            "test_access": "locked",
            "dataset_dir": config["dataset_dir"],
            "experiment_config": config,
            "split": split_ref,
            "num_params": params,
            "family": "ConvLSTM" if name.startswith("custom") else name,
            "validation_metrics": metrics,
            "selected_checkpoint": str(checkpoint),
            "checkpoint_selection": {"selected_epoch": 64},
            "early_stopping": {"actual_epochs": 64},
            "history": str(history),
            "validation_predictions": str(prediction_path),
            "train_time_s": 1,
        }
        if "--matched-training-evaluation" in args:
            train_records = [
                dict(
                    source=str((p.data / r["path"]).resolve()),
                    partition="train",
                    target=r["class_index"],
                    predicted=r["class_index"],
                    loss=0.1,
                )
                for r in split["samples"]
                if r["split"] == "train"
            ]
            train_metrics = extended_metrics(
                [r["predicted"] for r in train_records],
                [r["target"] for r in train_records],
                2,
            )
            train_metrics.update(loss=0.1, accuracy=1.0)
            result["training_evaluation"] = dict(
                protocol="matched_checkpoint_eval_v1",
                partition="train",
                model_mode="eval",
                augmentation=False,
                checkpoint=str(checkpoint),
                checkpoint_sha256=study_matrix.file_hash(checkpoint),
                selected_epoch=64,
                samples=len(train_records),
                metrics=train_metrics,
                predictions=str(
                    write_json(folder / "train_predictions.json", train_records)
                ),
                train_minus_validation_accuracy=1.0 - accuracy,
                validation_minus_train_loss=metrics["loss"] - 0.1,
            )
        write_json(folder / "summary.json", {"all": [result]})
        write_json(folder / "run.json", {"status": "complete"})
        p.trained.append((name, config))
        return folder

    return run


def run_protocol(p):
    return study_config.execute_study(
        p.path, *study_config.load_study(p.path), fake_train(p)
    )


def test_leaf_rejects_nonfinite_scores_short_budget_and_missing_predictions(protocol):
    p = protocol
    values, config, _ = study_config.load_study(p.path)
    row = plan.calibration_rows(values, config)[0]
    cp = write_json(p.root / "leaf.json", row["config"])
    leaf = fake_train(p)(["--config", str(cp), "--model", row["model"]])
    result = study_matrix.read_json(leaf / "summary.json")["all"][0]
    manifest = study_matrix.read_json(config.split_manifest)
    sources = {
        str((p.data / r["path"]).resolve()): r["class_index"]
        for r in manifest["samples"]
        if r["split"] == "validation"
    }
    search._validate_result(result, row, result["split"], sources, 2)
    invalid = copy.deepcopy(result)
    invalid["validation_metrics"]["macro_f1"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        search._validate_result(invalid, row, result["split"], sources, 2)
    invalid = copy.deepcopy(result)
    invalid["early_stopping"]["actual_epochs"] = 1
    with pytest.raises(ValueError, match="epoch budget"):
        search._validate_result(invalid, row, result["split"], sources, 2)
    records = study_matrix.read_json(result["validation_predictions"])
    write_json(result["validation_predictions"], records[:-1])
    with pytest.raises(ValueError, match="complete validation partition"):
        search._validate_result(result, row, result["split"], sources, 2)


def test_source_id_audit_reports_cross_split_grouping(protocol):
    p = protocol
    path = p.values["dataset"]["split_manifest"]
    manifest = study_matrix.read_json(path)
    for record in manifest["samples"]:
        record["source_id"] = record["path"]
    a = next(r for r in manifest["samples"] if r["split"] == "train")
    b = next(r for r in manifest["samples"] if r["split"] == "test")
    a["source_id"] = b["source_id"] = "shared_original"
    write_json(path, manifest)
    audit = search.audit_split(dataset.AHARDataset(p.data), path)
    assert audit["cross_split_source_ids"] == ["shared_original"]
    assert not audit["cross_split_file_hashes"]
    assert (
        not audit["source_independence_verified"] and audit["missing_source_ids"] == 0
    )


def test_complete_adaptive_pipeline_reuse_and_top3_handoff(protocol, monkeypatch):
    p = protocol
    torch.set_num_threads(1)
    monkeypatch.setattr(
        search, "archive_stage", lambda group, runs: p.archives.append(str(group))
    )
    group = run_protocol(p)
    assert len(p.archives) == 8
    assert len(p.trained) <= 156
    state = study_matrix.read_json(group / "progress.json")
    assert state["status"] == "complete" and state["test_access"] == "locked"
    assert [len(s["rows"]) for s in state["stages"]][:3] == [12, 63, 6]
    assert len(state["jobs"]) < sum(len(s["rows"]) for s in state["stages"])
    selected = study_matrix.load_selection(group / "selected_config.json")
    assert selected["seeds"] == [42, 2026]
    assert selected["top3"][0]["candidate"]["convlstm_layers"] == [[4, [3, 3]]]
    assert all(len(c["sources"]) == 6 for c in selected["top3"])
    assert not {c["candidate"]["name"] for c in selected["top3"]} & set(plan.REFERENCES)
    count = len(p.trained)
    assert run_protocol(p) == group and len(p.trained) == count
    comparison = study_matrix.read_json(
        ROOT / "configs/experiments/aad_model_comparison_colab.json"
    )
    comparison.update(
        selected_config=str(group / "selected_config.json"),
        runs_dir=str(p.root / "runs"),
    )
    comparison["dataset"].update(
        path=str(p.data), split_manifest=selected["split"]["manifest_path"]
    )
    comparison_path = write_json(p.root / "comparison.json", comparison)
    _, cfg, manifest = study_config.load_study(comparison_path)
    assert (cfg.sequence_length, cfg.height, cfg.epochs, cfg.batch_size) == (
        50,
        50,
        256,
        1,
    )
    assert [c.name for c in manifest.candidates] == study_matrix.CUSTOM_SLOTS
    table = study_matrix.read_json(group / "results.json")
    assert len(table["all"]) > 100 and not table["incomplete"]
    assert (group / "width_depth_resolution.png").is_file()
    # Real verified ZIP over these synthetic evidence files, including all leaves.
    archive = study_archive.make_archive(
        p.root, p.root / "all.zip", group / "progress.json", {group}, ()
    )
    inventory = study_archive.verify_archive(archive)
    assert any("selected_config.json" in key for key in inventory)
    assert sum(key.endswith("best.pth") for key in inventory) == count
    monkeypatch.setattr(aad_study, "PROFILES", ())
    retry_zip = study_archive.retry_package(p.root, group)
    assert study_archive.verify_archive(retry_zip)
    assert len(p.trained) == count  # Packaging is independent of training/testing.
    recommendations = group / "ablation_recommendations.json"
    saved_recommendations = recommendations.read_bytes()
    write_json(recommendations, {"unverified": "replacement"})
    with pytest.raises(ValueError, match="differs"):
        study_matrix.load_selection(group / "selected_config.json")
    recommendations.write_bytes(saved_recommendations)
    original = study_matrix.read_json(group / "selected_config.json")
    changed = copy.deepcopy(original)
    changed["top3"][0]["candidate"]["convlstm_layers"] = [[99, [3, 3]]]
    write_json(group / "selected_config.json", changed)
    with pytest.raises(ValueError, match="differs"):
        study_matrix.load_selection(group / "selected_config.json")
    write_json(group / "selected_config.json", original)
    checkpoint = Path(state["jobs"][0]["result"]["selected_checkpoint"])
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="modified"):
        study_matrix.load_selection(group / "selected_config.json")


@pytest.fixture
def real_clips(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "videos"
    generator = torch.Generator().manual_seed(69)
    for label in ("a", "b"):
        (data / label).mkdir(parents=True)
        for i in range(10):
            write_video_torchvision(
                torch.rand(6, 3, 8, 8, generator=generator),
                data / label / f"{i}.mp4",
                fps=5,
            )
    return data


def test_real_timestamp_cache_and_test_isolation(real_clips, tmp_path, monkeypatch):
    from src import temporal_sampling

    source = dataset.AHARDataset(
        real_clips, 8, (8, 8), sampling_version="timestamps_v1"
    )
    tr, val, test, split = dataset.load_split_subsets(source, tmp_path / "split.json")
    locked = {str(source.samples[i][0].resolve()) for i in test.indices}
    original = temporal_sampling.video_plan
    calls = []

    def guard(path, fps, count):
        assert str(Path(path).resolve()) not in locked
        calls.append(str(path))
        return original(path, fps, count)

    monkeypatch.setattr(temporal_sampling, "video_plan", guard)
    partitions = {
        **{i: "train" for i in tr.indices},
        **{i: "validation" for i in val.indices},
    }
    audit = temporal_sampling.prepare_sampling_audit(
        source, list(partitions), partitions
    )
    assert len(audit["records"]) == len(tr) + len(val) and audit["test_clips"] == 0
    assert all(r["source_fps"] == 5 for r in audit["records"])
    assert all(r["unique_frames"] < 8 for r in audit["records"])
    study_cache._CACHE.clear()
    cached, first = study_cache.cached_training_dataset(
        source, tr.indices, val.indices, split["manifest_hash"]
    )
    reused, second = study_cache.cached_training_dataset(
        source, tr.indices, val.indices, split["manifest_hash"]
    )
    assert reused is cached and second["reused"] and first["test_clips"] == 0
    assert cached[tr.indices[0]][0].shape == (8, 3, 8, 8)
    assert len(calls) == len(tr) + len(val)
    for fps, frames, size, version in [
        (8, 8, 8, "timestamps_v1"),
        (16, 16, 8, "timestamps_v1"),
        (16, 8, 10, "timestamps_v1"),
        (16, 8, 8, "legacy"),
    ]:
        changed = dataset.AHARDataset(
            real_clips, frames, (size, size), target_fps=fps, sampling_version=version
        )
        _, info = study_cache.cached_training_dataset(
            changed, tr.indices, val.indices, split["manifest_hash"]
        )
        assert info["key"] != first["key"]
        assert len(study_cache._CACHE) == 1
    monkeypatch.setattr(study_cache, "MAX_CACHE_BYTES", 1)
    lazy, info = study_cache.cached_training_dataset(
        source, tr.indices, val.indices, split["manifest_hash"]
    )
    assert lazy is source and not study_cache._CACHE
    assert info["mode"] == "lazy_over_cache_limit"


@pytest.mark.parametrize("matched", [False, True])
def test_real_timestamp_leaf_training_and_artifacts(real_clips, tmp_path, matched):
    config = ExperimentConfig(
        dataset_dir=str(real_clips),
        runs_dir=str(tmp_path / "runs"),
        split_manifest=str(tmp_path / "split.json"),
        sequence_length=8,
        height=8,
        width=8,
        convlstm_layers=((4, (3, 3)),),
        epochs=1,
        batch_size=4,
        cache_dataset=True,
        num_workers=0,
        prediction_samples_per_category=0,
        sampling_version="timestamps_v1",
    )
    cp = config.save_json(tmp_path / "leaf.json")
    candidates = write_json(
        tmp_path / "candidates.json",
        {
            "screening_id": "timestamp_smoke",
            "candidates": [plan.candidate([4])],
        },
    )
    leaf = experiments.main(
        [
            "--config",
            str(cp),
            "--model",
            "custom_4",
            "--candidates-config",
            str(candidates),
            *(
                [
                    "--matched-training-evaluation",
                    "--per-class-reporting",
                    "--campaign-number",
                    "7",
                    "--campaign-maximum",
                    "142",
                    "--campaign-stage",
                    "flat",
                ]
                if matched
                else []
            ),
        ]
    )
    result = study_matrix.read_json(leaf / "summary.json")["all"][0]
    assert result["early_stopping"]["actual_epochs"] == 1
    if matched:
        row = {"config": config.to_dict()}
        classes = study_matrix.read_json(config.split_manifest)["class_names"]
        search.validate_matched_training(result, row, len(classes))
        assert result["training_evaluation"]["model_mode"] == "eval"
        assert "macro" not in (leaf / "results.md").read_text(encoding="utf-8")
        assert "tp" in result["validation_per_class"][0]
    else:
        assert "training_evaluation" not in result
    assert result["test_access"] == "locked"
    assert result["sampling_report"] and result["validation_metrics"]["macro_f1"] >= 0
    assert result["efficiency"]["peak_cuda_memory_bytes"] is None
    assert result["total_parameters"] == result["num_params"]
    assert (leaf / "results.csv").is_file()
    audit = study_matrix.read_json(result["sampling_report"])
    split = study_matrix.read_json(config.split_manifest)
    expected = {
        str((real_clips / r["path"]).resolve())
        for r in split["samples"]
        if r["split"] != "test"
    }
    assert {r["source"] for r in audit["records"]} == expected
    assert (
        len(study_matrix.read_json(result["validation_predictions"]))
        == split["counts"]["validation"]
    )


def test_real_variable_frame_rate_timestamps(tmp_path):
    import av
    from fractions import Fraction
    import numpy as np

    path = tmp_path / "variable.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i, millis in enumerate((0, 33, 100, 267, 333)):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), i * 40, dtype=np.uint8), format="rgb24"
            )
            frame.pts = millis
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = video_plan(path, 8, 8)
    assert result["variable_frame_rate"]
    assert result["source_frame_count"] == 5
    assert result["padding_fraction"] > 0
    assert result["target_timestamps_seconds"] == [i / 8 for i in range(8)]


def test_interrupted_job_records_pending_and_never_retries(protocol, monkeypatch):
    p = protocol
    monkeypatch.setattr(search, "archive_stage", lambda *args: None)
    calls = []

    def fail(args):
        calls.append(args)
        raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        study_config.execute_study(p.path, *study_config.load_study(p.path), fail)
    group = next((p.root / "runs/studies").iterdir())
    table = study_matrix.read_json(group / "results.json")
    assert len(table["incomplete"]) == 12
    assert table["incomplete"][0]["status"] == "failed"
    with pytest.raises(RuntimeError, match="no automatic retry"):
        study_config.execute_study(p.path, *study_config.load_study(p.path), fail)
    assert len(calls) == 1
    assert not (group / "selected_config.json").exists()


def test_missing_split_and_cross_split_duplicates_block_without_decode(
    protocol, monkeypatch
):
    p = protocol
    monkeypatch.setattr(search, "archive_stage", lambda *args: None)
    monkeypatch.setattr(
        dataset.AHARDataset,
        "__getitem__",
        lambda *args: pytest.fail("test/data decode during split audit"),
    )
    split = study_matrix.read_json(p.values["dataset"]["split_manifest"])
    a = next(r for r in split["samples"] if r["split"] == "train")
    b = next(r for r in split["samples"] if r["split"] == "test")
    (p.data / b["path"]).write_bytes((p.data / a["path"]).read_bytes())
    with pytest.raises(ValueError, match="duplication"):
        run_protocol(p)
    assert not p.trained
    values = copy.deepcopy(p.values)
    values["dataset"]["split_manifest"] = str(p.root / "missing.json")
    path = write_json(p.root / "missing_profile.json", values)
    with pytest.raises(ValueError, match="existing AAD split"):
        study_config.execute_study(path, *study_config.load_study(path), fake_train(p))
    assert not (p.root / "missing.json").exists()


def test_strict_decode_failure_cannot_substitute_another_partition(protocol):
    source = dataset.AHARDataset(protocol.data, sampling_version="timestamps_v1")
    with pytest.raises(Exception):
        source[0]  # Invalid fixture video must raise, not recurse into the next clip.


def test_notebook_source_matches_export_and_new_entry_is_explicit():
    book = study_matrix.read_json(ROOT / "notebooks/aad_experiment_workflow.ipynb")
    source = "".join(
        c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        for c in book["cells"]
        if c["cell_type"] == "code"
    )
    script = (ROOT / "notebooks/aad_experiment_workflow.py").read_text(encoding="utf-8")
    assert source.strip() == script.removeprefix("# In[ ]:\n").strip()
    assert 'WORKFLOW_STAGE = "all"' in source
    assert "run_full_aad_study(IMPLEMENTATION_ROOT)" in source
    assert 'elif WORKFLOW_STAGE == "capacity_search"' in source
    assert 'elif WORKFLOW_STAGE == "search"' in source
