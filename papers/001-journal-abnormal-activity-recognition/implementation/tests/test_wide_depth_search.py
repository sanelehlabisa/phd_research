"""075: bounded candidate design, final-input evidence, safety and presentation."""

import copy
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest
import torch

from src import capacity_config as plan, capacity_search as search
from src import comparison_recipe as recipe, experiments, study_config
from src import study_matrix as matrix, wide_protocol as wide
from src.experiment_config import ExperimentConfig
from src.model import CustomConvLSTM
from src.study_resources import select_capacity_batch
from src.study_reporting import write_full_table, result_row
from src.utils import write_json
from test_capacity_search import protocol, fake_train, ROOT  # noqa: F401

PROFILE = ROOT / "configs/experiments/aad_wide_depth_search_colab.json"


def test_legacy_selection_note_remains_byte_compatible():
    assert plan.protocol_note({"schema_version": 3}) == (
        "Evidence-guided finite 1-3-layer search, not unbiased/exhaustive or a global optimum. "
        "Two seeds give limited uncertainty evidence; temporal ablations use 48x48. "
        "From-scratch low-resolution references are not exhaustively tuned baselines. "
        "Search never decodes test; the automatic workflow hands verified selection to "
        "fresh comparison training and validation-frozen testing. Filename independence "
        "is user-attested, not proof of subject/scene independence."
    )


def test_exact_manifest_budget_and_confirmation_controls():
    study, config, _ = study_config.load_study(PROFILE)
    stacks = [
        tuple(v[0] for v in c["convlstm_layers"])
        for c in plan.flats(study) + plan.shape_candidates(study)
    ]
    expected = {(w,) * d for d in (1, 2, 3, 4) for w in (32, 64, 96, 128)} - {(96,)}
    expected |= {(w, w, w) for w in (8, 16, 48, 80)}
    expected |= {(w, 64, 64) for w in (32, 96, 128)}
    expected |= {(64, w, 64) for w in (32, 96, 128)}
    expected |= {(64, 64, w) for w in (32, 96, 128)}
    expected |= {(32, 64), (64, 32), (16, 32, 32), (64,) * 5}
    assert set(stacks) == expected and len(stacks) == 32
    assert Counter(map(len, stacks)) == {1: 3, 2: 6, 3: 18, 4: 4, 5: 1}
    assert study["frame_sizes"] == [32, 64]
    assert study["max_runs"] == sum(plan.stage_limits(study).values()) == 131
    assert config.epochs == 256 and config.height == config.width == 64
    lrs = {"custom": 0.003, "r3d_18": 0.001, "swin3d_t": 0.01}
    candidates = plan.flats(study) + plan.shape_candidates(study)
    base = plan.matrix_rows(study, config, candidates, lrs, "flat")
    assert len(base) == 64 and all(r["minimum_epochs"] == 64 for r in base)
    shortlist = [plan.candidate([w] * 3) for w in (8, 16, 48, 80)] + [
        plan.candidate([32, 64])
    ]
    confirmed = plan.confirmation_specs(study, shortlist, 64)
    assert (
        len(confirmed) == 12
        and len({c["name"] if isinstance(c, dict) else c for c in confirmed}) == 12
    )
    assert any(
        isinstance(c, dict) and c["name"] == "custom_64_64_64_64_64" for c in confirmed
    )
    for key, value in (("max_runs", 132), ("frame_sizes", [32, 48, 64])):
        changed = copy.deepcopy(study)
        changed[key] = value
        with pytest.raises(ValueError):
            plan.load_capacity(changed)


def test_five_layer_forward_backward_and_adapted_paper_identity():
    torch.set_num_threads(1)
    model = CustomConvLSTM(3, layers=[(64, (3, 3))] * 5)
    output = model(torch.randn(2, 2, 3, 8, 8))
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert output.shape == (2, 3)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    entry = experiments.model_registry(selected_models=["paper_convlstm_adapted"])[0]
    assert "adapted" in entry["role"] and "not an exact" in entry["role"]
    assert "paper_convlstm_adapted" not in {
        r["name"] for r in experiments.model_registry()
    }
    with torch.device("meta"):
        adapted = experiments.build_registered_model(
            "paper_convlstm_adapted", 11, (3, 64, 64), 16
        )
        native = experiments.build_registered_model(
            "paper_convlstm_published", 11, (3, 50, 50), 50
        )
    assert sum(p.numel() for p in adapted.parameters()) != sum(
        p.numel() for p in native.parameters()
    )


def test_preflight_common_batches_cover_all_bounds_and_full_final_families():
    report = select_capacity_batch(
        32,
        lambda s: {
            "memory_safe": s["batch_size"] <= (8 if s["phase"] == "search" else 2)
        },
    )
    assert report["status"] == "complete"
    assert (report["selected_batch_size"], report["comparison_batch_size"]) == (8, 2)
    successful = [r for r in report["results"] if r["memory_safe"]]
    assert {r["model"] for r in successful if r["phase"] == "search"} == {
        "custom_bound",
        "custom_depth5",
        "r3d_18",
        "swin3d_t",
    }
    assert {r["model"] for r in successful if r["phase"] == "final_input"} == {
        "custom_bound",
        "custom_depth5",
        "paper_convlstm_adapted",
        "r3d_18",
        "mc3_18",
        "swin3d_t",
        "swin3d_s",
    }
    assert all(
        r["frames"] == (8 if r["phase"] == "search" else 16) and r["size"] == 64
        for r in successful
    )

    def oom(spec):
        if spec["model"] == "paper_convlstm_adapted":
            raise torch.cuda.OutOfMemoryError("synthetic")
        return {"memory_safe": True}

    failed = select_capacity_batch(32, oom)
    assert failed["status"] == "failed" and failed["selected_batch_size"] is None
    assert any(r.get("reason") == "CUDA OOM" for r in failed["results"])


def test_global_progress_not_leaf_one_of_one():
    args = experiments.parser.parse_args(
        [
            "--campaign-number",
            "117",
            "--campaign-maximum",
            "142",
            "--campaign-stage",
            "temporal",
        ]
    )
    label = experiments.campaign_label(args)
    assert label == "Experiment 117/142 maximum | temporal"
    assert experiments.campaign_label(Namespace()) == ""
    with pytest.raises(ValueError):
        experiments.campaign_label(
            Namespace(campaign_number=143, campaign_maximum=142, campaign_stage="flat")
        )


def make_fps_jobs(p):
    study, config, _ = study_config.load_study(PROFILE)
    config = ExperimentConfig.from_mapping(
        {
            **config.to_dict(),
            "dataset_dir": str(p.data),
            "split_manifest": p.values["dataset"]["split_manifest"],
            "runs_dir": str(p.root / "runs"),
            "batch_size": 2,
        }
    )
    shortlist = [plan.candidate([w] * 3) for w in (32, 64, 128)]
    rows = plan.ablation_rows(
        study,
        config,
        shortlist,
        {"custom": 0.003, "r3d_18": 0.001, "swin3d_t": 0.01},
        "temporal",
    )
    train = fake_train(p)
    jobs = []
    for row in rows:
        path = write_json(p.root / "fps" / f"{len(jobs)}.json", row["config"])
        folder = train(
            [
                "--config",
                str(path),
                "--model",
                row["model"],
                "--matched-training-evaluation",
            ]
        )
        result = matrix.read_json(folder / "summary.json")["all"][0]
        # All exact accuracies tie; 8 FPS wins on validation loss.
        result["validation_metrics"]["loss"] = (
            0.1 if row["config"]["target_fps"] == 8 else 0.5
        )
        result["training_evaluation"]["validation_minus_train_loss"] = (
            result["validation_metrics"]["loss"] - 0.1
        )
        predictions = matrix.read_json(result["validation_predictions"])
        for record in predictions:
            record["loss"] = result["validation_metrics"]["loss"]
        write_json(result["validation_predictions"], predictions)
        jobs.append(
            dict(
                job_id=search.job_key(row),
                experiment_number=len(jobs) + 1,
                result=result,
            )
        )
    return jobs, shortlist


def test_final_fps_complete_exact_matrix_warnings_and_raw_metrics(protocol, capsys):
    jobs, shortlist = make_fps_jobs(protocol)
    from src.study_reporting import extended_metrics

    result = jobs[0]["result"]
    records = matrix.read_json(result["validation_predictions"])
    for r in records:
        r["predicted"] = 0
    write_json(result["validation_predictions"], records)
    result["validation_metrics"]["accuracy"] = extended_metrics(
        [0] * len(records), [r["target"] for r in records], 2
    )["micro_f1"]
    report = wide.select_final_input(jobs, shortlist, 2)
    assert report["target_fps"] == 8 and len(report["rows"]) == 15
    assert all(row["training_accuracy"] == 1 for row in report["rows"])
    assert any(
        "at/below" in warning for row in report["rows"] for warning in row["warnings"]
    )
    wide.print_final_input(report)
    assert "prediction counts" in capsys.readouterr().out
    with pytest.raises(ValueError, match="fifteen"):
        wide.select_final_input(jobs[:-1], shortlist, 2)
    with pytest.raises(ValueError, match="fixed"):
        wide.select_final_input(jobs, shortlist, 1)
    for job in jobs:
        job["result"]["validation_metrics"]["loss"] = 0.5
    assert wide.select_final_input(jobs, shortlist, 2)["target_fps"] == 16
    result = jobs[0]["result"]
    row = result_row(result, raw=True, experiment_number=1)
    output = protocol.root / "raw"
    write_full_table(output, [row], raw=True)
    assert all("macro" not in k and "micro" not in k for k in row)
    assert "macro" not in (output / "results.md").read_text()
    assert (output / "results_per_class.csv").is_file()
    classes = wide.class_counts(
        [{"target": 0, "predicted": 0}, {"target": 1, "predicted": 0}],
        ["a", "b", "absent"],
    )
    assert classes[0] == dict(
        class_name="a", precision=0.5, recall=1.0, f1=2 / 3, support=1, tp=1, fp=1, fn=0
    )
    assert classes[2]["support"] == classes[2]["f1"] == 0


@pytest.mark.parametrize("prefer_lr", [False, True])
def test_joint_fps_recipe_retained_and_alias_checks(protocol, monkeypatch, prefer_lr):
    p = protocol
    jobs, shortlist = make_fps_jobs(p)
    original = next(
        j
        for j in jobs
        if j["result"]["name"] == shortlist[0]["name"]
        and j["result"]["experiment_config"]["target_fps"] == 8
    )
    result = original["result"]
    cfg = result["experiment_config"]
    folder = Path(result["selected_checkpoint"]).parent
    receipt = write_json(
        p.root / "source_receipt.json",
        dict(
            status="complete",
            result=result,
            files=matrix.inventory(folder),
            run_dir=str(folder),
        ),
    )
    original.update(receipt=str(receipt), run_dir=str(folder))
    write_json(p.root / "progress.json", {"jobs": [original]})
    selected = dict(
        protocol="capacity_top3_v3",
        source_group=str(p.root),
        top3=[{"candidate": shortlist[0]}],
        final_input={"selected_top1_job_id": original["job_id"]},
    )
    selected_path = write_json(p.root / "selected.json", selected)
    study = {"selected_config": str(selected_path)}
    base = ExperimentConfig.from_mapping(
        {
            **cfg,
            "epochs": 512,
            "early_stopping_patience": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "prediction_samples_per_category": 1,
        }
    )
    split = result["split"]
    monkeypatch.setattr(
        recipe, "search_proposal", lambda _: dict(learning_rate=0.003, weight_decay=0)
    )
    before = len(p.trained)
    base_train = fake_train(p)

    def run(args):
        folder = base_train(args)
        r = matrix.read_json(folder / "summary.json")["all"][0]
        if prefer_lr and r["experiment_config"]["learning_rate"] == 0.003:
            records = matrix.read_json(r["validation_predictions"])
            for record in records:
                record["loss"] = 0.15
            write_json(r["validation_predictions"], records)
            r["validation_metrics"]["loss"] = 0.15
            r["training_evaluation"]["validation_minus_train_loss"] = 0.05
            write_json(folder / "summary.json", {"all": [r]})
        return folder

    resolved, report = recipe.select_recipe(
        study, base, selected, p.root / "comparison", split, run, 100
    )
    assert report["selected_experiment"] == original["experiment_number"]
    assert (resolved.learning_rate, resolved.weight_decay) == (0.003, 0)
    assert len(p.trained) - before == len(report["jobs"]) == (2 if prefer_lr else 3)
    assert report["steps"][-1]["factor"] == "already_validated_joint_recipe"
    assert (
        recipe.select_recipe(
            study,
            base,
            selected,
            p.root / "comparison",
            split,
            lambda _: pytest.fail("must not retrain"),
            100,
        )[1]
        == report
    )
    common = {**cfg}
    assert recipe.verified_fps_recipe(selected, common, split)["alias"] == "custom_top1"
    with pytest.raises(ValueError, match="incompatible"):
        recipe.verified_fps_recipe(selected, {**common, "batch_size": 1}, split)
    changed = copy.deepcopy(selected)
    changed["top3"][0]["candidate"]["convlstm_layers"] = [[8, [3, 3]]]
    with pytest.raises(ValueError, match="architecture"):
        recipe.verified_fps_recipe(changed, common, split)
    Path(result["selected_checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="modified"):
        recipe.verified_fps_recipe(selected, common, split)
