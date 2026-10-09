"""Ticket 073: bounded shapes, exact selection, source review and automatic handoff."""

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from src import capacity_config as plan, capacity_search as search
from src import study_config, study_matrix as matrix, dataset
from src.study_reporting import extended_metrics
from src.utils import write_json
from notebooks.utils import aad_study, study_archive
from test_capacity_search import protocol, fake_train, ROOT

PROFILE = ROOT / "configs/experiments/aad_shape_search_colab.json"


def test_bounded_shapes_and_caps():
    study, config, _ = study_config.load_study(PROFILE)
    specs = plan.flats(study) + plan.shape_candidates()
    stacks = [tuple(c[0] for c in x["convlstm_layers"]) for x in specs]
    assert len(stacks) == len(set(stacks)) == 34
    assert (16, 24) in stacks and (24, 16) in stacks
    assert (32, 32, 32) in stacks and (64, 64, 64) in stacks
    assert not any(48 in s or len(s) > 3 for s in stacks)
    for depth, count in ((1, 1), (2, 3), (3, 13)):
        patterns = {
            tuple(sorted(set(s)).index(w) for w in s) for s in stacks if len(s) == depth
        }
        assert len(patterns) == count
    assert len(plan.flats(study)) * 3 == 48
    assert len(plan.shape_candidates()) * 3 == 54
    assert sum(plan.stage_limits(study).values()) == study["max_runs"] == 177
    assert config.epochs == 128
    final = matrix.read_json(
        ROOT / "configs/experiments/aad_final_comparison_colab.json"
    )
    assert (
        final["training"]["epochs"],
        final["minimum_epochs"],
        final["training"]["early_stopping_patience"],
    ) == (512, 128, 32)
    assert final["models"] == matrix.COMPARISON_MODELS


def result(name, size, count, total, loss, noise=0):
    cfg = study_config.load_study(PROFILE)[1].to_dict()
    cfg.update(height=size, width=size)
    return dict(
        name=name,
        seed=42,
        experiment_config=cfg,
        partition="validation",
        test_access="locked",
        split={"manifest_hash": "fixed"},
        dataset_dir="fixed",
        num_params=1,
        validation_metrics={"accuracy": count / total + noise, "loss": loss},
        validation_confusion_matrix={"matrix": [[count, total - count], [0, 0]]},
    )


def test_exact_tie_loss_not_float_noise_and_legacy_unchanged():
    rows = [
        result(n, s, c, 160, loss, noise)
        for n, counts, loss, noise in [
            ("flat", (146, 142, 136), 0.7607, 1e-8),
            ("taper", (152, 148, 124), 0.6218, -1e-8),
        ]
        for s, c in zip((32, 48, 64), counts)
    ]
    assert (
        matrix.rank_resolutions(rows, ["flat", "taper"], [32, 48, 64])[0]["name"]
        == "flat"
    )
    ranked = matrix.rank_resolutions(rows, ["flat", "taper"], [32, 48, 64], exact=True)
    assert ranked[0]["name"] == "taper"
    assert ranked[0]["accuracy_fraction"] == [53, 60]
    # Equal resolution weight, never pooled video-count weighting.
    varied = [
        result("x", s, c, n, 0.1) for s, c, n in ((32, 1, 2), (48, 9, 10), (64, 1, 1))
    ]
    r = matrix.rank_resolutions(varied, ["x"], [32, 48, 64], exact=True)[0]
    assert Fraction(*r["accuracy_fraction"]) == Fraction(4, 5)
    better = result("x", 32, 999999, 1000000, 0.2)
    worse = result("y", 32, 999998, 1000000, 0.1)
    assert (
        matrix.rank_resolutions([better, worse], ["x", "y"], [32], exact=True)[0][
            "name"
        ]
        == "x"
    )
    with pytest.raises(ValueError, match="counts"):
        matrix.validation_accuracy(
            {**better, "validation_metrics": {"accuracy": 0.5}}, True
        )
    with pytest.raises(ValueError, match="requires"):
        matrix.validation_accuracy({"validation_metrics": {"accuracy": 0.5}}, True)


def review_for(split_path):
    manifest = matrix.read_json(split_path)
    return dict(
        schema_version=1,
        dataset_name=manifest["dataset_name"],
        inventory_hash=manifest["inventory_hash"],
        split_content_sha256=matrix.value_hash(manifest),
        disposition="independent_recordings_user_confirmed",
        reviewer="synthetic test fixture, not research evidence",
    )


def test_review_is_identity_bound_and_never_overrides_duplicates(protocol):
    p = protocol
    split = p.values["dataset"]["split_manifest"]
    review = write_json(p.root / "review.json", review_for(split))
    source = dataset.AHARDataset(p.data)
    audit = search.audit_split(source, split, review)
    assert audit["source_review"]["reviewer"].startswith("synthetic")
    assert not audit["source_independence_verified"]
    assert audit["filename_only_cross_split_groups"]
    # JSON whitespace/newlines do not change the reviewed assignment.
    Path(split).write_text(json.dumps(matrix.read_json(split)), encoding="utf8")
    assert search.audit_split(source, split, review)["source_review"]
    manifest = matrix.read_json(split)
    train = next(r for r in manifest["samples"] if r["split"] == "train")
    test = next(r for r in manifest["samples"] if r["split"] == "test")
    (p.data / test["path"]).write_bytes((p.data / train["path"]).read_bytes())
    assert search.audit_split(source, split, review)["cross_split_file_hashes"]
    changed = matrix.read_json(review)
    changed["split_content_sha256"] = "wrong"
    write_json(review, changed)
    with pytest.raises(ValueError, match="does not cover"):
        search.audit_split(source, split, review)


def test_committed_review_matches_current_manifest_semantically():
    split = ROOT / "splits/abnormal-activities-dataset_seed42.json"
    review = matrix.read_json(ROOT / "configs/experiments/aad_source_review.json")
    assert review["split_content_sha256"] == matrix.value_hash(matrix.read_json(split))


def test_exact_confirmation_preserves_equal_seed_weights_and_tie_breaks():
    jobs = []
    for seed, correct in ((42, 80), (2026, 120)):
        for size in (32, 48, 64):
            for name, loss in (("a", 0.4), ("b", 0.3)):
                r = result(
                    name, size, correct, 160, loss, 1e-8 if name == "a" else -1e-8
                )
                r["seed"] = r["experiment_config"]["seed"] = seed
                jobs.append({"result": r})
    ranked = plan.architecture_ranking(
        jobs, ["a", "b"], [32, 48, 64], (42, 2026), exact=True
    )
    assert ranked[0]["name"] == "b"
    assert Fraction(*ranked[0]["accuracy_fraction"]) == Fraction(5, 8)
    assert ranked[0]["seed_sample_sd"] == pytest.approx(0.125 * 2**0.5)
    a, b = result("a", 32, 1, 2, 0.3), result("b", 32, 1, 2, 0.3)
    a["num_params"] = 2
    assert (
        matrix.rank_resolutions([a, b], ["a", "b"], [32], exact=True)[0]["name"] == "b"
    )
    a["num_params"] = 1
    assert (
        matrix.rank_resolutions([a, b], ["a", "b"], [32], exact=True)[0]["name"] == "a"
    )


def test_full_default_workflow_search_freeze_test_zip_and_reuse(protocol, monkeypatch):
    p = protocol
    p.tested = []
    values = matrix.read_json(PROFILE)
    values["dataset"].update(
        path=str(p.data), split_manifest=p.values["dataset"]["split_manifest"]
    )
    values["runs_dir"] = str(p.root / "runs")
    for _, rel in aad_study.PROFILES:
        write_json(p.root / rel, matrix.read_json(ROOT / rel))
    write_json(p.root / "configs/experiments/aad_shape_search_colab.json", values)
    final = matrix.read_json(
        ROOT / "configs/experiments/aad_final_comparison_colab.json"
    )
    final["runs_dir"] = str(p.root / "runs")
    write_json(p.root / "configs/experiments/aad_final_comparison_colab.json", final)
    review = write_json(
        p.root / "review.json", review_for(values["dataset"]["split_manifest"])
    )
    real_audit = search.audit_split
    monkeypatch.setattr(
        search,
        "audit_split",
        lambda source, split, review_path=None: real_audit(source, split, review),
    )
    monkeypatch.setattr(
        search, "archive_stage", lambda group, runs: p.archives.append(str(group))
    )
    monkeypatch.setattr(
        aad_study, "resolve_dataset", lambda name, path, root: str(p.data)
    )
    downloads = []
    monkeypatch.setattr(
        aad_study, "_download_archive", lambda path: downloads.append(path)
    )
    base_train = fake_train(p)

    def train(args):
        folder = base_train(args)
        result = matrix.read_json(folder / "summary.json")["all"][0]
        if result["experiment_config"]["epochs"] == 512:
            name = result["name"]
            records = matrix.read_json(result["validation_predictions"])
            for i, r in enumerate(records):
                r["predicted"] = (
                    r["target"]
                    if name == "custom_top2" or i % 2 == 0
                    else 1 - r["target"]
                )
                r["correct"] = r["predicted"] == r["target"]
            write_json(result["validation_predictions"], records)
            metrics = extended_metrics(
                [r["predicted"] for r in records], [r["target"] for r in records], 2
            )
            result["validation_metrics"].update(metrics)
            result["validation_metrics"].update(
                {
                    k: metrics["micro_f1"]
                    for k in ("accuracy", "precision", "recall", "f1")
                }
            )
            result["early_stopping"]["actual_epochs"] = 128
            result["checkpoint_selection"]["selected_epoch"] = 128
            side = Path(result["selected_checkpoint"]).with_suffix(".json")
            write_json(side, {**matrix.read_json(side), "selected_epoch": 128})
        curve = folder / "curve.png"
        curve.write_bytes(b"synthetic plot")
        result.update(
            weight_tensor_bytes=400,
            checkpoint_bytes=25,
            validation_metric_pass_seconds=0.1,
            training_curves=str(curve),
        )
        write_json(folder / "summary.json", {"all": [result]})
        return folder

    def test(path):
        cfg = matrix.read_json(path)
        frozen = matrix.read_json(Path(path).parent.parent / "validation_frozen.json")
        assert len(frozen["models"]) == 8 and frozen["best_custom"] == "custom_top2"
        assert sum(c["epochs"] == 512 for _, c in p.trained) == 8
        split = matrix.read_json(cfg["split_manifest"])
        records = [
            dict(
                source=str((p.data / r["path"]).resolve()),
                partition="test",
                target=r["class_index"],
                predicted=r["class_index"],
                loss=0.0,
            )
            for r in split["samples"]
            if r["split"] == "test"
        ]
        folder = p.root / "runs/evaluate" / Path(path).stem
        predictions = write_json(folder / "predictions.json", records)
        report = dict(
            partition="test",
            split={"manifest_hash": matrix.file_hash(cfg["split_manifest"])},
            checkpoint={"path": cfg["checkpoint_path"]},
            experiment_config={k: v for k, v in cfg.items() if k != "checkpoint_path"},
            predictions=str(predictions),
            metrics=dict(loss=0.0, accuracy=1.0, precision=1.0, recall=1.0, f1=1.0),
            metric_pass_seconds=0.1,
            prediction_examples={
                "records": [],
                "saved_counts": {"correct": 0, "incorrect": 0},
            },
        )
        write_json(folder / "metrics/final.json", report)
        write_json(folder / "run.json", {"status": "complete"})
        p.tested.append(Path(path).stem)
        return folder

    def command(args, root, log_path=None):
        path = Path(args[args.index("--config") + 1])
        loaded = study_config.load_study(path)
        if "--list-plan" not in args:
            study_config.execute_study(path, *loaded, train, test)

    monkeypatch.setattr(aad_study, "_run_command", command)
    archive = aad_study.run_full_aad_study(p.root)
    assert archive.is_file() and len(downloads) == 1
    count = len(p.trained)
    assert 102 < count - 8 <= 177 and len(p.tested) == 8
    entries = study_archive.verify_archive(archive)
    assert any(k.endswith("comparison.csv") for k in entries)
    assert any(k.endswith("split_audit.json") for k in entries)
    comparison = next((p.root / "runs/studies").glob("*top3_comparison"))
    rows = matrix.read_json(comparison / "comparison.json")["table"]
    assert len(rows) == 8 and all(
        "test_macro_f1" in r and "validation_macro_precision" in r for r in rows
    )
    assert aad_study.run_full_aad_study(p.root) == archive
    assert len(p.trained) == count and len(p.tested) == 8
    # A partial test marker must never trigger a repeat.
    receipt = comparison / "test_receipts/custom_top1.json"
    write_json(receipt, {"status": "started"})
    with pytest.raises(RuntimeError):
        aad_study.run_full_aad_study(p.root)
    assert len(p.trained) == count and len(p.tested) == 8


def test_failed_search_never_starts_comparison(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("incomplete search")

    monkeypatch.setattr(aad_study, "run_capacity_study", fail)
    monkeypatch.setattr(
        aad_study,
        "run_saved_comparison",
        lambda *a, **kw: pytest.fail("comparison must not start"),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        aad_study.run_full_aad_study(tmp_path)


def test_ambiguous_saved_request_is_not_newest_directory(tmp_path):
    for n in ("one", "two"):
        write_json(
            tmp_path / "runs/notebook_studies" / n / "request.json",
            dict(request={"stage": "search"}, profile=n),
        )
    with pytest.raises(ValueError, match="Multiple"):
        aad_study._saved_request(tmp_path, {"stage": "search"})
