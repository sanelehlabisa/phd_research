"""088: pooled receipts, per-class/count evidence and explicit legacy compatibility."""

from copy import deepcopy
from pathlib import Path

import pytest

from src import capacity_config, capacity_search, study_config, study_matrix
from src.metrics import metric_protocol
from src.study_reporting import (
    LEGACY_REPORTING,
    POOLED_REPORTING,
    confusion_counts,
    pooled_metrics,
    reporting_protocol,
    result_row,
    validate_pooled_evidence,
)
from src.utils import write_json
from src.wide_protocol import class_counts
from test_capacity_search import fake_train, protocol  # synthetic files only


def pooled_fixture(p):
    study, config, _ = study_config.load_study(p.path)
    row = capacity_config.calibration_rows(study, config)[0]
    row["matched_checkpoint_metrics"] = True
    cfg = write_json(p.root / "leaf.json", row["config"])
    folder = fake_train(p)(
        ["--config", str(cfg), "--model", row["model"], "--matched-training-evaluation"]
    )
    result = study_matrix.read_json(folder / "summary.json")["all"][0]
    manifest = study_matrix.read_json(config.split_manifest)
    classes = manifest["class_names"]
    records = study_matrix.read_json(result["validation_predictions"])
    result.update(
        reporting_protocol=POOLED_REPORTING,
        metric_protocol=metric_protocol(),
        validation_metrics=pooled_metrics(records, classes),
        validation_per_class=class_counts(records, classes),
        validation_confusion_matrix=confusion_counts(records, classes),
    )
    sidecar = Path(result["selected_checkpoint"]).with_suffix(".json")
    write_json(
        sidecar,
        {**study_matrix.read_json(sidecar), "metric_protocol": metric_protocol()},
    )
    train = result["training_evaluation"]
    train_records = study_matrix.read_json(train["predictions"])
    train.update(
        protocol="matched_checkpoint_eval_v2",
        reporting_protocol=POOLED_REPORTING,
        metrics=pooled_metrics(train_records, classes),
        per_class=class_counts(train_records, classes),
        confusion_matrix=confusion_counts(train_records, classes),
    )
    sources = {
        str((p.data / r["path"]).resolve()): r["class_index"]
        for r in manifest["samples"]
        if r["split"] == "validation"
    }
    return result, row, sources, classes


def test_pooled_counts_include_absent_classes_without_averaging():
    records = [
        dict(target=t, predicted=p, loss=i / 10)
        for i, (t, p) in enumerate([(0, 0), (0, 0), (0, 0), (1, 0)])
    ]
    names = ["a", "b", "absent"]
    scores = pooled_metrics(records, names)
    assert scores == pytest.approx(
        dict(loss=0.15, accuracy=0.75, precision=0.75, recall=0.75, f1=0.75)
    )
    counts = class_counts(records, names)
    assert counts[0]["tp"] == 3 and counts[0]["fp"] == 1
    assert counts[1]["fn"] == 1 and counts[2]["support"] == 0
    validate_pooled_evidence(
        scores, records, names, counts, confusion_counts(records, names)
    )
    records[0]["loss"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pooled_metrics(records, names)


def test_pooled_campaign_and_matched_receipts_do_not_compute_legacy_metrics(
    protocol, monkeypatch
):
    result, row, sources, classes = pooled_fixture(protocol)

    def forbidden(*args):
        raise AssertionError("New reports must not compute legacy averages")

    monkeypatch.setattr(capacity_search, "extended_metrics", forbidden)
    capacity_search._validate_result(
        result, row, result["split"], sources, len(classes)
    )
    table = result_row(result, raw=True)
    assert table["reporting_protocol"] == POOLED_REPORTING
    assert table["validation_micro_f1"] == table["validation_accuracy"]
    assert table["selected_training_micro_f1"] == table["selected_training_accuracy"]
    assert "macro" not in str(table) and "balanced_accuracy" not in str(table)
    assert "tp" in table["validation_per_class"][0]


def test_comparison_table_keeps_pooled_scores_for_older_input_configs(
    protocol, monkeypatch
):
    result, _, _, _ = pooled_fixture(protocol)
    result["experiment_config"].pop("target_fps", None)
    result.update(
        weight_tensor_bytes=100,
        checkpoint_bytes=200,
        validation_metric_pass_seconds=0.1,
        training_curves="fixture.png",
    )
    report = dict(
        reporting_protocol=POOLED_REPORTING,
        metric_pass_seconds=0.1,
        metrics=result["validation_metrics"],
        predictions=result["validation_predictions"],
        artifacts={"confusion_matrix": result["validation_confusion_matrix"]},
    )
    import src.study_reporting as reporting

    def forbidden(*args):
        raise AssertionError("New comparison must not compute legacy averages")

    monkeypatch.setattr(reporting, "extended_metrics", forbidden)
    row = study_matrix._table_row(
        dict(result=result, run_dir="fixture"), report, extended=True
    )
    assert row["target_fps"] == 16
    assert row["test_micro_f1"] == row["test_accuracy"]
    assert "tp" in row["test_per_class"][0]
    assert "macro" not in str(row) and "balanced_accuracy" not in str(row)


@pytest.mark.parametrize(
    "change",
    [
        "accuracy",
        "extra_average",
        "per_class",
        "confusion",
        "protocol",
        "split",
        "seed",
        "matched_counts",
        "matched_gap",
        "matched_protocol",
        "missing_predictions",
    ],
)
def test_campaign_rejects_tampered_evidence(protocol, change):
    result, row, sources, classes = pooled_fixture(protocol)
    if change == "accuracy":
        result["validation_metrics"]["accuracy"] = -1
    elif change == "extra_average":
        result["validation_metrics"]["macro_f1"] = 0.1
    elif change == "per_class":
        result["validation_per_class"][0]["support"] += 1
    elif change == "confusion":
        result["validation_confusion_matrix"]["matrix"][0][0] += 1
    elif change == "protocol":
        result["reporting_protocol"] = "unsupported"
    elif change == "split":
        result["split"]["manifest_hash"] = "changed"
    elif change == "seed":
        sidecar = Path(result["selected_checkpoint"]).with_suffix(".json")
        write_json(sidecar, {**study_matrix.read_json(sidecar), "seed": 99})
    elif change == "matched_counts":
        result["training_evaluation"]["per_class"][0]["tp"] += 1
    elif change == "matched_gap":
        result["training_evaluation"]["train_minus_validation_accuracy"] += 0.2
    elif change == "matched_protocol":
        result["training_evaluation"]["protocol"] = "matched_checkpoint_eval_v1"
    else:
        records = study_matrix.read_json(result["validation_predictions"])
        write_json(result["validation_predictions"], records[:-1])
    with pytest.raises(ValueError):
        capacity_search._validate_result(
            result, row, result["split"], sources, len(classes)
        )


def test_legacy_verification_keeps_original_scores_and_hashes(protocol):
    p = protocol
    study, config, _ = study_config.load_study(p.path)
    row = capacity_config.calibration_rows(study, config)[0]
    cfg = write_json(p.root / "legacy.json", row["config"])
    folder = fake_train(p)(["--config", str(cfg), "--model", row["model"]])
    result = study_matrix.read_json(folder / "summary.json")["all"][0]
    original = deepcopy(result)
    before = study_matrix.inventory(folder)
    records = study_matrix.read_json(result["validation_predictions"])
    sources = {r["source"]: r["target"] for r in records}
    capacity_search._validate_result(result, row, result["split"], sources, 2)
    assert reporting_protocol(result) == LEGACY_REPORTING
    assert result_row(result)["reporting_protocol"] == LEGACY_REPORTING
    assert (
        result_row(result)["validation_macro_f1"]
        == result["validation_metrics"]["macro_f1"]
    )
    assert result == original and study_matrix.inventory(folder) == before
    result["validation_metrics"]["macro_f1"] -= 0.2
    with pytest.raises(ValueError, match="reconcile"):
        capacity_search._validate_result(result, row, result["split"], sources, 2)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "targets",
        "confusion",
        "per_class",
        "checkpoint",
        "metrics",
        "classes",
        "manifest",
    ],
)
def test_frozen_test_reconciles_counts_and_keeps_provenance_guards(protocol, change):
    result, row, _, classes = pooled_fixture(protocol)
    config = {**row["config"], "checkpoint_path": result["selected_checkpoint"]}
    manifest = study_matrix.read_json(config["split_manifest"])
    records = [
        dict(
            source=str((protocol.data / r["path"]).resolve()),
            partition="test",
            target=r["class_index"],
            predicted=r["class_index"],
            loss=0.25,
        )
        for r in manifest["samples"]
        if r["split"] == "test"
    ]
    sources = {r["source"] for r in records}
    predictions = write_json(protocol.root / "test_predictions.json", records)
    report = dict(
        partition="test",
        reporting_protocol=POOLED_REPORTING,
        metric_protocol=metric_protocol(),
        classes=classes,
        checkpoint={"path": result["selected_checkpoint"]},
        split=result["split"],
        experiment_config=row["config"],
        predictions=str(predictions),
        metrics=pooled_metrics(records, classes),
        per_class=class_counts(records, classes),
        artifacts={"confusion_matrix": confusion_counts(records, classes)},
    )
    if change == "targets":
        records[0]["target"] = 1 - records[0]["target"]
        write_json(predictions, records)
        # Internally consistent forged counts must still fail the manifest check.
        report.update(
            metrics=pooled_metrics(records, classes),
            per_class=class_counts(records, classes),
            artifacts={"confusion_matrix": confusion_counts(records, classes)},
        )
    elif change == "confusion":
        report["artifacts"]["confusion_matrix"]["sample_count"] += 1
    elif change == "per_class":
        report["per_class"][0]["precision"] = -1
    elif change == "checkpoint":
        report["checkpoint"]["path"] = "other.pth"
    elif change == "metrics":
        report["metrics"]["f1"] = float("nan")
    elif change == "classes":
        report["classes"] = list(reversed(classes))
    elif change == "manifest":
        manifest["class_names"] = list(reversed(classes))
        write_json(config["split_manifest"], manifest)
    if change:
        with pytest.raises(ValueError):
            study_matrix._validate_test(report, config, result["split"], sources)
    else:
        study_matrix._validate_test(report, config, result["split"], sources)
