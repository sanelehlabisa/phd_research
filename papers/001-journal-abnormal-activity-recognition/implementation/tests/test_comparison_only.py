"""083: portable evidence, ten-only execution and conservative test recovery."""

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import torch

from src import focused_protocol as focused, study_matrix as matrix, study_config
from src import comparison_import, comparison_recovery, study_resources, experiments
from src.experiment_config import ExperimentConfig
from src.metrics import metric_protocol, validate_selected_checkpoint
from src.study_reporting import POOLED_REPORTING, pooled_metrics, confusion_counts
from src.wide_protocol import class_counts
from src.utils import write_json
from notebooks.utils import aad_study, study_archive
from test_capacity_search import protocol, fake_train, ROOT


def checkpoint_metadata():
    return dict(
        checkpoint_role="validation_selected_lowest_loss",
        selection_partition="validation",
        selection_metric="loss",
        selection_value=0.2,
        selected_epoch=128,
        seed=42,
        validation_metrics=dict(
            loss=0.2, accuracy=0.5, precision=0.5, recall=0.5, f1=0.5
        ),
        metric_protocol=metric_protocol(),
        dataset_name="abnormal-activities-dataset",
        split_manifest_hash="split",
        model_config={"num_classes": 2},
        early_stopping={"selected_epoch": 128},
    )


def test_alias_acceptance_keeps_real_mismatch_guards():
    cp = checkpoint_metadata()
    original = deepcopy(cp)
    assert (
        validate_selected_checkpoint(cp, "aad", "split")["dataset_name"]
        == "abnormal-activities-dataset"
    )
    assert cp == original
    for dataset, split in (("vdd", "split"), ("aad", "other"), ("AAD", "split")):
        with pytest.raises(ValueError):
            validate_selected_checkpoint(cp, dataset, split)


def test_exact_ten_preflight_common_batch_and_no_fit():
    cfg = ExperimentConfig(batch_size=2, sequence_length=8, height=64, width=64)
    models = focused.SLOTS + focused.BASELINES
    probes = []

    def probe(spec):
        probes.append(spec)
        return dict(memory_safe=spec["batch_size"] == 1)

    report = study_resources.comparison_preflight(cfg, None, models, probe=probe)
    assert report["selected_batch_size"] == 1
    assert [r["model"] for r in probes if r["batch_size"] == 1] == models
    assert all(r["frames"] == 8 and r["size"] == 64 for r in probes)
    failed = study_resources.comparison_preflight(
        cfg, None, models, probe=lambda s: dict(memory_safe=False)
    )
    assert failed["selected_batch_size"] is None and failed["status"] == "failed"
    with pytest.raises(RuntimeError, match="CUDA preflight"):
        study_resources.comparison_preflight(
            cfg, None, models, device=torch.device("cpu")
        )


def test_cli_lists_ten_without_search_and_missing_evidence_fails(
    tmp_path, monkeypatch, capsys
):
    import src.capacity_search as search

    monkeypatch.setattr(
        search, "execute_capacity", lambda *a: pytest.fail("No search permitted")
    )
    path = ROOT / "configs/experiments/aad_comparison_only_local.json"
    experiments.main(["--config", str(path), "--list-plan"])
    output = capsys.readouterr().out
    assert "0 search jobs" in output and "Model 10/10: swin3d_s" in output
    values = matrix.read_json(path)
    values["selected_config"] = str(tmp_path / "missing.json")
    profile = write_json(tmp_path / "profile.json", values)
    with pytest.raises(FileNotFoundError, match="No search will be launched"):
        experiments.main(["--config", str(profile)])


def test_public_handoff_accepts_archived_split_counts_with_total(tmp_path, monkeypatch):
    import zipfile
    import json
    from src import comparison_only
    from src.dataset_source import AAD_CLASS_NAMES

    with zipfile.ZipFile(
        ROOT / "reports/2026-10-14/aad_results_essential.zip"
    ) as archive:
        name = next(
            n for n in archive.namelist() if "top3_comparison/summary.json" in n
        )
        original = json.loads(archive.read(name))["all"][0]
    split = write_json(
        tmp_path / "split.json",
        dict(class_names=list(AAD_CLASS_NAMES), counts=original["split"]["counts"]),
    )
    path = ROOT / "configs/experiments/aad_comparison_only_local.json"
    values = matrix.read_json(path)
    values["selected_config"] = str(write_json(tmp_path / "selection.json", {}))
    profile = write_json(tmp_path / "profile.json", values)
    selection = dict(
        dataset_dir=str((ROOT / values["dataset"]["path"]).resolve()),
        top5=[
            dict(candidate=dict(convlstm_layers=[[w, [3, 3]] for w in widths]))
            for widths in comparison_only.STACKS
        ],
        final_input=dict(sequence_length=8, height=64, width=64, target_fps=16),
        split=dict(manifest_path=str(split)),
    )
    monkeypatch.setattr(comparison_only, "load_selection", lambda _: selection)
    monkeypatch.setattr(aad_study, "run_saved_comparison", lambda *a, **kw: "handoff")
    assert comparison_only.run_command(profile) == "handoff"
    values["training"]["learning_rate"] = 0.00001
    write_json(profile, values)
    with pytest.raises(ValueError, match="separately approved"):
        experiments.main(["--config", str(profile)])


@pytest.fixture
def imported(protocol):
    """Create metadata for a completed synthetic 64-job search, not training."""
    p = protocol
    group = p.root / "runs/imports/search"
    study = matrix.read_json(ROOT / "configs/experiments/aad_focused_search_colab.json")
    study["dataset"].update(path=str(p.data), split_manifest=str(p.root / "split.json"))
    study["runs_dir"] = str(p.root / "runs")
    study_path = write_json(group / "study.json", study)
    _, cfg, _ = study_config.load_study(study_path)
    split = matrix.read_json(cfg.split_manifest)
    audit = dict(
        cross_split_file_hashes=[],
        cross_split_source_ids=[],
        source_review=dict(
            disposition="independent_recordings_user_confirmed",
            split_content_sha256=matrix.value_hash(split),
        ),
        files=[
            dict(
                path=str(p.data / r["path"]),
                sha256=matrix.file_hash(p.data / r["path"]),
            )
            for r in split["samples"]
        ],
    )
    write_json(group / "split_audit.json", audit)
    state = dict(
        status="complete",
        jobs=[],
        effective_config=cfg.to_dict(),
        split=dict(
            manifest_path=cfg.split_manifest,
            manifest_hash=matrix.file_hash(cfg.split_manifest),
        ),
        preparation_files={str(study_path): matrix.file_hash(study_path)},
        code_identity={"synthetic": "fixture"},
    )
    train = fake_train(p)
    for number, row in enumerate(
        (r for _, rows in focused.blocks(cfg) for r in rows), 1
    ):
        leaf = train(
            [
                "--config",
                str(write_json(p.root / "leaf.json", row["config"])),
                "--model",
                row["model"],
            ]
        )
        result = matrix.read_json(leaf / "summary.json")["all"][0]
        records = matrix.read_json(result["validation_predictions"])
        result["validation_confusion_matrix"] = confusion_counts(
            records, split["class_names"]
        )
        write_json(leaf / "summary.json", {"all": [result]})
        receipt = write_json(
            group / "jobs" / str(number) / "receipt.json",
            dict(status="complete", result=result, files=matrix.inventory(leaf)),
        )
        state["jobs"].append(
            dict(
                status="complete",
                run_dir=str(leaf),
                receipt=str(receipt),
                result=result,
            )
        )
    write_json(group / "progress.json", state)
    selected = focused.selection_bundle(
        study, state, group, focused.decisions(state["jobs"])
    )
    selection_path = write_json(group / "selected_config.json", selected)
    # Copy byte-for-byte into a different root; old paths must not be used.
    destination = p.root.parent / (p.root.name + "_relocated")
    shutil.copytree(p.root, destination)
    path = write_json(
        p.root / "import.json",
        dict(
            protocol="comparison_import_v1",
            selection=dict(
                path=str(selection_path), sha256=matrix.file_hash(selection_path)
            ),
            path_map=[dict(source=str(p.root), destination=str(destination))],
        ),
    )
    p.import_path = path
    p.imported_root = destination
    p.original_selection = selection_path
    return p


def test_import_relocates_without_rewriting_and_rejects_hash_changes(imported):
    p = imported
    original = matrix.file_hash(p.original_selection)
    selected = matrix.load_selection(p.import_path)
    assert selected["verified_import"]["search_jobs"] == 64
    assert Path(selected["dataset_dir"]) == p.imported_root / "aad"
    assert matrix.file_hash(p.original_selection) == original
    evidence = comparison_import.EvidenceImport(p.import_path)
    local = evidence.relocate(p.original_selection)
    original_bytes = local.read_bytes()
    local.write_bytes(original_bytes + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        matrix.load_selection(p.import_path)
    local.write_bytes(original_bytes)
    with pytest.raises(FileNotFoundError, match="explicit import mapping"):
        evidence.relocate("/unmapped/file.json")
    with pytest.raises(ValueError, match="traversal"):
        evidence.relocate(str(p.root) + "/../outside.json")
    local.unlink()
    with pytest.raises(FileNotFoundError, match="Required imported evidence missing"):
        matrix.load_selection(p.import_path)


def pooled_train(p):
    base = fake_train(p)

    def train(args):
        folder = base(args)
        r = matrix.read_json(folder / "summary.json")["all"][0]
        split = matrix.read_json(r["experiment_config"]["split_manifest"])
        classes = split["class_names"]
        records = matrix.read_json(r["validation_predictions"])
        r.update(
            reporting_protocol=POOLED_REPORTING,
            metric_protocol=metric_protocol(),
            validation_metrics=pooled_metrics(records, classes),
            validation_per_class=class_counts(records, classes),
            validation_confusion_matrix=confusion_counts(records, classes),
            weight_tensor_bytes=100,
            checkpoint_bytes=100,
            validation_metric_pass_seconds=0.1,
            training_curves=str(folder / "curves.png"),
        )
        (folder / "curves.png").write_bytes(b"synthetic image")
        r["early_stopping"]["actual_epochs"] = 128
        r["checkpoint_selection"]["selected_epoch"] = 128
        tr = r["training_evaluation"]
        tr_records = matrix.read_json(tr["predictions"])
        tr.update(
            protocol="matched_checkpoint_eval_v2",
            reporting_protocol=POOLED_REPORTING,
            selected_epoch=128,
            metrics=pooled_metrics(tr_records, classes),
            per_class=class_counts(tr_records, classes),
            confusion_matrix=confusion_counts(tr_records, classes),
        )
        side_path = Path(r["selected_checkpoint"]).with_suffix(".json")
        side = matrix.read_json(side_path)
        side.update(checkpoint_metadata())
        side.update(
            experiment_config=r["experiment_config"],
            dataset_name="aad",
            metric_protocol=metric_protocol(),
            validation_metrics=r["validation_metrics"],
            selection_value=r["validation_metrics"]["loss"],
            split_manifest_hash=r["split"]["manifest_hash"],
        )
        torch.save(side, r["selected_checkpoint"])
        tr["checkpoint_sha256"] = matrix.file_hash(r["selected_checkpoint"])
        write_json(side_path, side)
        write_json(folder / "summary.json", {"all": [r]})
        return folder

    return train


def pooled_test(p, calls):
    def evaluate(path):
        cfg = matrix.read_json(path)
        freeze = matrix.read_json(Path(path).parent.parent / "validation_frozen.json")
        assert len(freeze["models"]) == 10
        assert len(p.trained) == 10  # No search and all training precedes testing.
        assert cfg["prediction_samples_per_category"] == (
            5 if Path(path).stem == freeze["best_custom"] else 0
        )
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
        folder = p.root / "runs/evaluate" / (Path(path).stem + str(len(calls)))
        predictions = write_json(folder / "predictions.json", records)
        report = dict(
            reporting_protocol=POOLED_REPORTING,
            metric_protocol=metric_protocol(),
            partition="test",
            split=dict(manifest_hash=matrix.file_hash(cfg["split_manifest"])),
            classes=split["class_names"],
            checkpoint=dict(path=cfg["checkpoint_path"]),
            experiment_config={k: v for k, v in cfg.items() if k != "checkpoint_path"},
            predictions=str(predictions),
            metrics=pooled_metrics(records, split["class_names"]),
            per_class=class_counts(records, split["class_names"]),
            metric_pass_seconds=0.1,
            prediction_examples=dict(
                records=[], saved_counts=dict(correct=0, incorrect=0)
            ),
            artifacts=dict(
                confusion_matrix=confusion_counts(records, split["class_names"])
            ),
        )
        write_json(folder / "metrics/final.json", report)
        write_json(folder / "run.json", {"status": "complete"})
        calls.append(Path(path).stem)
        return folder

    return evaluate


def comparison_fixture(p, monkeypatch):
    """Use a verified-selection boundary double; import validation is tested above."""
    selected = dict(
        protocol="focused_top5_v1",
        seed=42,
        source_audit_sha256="audit",
        dataset_dir=str(p.data),
        split=dict(
            manifest_path=str(p.root / "split.json"),
            manifest_hash=matrix.file_hash(p.root / "split.json"),
        ),
        final_input=dict(height=64, width=64, sequence_length=8, target_fps=16),
        shared_recipe=dict(
            learning_rate=0.001,
            weight_decay=0,
            augment=False,
            scheduler="reduce_on_plateau",
        ),
        top5=[
            dict(candidate=c, comparison_name=n)
            for c, n in zip(focused.candidates()[:5], focused.SLOTS)
        ],
    )
    selection_path = write_json(p.root / "selected.json", selected)
    monkeypatch.setattr(matrix, "load_selection", lambda path: deepcopy(selected))
    monkeypatch.setattr(
        study_resources,
        "comparison_preflight",
        lambda cfg, manifest, models: dict(
            status="complete",
            selected_batch_size=1,
            results=[dict(model=n, memory_safe=True) for n in models],
        ),
    )
    values = matrix.read_json(
        ROOT / "configs/experiments/aad_comparison_only_local.json"
    )
    values.pop("entrypoint")
    values.update(selected_config=str(selection_path), runs_dir=str(p.root / "runs"))
    values["dataset"].update(
        path=str(p.data), split_manifest=str(p.root / "split.json")
    )
    path = write_json(p.root / "comparison.json", values)
    return path, selected


def test_ten_train_freeze_test_zip_and_no_repeats(protocol, monkeypatch, capsys):
    p = protocol
    path, _ = comparison_fixture(p, monkeypatch)
    calls = []
    monkeypatch.setattr(experiments, "main", pooled_train(p))
    import src.evaluate as evaluation

    monkeypatch.setattr(evaluation, "main", lambda args: pooled_test(p, calls)(args[1]))
    group = aad_study.run_saved_comparison(
        p.root, p.root / "selected.json", template_path=path
    )
    assert len(p.trained) == len(calls) == 10
    assert {cfg["batch_size"] for _, cfg in p.trained} == {1}
    summary = matrix.read_json(group / "comparison.json")
    assert len(summary["table"]) == 10 and all(
        r["reporting_protocol"] == POOLED_REPORTING for r in summary["table"]
    )
    assert "macro" not in str(summary["table"])
    archive = p.root / "runs/focused_archives" / group.name / "results.zip"
    assert not any(
        name.endswith(".pth") for name in study_archive.verify_archive(archive)
    )
    assert "Experiment 10/10 maximum" in capsys.readouterr().out
    assert (
        aad_study.run_saved_comparison(
            p.root, p.root / "selected.json", template_path=path
        )
        == group
    )
    assert len(p.trained) == len(calls) == 10


@pytest.mark.parametrize(
    "phase",
    ["before_scoring", "scoring_started", "complete", "legacy_alias", "legacy_unknown"],
)
def test_only_proven_pre_scoring_failures_allow_recovery(tmp_path, phase):
    failure = tmp_path / "failed"
    checkpoint = str(tmp_path / "best.pth")
    cfg = write_json(failure / "config.json", {})
    error = (
        "Checkpoint is incompatible: checkpoint dataset does not match the requested dataset"
        if phase == "legacy_alias"
        else "some later error"
    )
    run_path = write_json(
        failure / "run.json",
        dict(
            status="failed",
            error=error,
            arguments=dict(checkpoint_path=checkpoint, config_file=str(cfg)),
        ),
    )
    files = {str(run_path): matrix.file_hash(run_path)}
    if not phase.startswith("legacy"):
        marker = write_json(
            failure / "evaluation_phase.json",
            dict(
                phase=phase,
                checkpoint_sha256="weights",
                evaluation_config_sha256=matrix.file_hash(cfg),
            ),
        )
        files[str(marker)] = matrix.file_hash(marker)
    path = write_json(
        tmp_path / "import.json",
        dict(
            protocol="comparison_import_v1",
            path_map=[dict(source=str(tmp_path), destination=str(tmp_path))],
            recovery=dict(failure_files=files),
        ),
    )
    evidence = comparison_import.EvidenceImport(path)
    receipt = dict(
        status="failed",
        freeze_sha256="freeze",
        partial_runs=[str(failure)],
        error=error,
    )
    if phase in {"before_scoring", "legacy_alias"}:
        comparison_recovery.verify_before_scoring(
            evidence, receipt, "freeze", {checkpoint: "weights"}
        )
        write_json(failure / "test_predictions.json", [])
        with pytest.raises(ValueError, match="Partial/completed"):
            comparison_recovery.verify_before_scoring(
                evidence, receipt, "freeze", {checkpoint: "weights"}
            )
    else:
        with pytest.raises(ValueError):
            comparison_recovery.verify_before_scoring(
                evidence, receipt, "freeze", {checkpoint: "weights"}
            )
    receipt["status"] = "complete"
    with pytest.raises(ValueError, match="Completed"):
        comparison_recovery.verify_before_scoring(
            evidence, receipt, "freeze", {checkpoint: "weights"}
        )


def test_checkpoint_recovery_claim_survives_renamed_profile(tmp_path):
    frozen = dict(split_sha256="split", models=[dict(checkpoint_sha256="weights")])
    first = tmp_path / "first"
    comparison_recovery.claim_recovery(tmp_path, first, frozen)
    comparison_recovery.claim_recovery(tmp_path, first, frozen)
    with pytest.raises(RuntimeError, match="already claimed"):
        comparison_recovery.claim_recovery(tmp_path, tmp_path / "second", frozen)


def test_frozen_checkpoint_recovery_runs_ten_tests_without_retraining(
    protocol, monkeypatch
):
    p = protocol
    # Metadata-only training doubles; actual state compatibility has its own test.
    monkeypatch.setattr(comparison_recovery, "validate_model_state", lambda cp: None)
    path, selected = comparison_fixture(p, monkeypatch)
    failed_run = p.root / "runs/evaluate/before_scoring_failure"

    def fail_before_scoring(config_path):
        cfg = matrix.read_json(config_path)
        write_json(
            failed_run / "run.json",
            dict(
                status="running",
                arguments=dict(
                    checkpoint_path=cfg["checkpoint_path"], config_file=str(config_path)
                ),
            ),
        )
        write_json(
            failed_run / "evaluation_phase.json",
            dict(
                phase="before_scoring",
                checkpoint_sha256=matrix.file_hash(cfg["checkpoint_path"]),
                evaluation_config_sha256=matrix.file_hash(config_path),
            ),
        )
        raise SystemExit(
            "Checkpoint is incompatible: checkpoint dataset does not match the requested dataset"
        )

    with pytest.raises(SystemExit):
        study_config.execute_study(
            path, *study_config.load_study(path), pooled_train(p), fail_before_scoring
        )
    assert len(p.trained) == 10
    original = next((p.root / "runs/studies").glob("*"))
    frozen_path = original / "validation_frozen.json"
    progress_path = original / "progress.json"
    receipt_files = {
        str(f): matrix.file_hash(f) for f in (original / "test_receipts").glob("*.json")
    }
    recovery = write_json(
        p.root / "recovery.json",
        dict(
            protocol="comparison_import_v1",
            path_map=[dict(source=str(p.root), destination=str(p.root))],
            recovery=dict(
                freeze=dict(
                    path=str(frozen_path), sha256=matrix.file_hash(frozen_path)
                ),
                progress=dict(
                    path=str(progress_path), sha256=matrix.file_hash(progress_path)
                ),
                test_receipts=receipt_files,
                failure_files={
                    str(f): matrix.file_hash(f) for f in failed_run.glob("*.json")
                },
            ),
        ),
    )
    before = matrix.inventory(original)
    values = matrix.read_json(path)
    values["recovery_source"] = str(recovery)
    retry = write_json(p.root / "recovery_profile.json", values)
    calls = []

    def no_training(*args):
        pytest.fail("Recovery must not retrain or search")

    group = study_config.execute_study(
        retry, *study_config.load_study(retry), no_training, pooled_test(p, calls)
    )
    assert group != original and len(calls) == 10 and len(p.trained) == 10
    assert matrix.inventory(original) == before
    assert (
        matrix.read_json(group / "comparison.json")["validation_frozen"]["best_custom"]
        == matrix.read_json(frozen_path)["best_custom"]
    )
    assert (
        study_config.execute_study(
            retry, *study_config.load_study(retry), no_training, pooled_test(p, calls)
        )
        == group
    )
    assert len(calls) == 10
    # Omitted weights in a metrics-only ZIP never silently trigger training.
    first_checkpoint = Path(matrix.read_json(frozen_path)["models"][0]["checkpoint"])
    held = first_checkpoint.with_suffix(".unavailable")
    first_checkpoint.rename(held)
    try:
        study, cfg, _ = study_config.load_study(retry)
        with pytest.raises(
            FileNotFoundError, match="Required imported evidence missing"
        ):
            comparison_recovery.recover_jobs(
                recovery, study, cfg, selected, p.root / "missing_check"
            )
    finally:
        held.rename(first_checkpoint)


def test_recovery_checks_real_state_shapes_and_architecture_before_scoring():
    from src.model import CustomConvLSTM

    model = CustomConvLSTM(2, layers=[(2, (3, 3))])
    payload = dict(
        model_config=model.configuration(),
        model_state_dict=model.state_dict(),
        model_registry_entry=dict(name="custom_top1"),
        experiment_config=dict(
            convlstm_layers=[[2, [3, 3]]], hidden_classifier_width=None
        ),
    )
    comparison_recovery.validate_model_state(payload)
    payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
    with pytest.raises(ValueError, match="incompatible"):
        comparison_recovery.validate_model_state(payload)


@pytest.mark.parametrize("name", focused.BASELINES)
def test_recovery_checks_each_actual_baseline_topology_on_meta(name):
    with torch.device("meta"):
        model = experiments.build_registered_model(name, 11, (3, 64, 64), 8)
    dimensions = dict(height=64, width=64, sequence_length=8, channels=3)
    payload = dict(
        model_config=dict(
            registry_name=name, num_classes=11, input_dimensions=dimensions
        ),
        model_state_dict=model.state_dict(),
        model_registry_entry=dict(name=name),
        experiment_config=dict(height=64, width=64, sequence_length=8),
    )
    comparison_recovery.validate_model_state(payload)
    payload["model_config"]["registry_name"] = "wrong_model"
    with pytest.raises(ValueError, match="metadata mismatch"):
        comparison_recovery.validate_model_state(payload)


@pytest.mark.parametrize(
    "field,value", [("target_fps", 16), ("sampling_version", "legacy")]
)
def test_evaluator_honors_temporal_config_and_rejects_mismatch_before_scoring(
    protocol, monkeypatch, field, value
):
    import src.evaluate as evaluation

    p = protocol
    cfg = ExperimentConfig(
        dataset_name="aad",
        dataset_dir=str(p.data),
        split_manifest=str(p.root / "split.json"),
        runs_dir=str(p.root / "runs"),
        target_fps=8,
        sampling_version="timestamps_v1",
    )
    checkpoint = checkpoint_metadata()
    checkpoint.update(
        dataset_name="aad",
        split_manifest_hash=matrix.file_hash(cfg.split_manifest),
        experiment_config=cfg.to_dict(),
    )
    weights = p.root / "frozen.pth"
    torch.save(checkpoint, weights)
    values = {**cfg.to_dict(), field: value, "checkpoint_path": str(weights)}
    path = write_json(p.root / "evaluate.json", values)
    actual_dataset = evaluation.AHARDataset
    seen = []

    def dataset(*args, **kwargs):
        seen.append(kwargs)
        return actual_dataset(*args, **kwargs)

    monkeypatch.setattr(evaluation, "AHARDataset", dataset)
    monkeypatch.setattr(
        evaluation, "model_from_checkpoint", lambda _: torch.nn.Linear(1, 1)
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_classifier",
        lambda *a, **kw: pytest.fail("Test scoring forbidden"),
    )
    with pytest.raises(SystemExit, match="input dimensions do not match"):
        evaluation.main(["--config", str(path)])
    assert seen == [
        dict(
            target_fps=values["target_fps"], sampling_version=values["sampling_version"]
        )
    ]
    marker = next((p.root / "runs/evaluate").glob("*/evaluation_phase.json"))
    assert matrix.read_json(marker)["phase"] == "before_scoring"
