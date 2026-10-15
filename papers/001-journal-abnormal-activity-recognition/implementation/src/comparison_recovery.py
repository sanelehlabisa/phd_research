"""Explicit frozen-checkpoint recovery; never retrain or erase test attempts."""

from copy import deepcopy
from pathlib import Path
import shutil

from .comparison_import import EvidenceImport, portable
from .study_matrix import atomic_json, file_hash, inventory, read_json, value_hash


def validate_model_state(checkpoint):
    """Validate every frozen architecture/state before opening any test clips."""
    import torch
    from .experiments import build_registered_model
    from .model import CustomConvLSTM

    cfg, description = checkpoint["experiment_config"], checkpoint["model_config"]
    name = checkpoint["model_registry_entry"]["name"]
    with torch.device("meta"):
        if name.startswith("custom_top"):
            model = CustomConvLSTM(
                description["num_classes"],
                layers=[(w, tuple(k)) for w, k in cfg["convlstm_layers"]],
                hidden_classifier_width=cfg["hidden_classifier_width"],
            )
            if description != model.configuration():
                raise ValueError(
                    "Frozen custom topology differs from its training config"
                )
        else:
            if description.get("registry_name") != name or description.get(
                "input_dimensions"
            ) != {
                "height": cfg["height"],
                "width": cfg["width"],
                "sequence_length": cfg["sequence_length"],
                "channels": 3,
            }:
                raise ValueError("Frozen baseline topology/input metadata mismatch")
            model = build_registered_model(
                name,
                description["num_classes"],
                (3, cfg["height"], cfg["width"]),
                cfg["sequence_length"],
            )
    actual, expected = checkpoint.get("model_state_dict", {}), model.state_dict()
    if set(actual) != set(expected) or any(
        not isinstance(actual[k], torch.Tensor)
        or actual[k].shape != v.shape
        or actual[k].dtype != v.dtype
        for k, v in expected.items()
    ):
        raise ValueError(
            "Frozen checkpoint weights are incompatible with the registered architecture"
        )


def claim_recovery(runs_dir, group, freeze):
    """One recovery owner per checkpoint set, even if a profile is renamed."""
    key = value_hash(
        [
            freeze["split_sha256"],
            sorted(m["checkpoint_sha256"] for m in freeze["models"]),
        ]
    )
    path = Path(runs_dir) / "study_receipts" / f"recovery_{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"run_dir": str(group.resolve()), "checkpoint_set": key}
    try:
        with path.open("x", encoding="utf-8") as stream:
            import json

            json.dump(owner, stream)
    except FileExistsError:
        if read_json(path) != owner:
            raise RuntimeError(
                "Frozen checkpoint test recovery already claimed; inspect the existing attempt, never repeat"
            )
    return path


def verify_before_scoring(evidence, receipt, freeze_hash, checkpoints):
    """Only affirmative pre-scoring evidence permits a new test attempt."""
    if receipt.get("status") != "failed" or receipt.get("freeze_sha256") != freeze_hash:
        raise ValueError(
            "Completed, partial or unknown test attempt cannot be recovered"
        )
    partial = receipt.get("partial_runs", [])
    if len(partial) != 1:
        raise ValueError(
            "Ambiguous test failure: require exactly one recorded evaluator"
        )
    folder = portable(partial[0])
    files = evidence.spec["recovery"]["failure_files"]
    lifecycle_path = folder / "run.json"
    run = read_json(evidence.checked(lifecycle_path, files))
    original_checkpoint = run.get("arguments", {}).get("checkpoint_path")
    if run.get("status") != "failed" or original_checkpoint not in checkpoints:
        raise ValueError("Failed evaluator does not belong to the frozen checkpoint")
    phase_path = folder / "evaluation_phase.json"
    if evidence.relocate(phase_path).is_file():
        phase = read_json(evidence.checked(phase_path, files))
        cfg_path = run["arguments"]["config_file"]
        evidence.verify(cfg_path, phase["evaluation_config_sha256"])
        if (
            phase["phase"] != "before_scoring"
            or phase["checkpoint_sha256"] != checkpoints[original_checkpoint]
        ):
            raise ValueError("Test scoring may already have started; repeat forbidden")
    else:
        # The historical 14-Oct failure has no phase marker. Its exact known
        # compatibility exception is raised before the first test iteration.
        expected = "Checkpoint is incompatible: checkpoint dataset does not match the requested dataset"
        if run.get("error") != expected or receipt.get("error") != expected:
            raise ValueError("Legacy failure has no affirmative pre-scoring evidence")
    for output in evidence.relocate(folder).rglob("*"):
        if output.is_file() and (
            output.name in {"final.json", "test_predictions.json"}
            or "confusion" in output.name
            or output.suffix == ".mp4"
        ):
            raise ValueError("Partial/completed test outputs exist; repeat forbidden")


def recover_jobs(path, study, config, selected, group):
    """Verify originals, then write separate pooled views for the existing runner."""
    import torch
    from .metrics import validate_selected_checkpoint
    from .study_reporting import POOLED_REPORTING, pooled_metrics, confusion_counts
    from .wide_protocol import class_counts

    evidence = EvidenceImport(path)
    spec = evidence.spec["recovery"]
    freeze_ref, progress_ref = spec["freeze"], spec["progress"]
    freeze = read_json(evidence.verify(freeze_ref["path"], freeze_ref["sha256"]))
    progress = read_json(evidence.verify(progress_ref["path"], progress_ref["sha256"]))
    selection_hash = selected.get("verified_import", {}).get(
        "selection_sha256"
    ) or file_hash(study["selected_config"])
    if (
        freeze.get("selection_partition") != "validation"
        or freeze.get("seed") != 42
        or freeze.get("split_sha256") != selected["split"]["manifest_hash"]
        or freeze.get("search_selection_sha256") != selection_hash
        or [m["model"] for m in freeze["models"]] != study["models"]
        or len(progress["jobs"]) != 10
        or progress.get("tests")
    ):
        raise ValueError(
            "Recovery requires the original ten-model validation freeze, before test scores"
        )
    original_group = portable(freeze_ref["path"]).parent
    receipt_files = spec["test_receipts"]
    actual_receipts = {
        p.resolve()
        for p in (evidence.relocate(original_group) / "test_receipts").glob("*.json")
    }
    if actual_receipts != {evidence.relocate(p) for p in receipt_files}:
        raise ValueError("Recovery must declare every original test receipt")
    checkpoints = {m["checkpoint"]: m["checkpoint_sha256"] for m in freeze["models"]}
    for source, digest in receipt_files.items():
        verify_before_scoring(
            evidence,
            read_json(evidence.verify(source, digest)),
            freeze_ref["sha256"],
            checkpoints,
        )
    if progress.get("test_access") not in {"locked", "started_after_validation_freeze"}:
        raise ValueError("Test access is already completed or ambiguous")
    if progress.get("test_access") != "locked" and not receipt_files:
        raise ValueError("Attempted test has missing receipts")
    split = read_json(config.split_manifest)
    recovered = []
    batches = set()
    for number, (job, frozen) in enumerate(zip(progress["jobs"], freeze["models"]), 1):
        receipt = evidence.json(job["receipt"])
        result = job["result"]
        if (
            receipt["status"] != "complete"
            or (result["partition"], result["test_access"], result["seed"])
            != ("validation", "locked", 42)
            or result["split"]["manifest_hash"] != freeze["split_sha256"]
            or receipt["result"] != result
            or result["name"] != frozen["model"]
            or value_hash(result["experiment_config"]) != frozen["config_sha256"]
            or result["selected_checkpoint"] != frozen["checkpoint"]
        ):
            raise ValueError("Frozen training receipt/config mismatch")
        evidence.inventory(receipt["files"])
        checkpoint = evidence.verify(frozen["checkpoint"], frozen["checkpoint_sha256"])
        # Load originals read-only; never convert historical model weights.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        selection = validate_selected_checkpoint(
            payload, config.dataset_name, freeze["split_sha256"]
        )
        cfg = result["experiment_config"]
        if (
            payload.get("experiment_config") != cfg
            or payload.get("model_registry_entry", {}).get("name") != result["name"]
            or payload["model_config"]["num_classes"] != len(split["class_names"])
            or selection["selected_epoch"]
            != result["checkpoint_selection"]["selected_epoch"]
            or selection["seed"] != config.seed
        ):
            raise ValueError("Frozen checkpoint model/config/epoch/seed mismatch")
        validate_model_state(payload)
        del payload
        expected = config.to_dict()
        machine_fields = {
            "dataset_dir",
            "split_manifest",
            "runs_dir",
            "batch_size",
            "num_workers",
            "pin_memory",
            "convlstm_layers",
            "hidden_classifier_width",
            "prediction_samples_per_category",
            "prediction_samples",
        }
        if any(
            cfg.get(k, expected[k]) != expected[k]
            for k in expected
            if k not in machine_fields
        ):
            raise ValueError("Recovery cannot change the frozen training/input recipe")
        if cfg.get("sampling_version", "legacy") != config.sampling_version:
            raise ValueError("Recovery cannot change the original sampling protocol")
        if (
            not study["minimum_epochs"]
            <= result["early_stopping"]["actual_epochs"]
            <= config.epochs
        ):
            raise ValueError("Frozen model lacks the required longer training")
        if (
            not 1
            <= selection["selected_epoch"]
            <= result["early_stopping"]["actual_epochs"]
        ):
            raise ValueError("Frozen selected epoch exceeds completed training")
        if number <= 5:
            candidate = study["custom_candidates"][number - 1]
            if any(
                cfg[k] != candidate[k]
                for k in ("convlstm_layers", "hidden_classifier_width")
            ):
                raise ValueError(
                    "Frozen custom differs from selected search architecture"
                )
        batches.add(cfg["batch_size"])
        records = evidence.result_records(result, split)
        leaf = group / "recovered" / result["name"]
        for record in records:
            record["source"] = str(evidence.relocate(record["source"]))
        predictions = atomic_json(leaf / "validation_predictions.json", records)
        local = deepcopy(result)
        local.pop("extended_metric_protocol", None)
        local.update(
            reporting_protocol=POOLED_REPORTING,
            validation_metrics=pooled_metrics(records, split["class_names"]),
            validation_per_class=class_counts(records, split["class_names"]),
            validation_confusion_matrix=confusion_counts(records, split["class_names"]),
            validation_predictions=str(predictions),
            selected_checkpoint=str(checkpoint),
            experiment_config={
                **cfg,
                "dataset_dir": config.dataset_dir,
                "split_manifest": config.split_manifest,
                "runs_dir": config.runs_dir,
                "num_workers": config.num_workers,
                "pin_memory": config.pin_memory,
            },
            split={**result["split"], "manifest_path": config.split_manifest},
        )
        import math

        train = result["training_evaluation"]
        train_records = read_json(
            evidence.checked(train["predictions"], receipt["files"])
        )
        expected_train = {
            str(portable(cfg["dataset_dir"]) / r["path"]): r["class_index"]
            for r in split["samples"]
            if r["split"] == "train"
        }
        train_paths = [str(portable(r["source"])) for r in train_records]
        if (
            len(train_paths) != len(expected_train)
            or set(train_paths) != set(expected_train)
            or any(
                r["partition"] != "train" or r["target"] != expected_train[p]
                for r, p in zip(train_records, train_paths)
            )
            or train["checkpoint_sha256"] != frozen["checkpoint_sha256"]
            or train["selected_epoch"] != selection["selected_epoch"]
            or train["model_mode"] != "eval"
            or train["augmentation"] is not False
        ):
            raise ValueError("Frozen matched-training predictions/checkpoint mismatch")
        train_metrics = pooled_metrics(train_records, split["class_names"])
        if any(
            not math.isclose(train["metrics"][k], train_metrics[k], abs_tol=1e-5)
            for k in ("loss", "accuracy")
        ):
            raise ValueError("Frozen training metrics disagree with predictions")
        for record in train_records:
            record["source"] = str(evidence.relocate(record["source"]))
        local["training_evaluation"] = {
            **train,
            "protocol": "matched_checkpoint_eval_v2",
            "reporting_protocol": POOLED_REPORTING,
            "checkpoint": str(checkpoint),
            "metrics": train_metrics,
            "predictions": str(
                atomic_json(leaf / "training_predictions.json", train_records)
            ),
            "per_class": class_counts(train_records, split["class_names"]),
            "confusion_matrix": confusion_counts(train_records, split["class_names"]),
        }
        for key in ("history", "training_curves"):
            source = result[key]
            local_file = evidence.verify(source, receipt["files"][source])
            destination = leaf / local_file.name
            shutil.copyfile(local_file, destination)
            local[key] = str(destination)
        # Derived table view is explicit and separate; original metrics stay intact.
        atomic_json(
            leaf / "import_provenance.json",
            {
                "original_result": result,
                "original_freeze_sha256": freeze_ref["sha256"],
                "import_manifest_sha256": file_hash(path),
                "original_checkpoint_sha256": frozen["checkpoint_sha256"],
                "note": "Pooled view recomputed from original validation predictions; no retraining or checkpoint rewrite.",
            },
        )
        atomic_json(leaf / "summary.json", {"all": [local]})
        receipt_path = group / "jobs" / f"{number:02d}" / "receipt.json"
        atomic_json(
            receipt_path,
            dict(
                status="complete",
                result=local,
                run_dir=str(leaf),
                files=inventory(leaf),
                recovered=True,
                experiment_number=number,
            ),
        )
        recovered.append(
            dict(
                result=local,
                run_dir=str(leaf),
                receipt=str(receipt_path),
                experiment_number=number,
            )
        )
    if len(batches) != 1:
        raise ValueError("Frozen models did not share one common batch")
    atomic_json(
        group / "recovery_provenance.json",
        {
            "freeze": freeze,
            "import_manifest": evidence.spec,
            "freeze_sha256": freeze_ref["sha256"],
        },
    )
    return recovered, freeze
