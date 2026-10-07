"""Colab orchestration for AAD final confirmation and one-time evaluation.

Calls the existing experiment/evaluation runners; never implements another
training loop. Only completed staged-study evidence can unlock confirmation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src import evaluate, experiments
from src.experiment_config import ExperimentConfig
from src.metrics import metric_protocol, validate_selected_checkpoint
from src.study_config import aggregate_screen, load_study, study_rows
from src.utils import plot_training_curves, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _complete(run_dir):
    if read_json(Path(run_dir) / "run.json").get("status") != "complete":
        raise ValueError(f"Run is incomplete: {run_dir}")


def _checked_result(run_dir, expected, name):
    _complete(run_dir)
    entries = read_json(Path(run_dir) / "summary.json")["all"]
    if len(entries) != 1:
        raise ValueError("Expected one model per confirmation/study job")
    result = entries[0]
    if (
        result["name"] != name
        or result["seed"] != expected["seed"]
        or result["experiment_config"] != expected
        or result["partition"] != "validation"
        or result["test_access"] != "locked"
    ):
        raise ValueError("Run does not match its validation-only configuration")
    return result


def selection_plan(study_dir, *, epochs=64):
    """Validate the complete study and choose one tested configuration, not a grid."""
    if not str(study_dir).strip():
        raise ValueError("Set STUDY_RUN_DIR to the completed ticket-052/053 study")
    study_dir = Path(study_dir).resolve()
    _complete(study_dir)
    study, config, manifest = load_study(study_dir / "study.json")
    if study["seeds"] != [42, 2026]:
        raise ValueError("Final confirmation requires the planned seeds 42 and 2026")
    if type(epochs) is not int or not 1 <= epochs <= 64:
        raise ValueError("Final confirmation budget must be between 1 and 64 epochs")
    metadata = read_json(study_dir / "run.json")
    if metadata.get("study_sha256") != _digest(study):
        raise ValueError("Study configuration hash mismatch")
    if read_json(study_dir / "candidates.json") != manifest.to_dict():
        raise ValueError("Study candidate manifest mismatch")
    summary = read_json(study_dir / "summary.json")
    if (
        summary["selection_partition"] != "validation"
        or summary["test_access"] != "locked"
    ):
        raise ValueError("Study must use validation only")
    jobs = summary["jobs"]
    screening = [j["result"] for j in jobs if j["stage"] == "screen"]
    ranking = aggregate_screen(screening, study["models"], study["seeds"])
    winner = ranking[0]["name"]
    if summary["selected_model"] != winner:
        raise ValueError("Stored winner differs from recomputed validation ranking")
    rows = study_rows(study, config, winner)
    if len(rows) != len(jobs):
        raise ValueError("Complete screen and every planned ablation are required")
    dataset_dir = screening[0]["dataset_dir"]
    split_hash = screening[0]["split"]["manifest_hash"]
    candidates = {item.name: item for item in manifest.candidates}
    groups = {"reference": [item for item in screening if item["name"] == winner]}
    for row, job in zip(rows, jobs):
        expected = {**row["config"], "dataset_dir": dataset_dir}
        if row["model"] in candidates:
            candidate = candidates[row["model"]]
            expected.update(
                convlstm_layers=candidate.to_dict()["convlstm_layers"],
                hidden_classifier_width=candidate.hidden_classifier_width,
            )
        entry = _checked_result(job["run_dir"], expected, row["model"])
        if (
            entry != job["result"]
            or job["stage"] != row["stage"]
            or job["changed_factor"] != row["changed_factor"]
            or entry["split"]["manifest_hash"] != split_hash
            or entry["metric_protocol"] != metric_protocol()
        ):
            raise ValueError("Study job/configuration/split evidence mismatch")
        if row["stage"] == "ablation":
            key = _digest({k: v for k, v in expected.items() if k != "seed"})
            groups.setdefault(key, []).append(entry)
    variants = []
    for key, entries in groups.items():
        aggregate = aggregate_screen(entries, [winner], study["seeds"])[0]
        aggregate["variant"] = key
        variants.append(aggregate)
    # Stable ties retain the reference before a single-factor alternative.
    chosen = min(
        variants,
        key=lambda item: (
            -item["validation_metrics"]["accuracy"],
            item["validation_metrics"]["loss"],
            item["num_params"],
        ),
    )
    selected = chosen["runs"][0]
    final_config = ExperimentConfig.from_mapping(
        {
            **selected["experiment_config"],
            "epochs": epochs,
        }
    ).to_dict()
    return {
        "study_dir": str(study_dir),
        "study_sha256": metadata["study_sha256"],
        "study_summary_sha256": file_hash(study_dir / "summary.json"),
        "selected_model": winner,
        "selected_variant": chosen["variant"],
        "selection_partition": "validation",
        "test_access": "locked",
        "configuration": final_config,
        "seeds": study["seeds"],
        "split_hash": split_hash,
        "num_params": chosen["num_params"],
        "split_file_sha256": file_hash(final_config["split_manifest"]),
        "source_validation_mean": chosen["validation_metrics"],
        "source_validation_std": chosen["validation_std"],
        "configuration_rule": "Mean validation accuracy, loss, parameters; single-factor only",
        "checkpoint_rule": "Lowest confirmation validation loss, then accuracy, then seed",
    }


def _check_checkpoint(result):
    path = Path(result["selected_checkpoint"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_selected_checkpoint(
        checkpoint, Path(result["dataset_dir"]).name, result["split"]["manifest_hash"]
    )
    if (
        checkpoint["experiment_config"] != result["experiment_config"]
        or checkpoint["seed"] != result["seed"]
        or checkpoint["selected_epoch"]
        != result["checkpoint_selection"]["selected_epoch"]
        or checkpoint["trainable_parameters"] != result["num_params"]
    ):
        raise ValueError("Checkpoint/configuration/seed provenance mismatch")
    return file_hash(path)


def _verify_frozen(final_dir):
    final_dir = Path(final_dir)
    frozen = read_json(final_dir / "frozen.json")
    if frozen["plan_sha256"] != file_hash(final_dir / "plan.json"):
        raise ValueError("Frozen plan changed")
    plan = read_json(final_dir / "plan.json")
    if (
        frozen["selection_partition"] != "validation"
        or frozen["seeds"] != plan["seeds"]
    ):
        raise ValueError("Frozen selection requires every confirmation seed")
    results = []
    for seed in plan["seeds"]:
        record = read_json(final_dir / f"seed-{seed}.json")
        if record["status"] != "complete":
            raise ValueError("Confirmation seed is incomplete")
        result = record["result"]
        expected = {**plan["configuration"], "seed": seed}
        if (
            _checked_result(record["run_dir"], expected, plan["selected_model"])
            != result
        ):
            raise ValueError("Frozen result/configuration mismatch")
        results.append(result)
        if file_hash(result["selected_checkpoint"]) != record["checkpoint_sha256"]:
            raise ValueError("Confirmation checkpoint changed")
        if (
            file_hash(Path(record["run_dir"]) / "summary.json")
            != record["summary_sha256"]
        ):
            raise ValueError("Confirmation results changed")
        if file_hash(result["history"]) != record["history_sha256"]:
            raise ValueError("Confirmation history changed")
        if (
            file_hash(final_dir / f"seed-{seed}.json")
            != frozen["seed_record_hashes"][str(seed)]
        ):
            raise ValueError("Frozen confirmation record changed")
    if file_hash(frozen["checkpoint"]) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    selected = min(
        results,
        key=lambda r: (
            r["validation_metrics"]["loss"],
            -r["validation_metrics"]["accuracy"],
            r["seed"],
        ),
    )
    if (
        frozen["checkpoint"] != selected["selected_checkpoint"]
        or frozen["configuration"] != selected["experiment_config"]
        or frozen["selected_seed"] != selected["seed"]
        or frozen["selected_model"] != plan["selected_model"]
    ):
        raise ValueError(
            "Frozen checkpoint is not the validation-selected confirmation"
        )
    if file_hash(plan["configuration"]["split_manifest"]) != plan["split_file_sha256"]:
        raise ValueError("Frozen split changed")
    if (
        file_hash(Path(plan["study_dir"]) / "summary.json")
        != plan["study_summary_sha256"]
    ):
        raise ValueError("Source study evidence changed")
    return frozen


def train_and_freeze(study_dir, *, epochs=64):
    """Train both seeds, persist progress, freeze by validation; test stays locked."""
    plan = selection_plan(study_dir, epochs=epochs)
    final_dir = Path(plan["study_dir"]) / "final"
    plan_path = final_dir / "plan.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError(
                "Final plan already exists and differs; do not overwrite evidence"
            )
    else:
        write_json(plan_path, plan)
    if (final_dir / "frozen.json").exists():
        _verify_frozen(final_dir)
        print(f"Reusing frozen confirmation: {final_dir}", flush=True)
        return final_dir
    if (final_dir / "test-attempt.json").exists():
        raise ValueError(
            "Test already attempted; no retraining or reselection permitted"
        )
    results = []
    for seed in plan["seeds"]:
        state_path = final_dir / f"seed-{seed}.json"
        expected = {**plan["configuration"], "seed": seed}
        if state_path.exists():
            state = read_json(state_path)
            if state["status"] != "complete":
                raise ValueError(
                    f"Seed {seed} has an interrupted attempt; inspect its evidence before resuming"
                )
            result = _checked_result(state["run_dir"], expected, plan["selected_model"])
            if (
                result != state["result"]
                or _check_checkpoint(result) != state["checkpoint_sha256"]
                or file_hash(result["history"]) != state["history_sha256"]
                or file_hash(Path(state["run_dir"]) / "summary.json")
                != state["summary_sha256"]
            ):
                raise ValueError("Completed seed evidence changed")
        else:
            config_path = write_json(final_dir / f"config-{seed}.json", expected)
            arguments = [
                "--config",
                str(config_path),
                "--model",
                plan["selected_model"],
                "--run-label",
                "final-confirmation",
                "--trial-name",
                f"final_seed_{seed}",
                "--changed-factor",
                "confirmation",
            ]
            study = read_json(Path(plan["study_dir"]) / "study.json")
            if plan["selected_model"] in {
                c["name"] for c in study["custom_candidates"]
            }:
                arguments += [
                    "--candidates-config",
                    str(Path(plan["study_dir"]) / "candidates.json"),
                ]
            write_json(
                state_path,
                {"status": "running", "arguments": arguments, "test_access": "locked"},
            )
            print(
                f"Final confirmation: {plan['selected_model']}, seed {seed}, epochs <= {epochs}",
                flush=True,
            )
            try:
                run_dir = experiments.main(arguments)
                result = _checked_result(run_dir, expected, plan["selected_model"])
                if (
                    result["split"]["manifest_hash"] != plan["split_hash"]
                    or result["num_params"] != plan["num_params"]
                ):
                    raise ValueError(
                        "Confirmation differs from selected study model/split"
                    )
                checkpoint_hash = _check_checkpoint(result)
                write_json(
                    state_path,
                    {
                        "status": "complete",
                        "run_dir": str(run_dir),
                        "result": result,
                        "checkpoint_sha256": checkpoint_hash,
                        "summary_sha256": file_hash(Path(run_dir) / "summary.json"),
                        "history_sha256": file_hash(result["history"]),
                        "test_access": "locked",
                    },
                )
            except BaseException as error:
                write_json(
                    state_path,
                    {"status": "failed", "error": str(error), "test_access": "locked"},
                )
                raise
        results.append(result)
    aggregate = aggregate_screen(results, [plan["selected_model"]], plan["seeds"])[0]
    selected = min(
        results,
        key=lambda r: (
            r["validation_metrics"]["loss"],
            -r["validation_metrics"]["accuracy"],
            r["seed"],
        ),
    )
    write_json(
        final_dir / "frozen.json",
        {
            "plan_sha256": file_hash(plan_path),
            "selection_partition": "validation",
            "test_access": "locked",
            "selected_model": plan["selected_model"],
            "seeds": plan["seeds"],
            "selected_seed": selected["seed"],
            "configuration": selected["experiment_config"],
            "checkpoint": selected["selected_checkpoint"],
            "checkpoint_sha256": _check_checkpoint(selected),
            "validation_mean": aggregate["validation_metrics"],
            "validation_std": aggregate["validation_std"],
            "seed_record_hashes": {
                str(seed): file_hash(final_dir / f"seed-{seed}.json")
                for seed in plan["seeds"]
            },
        },
    )
    _verify_frozen(final_dir)
    print(f"Frozen checkpoint and both seed results: {final_dir}", flush=True)
    return final_dir


def evaluate_frozen(final_dir):
    """Evaluate once; later calls verify and display saved evidence without inference."""
    final_dir = Path(final_dir)
    frozen = _verify_frozen(final_dir)
    freeze_hash = file_hash(final_dir / "frozen.json")
    attempt_path = final_dir / "test-attempt.json"
    if attempt_path.exists():
        attempt = read_json(attempt_path)
        if attempt["frozen_sha256"] != freeze_hash:
            raise ValueError(
                "Selection changed after test access; evaluation is locked"
            )
        if attempt["status"] != "complete":
            raise ValueError(
                "Test was already attempted; inspect partial evidence, do not rerun blindly"
            )
        _complete(attempt["run_dir"])
        if file_hash(attempt["report"]) != attempt["report_sha256"]:
            raise ValueError("Saved test report changed")
        for path, expected_hash in attempt["artifact_hashes"].items():
            if file_hash(path) != expected_hash:
                raise ValueError("Saved test artifact changed")
        print(
            "Reusing the saved final test report; no test inference repeated.",
            flush=True,
        )
        return Path(attempt["report"])
    config_path = write_json(
        final_dir / "evaluation.json",
        {
            **frozen["configuration"],
            "checkpoint_path": frozen["checkpoint"],
        },
    )
    state = {"status": "running", "frozen_sha256": freeze_hash}
    write_json(attempt_path, state)  # Durable guard BEFORE any test access.
    try:
        run_dir = evaluate.main(["--config", str(config_path)])
        report = Path(run_dir) / "metrics" / "final.json"
        _complete(run_dir)
        write_json(
            attempt_path,
            {
                **state,
                "status": "complete",
                "run_dir": str(run_dir),
                "report": str(report),
                "report_sha256": file_hash(report),
                "artifact_hashes": {
                    str(path): file_hash(path)
                    for path in Path(run_dir).rglob("*")
                    if path.is_file()
                },
            },
        )
    except BaseException as error:
        write_json(attempt_path, {**state, "status": "failed", "error": str(error)})
        raise
    return report


def show_results(final_dir, report_path=None):
    """Save curves and print artifact paths; show pictures/videos only in a kernel."""
    final_dir = Path(final_dir)
    frozen = _verify_frozen(final_dir)
    print("Confirmation mean:", frozen["validation_mean"])
    print("Confirmation sample SD:", frozen["validation_std"])
    try:
        from IPython import get_ipython
        from IPython.display import Image, Video, display

        interactive = get_ipython() is not None
    except ImportError:
        interactive = False
    for seed in frozen["seeds"]:
        result = read_json(final_dir / f"seed-{seed}.json")["result"]
        history = read_json(result["history"])["epochs"]
        curve = plot_training_curves(
            [e["training_metrics"]["loss"] for e in history],
            [e["validation_metrics"]["loss"] for e in history],
            [e["training_metrics"]["accuracy"] for e in history],
            [e["validation_metrics"]["accuracy"] for e in history],
            final_dir / f"curves-{seed}.png",
        )
        if interactive:
            display(Image(filename=curve))
    if report_path is None:
        return
    report = read_json(report_path)
    print("Final test metrics:", report["metrics"])
    print("Per-class metrics:", report["per_class"])
    print("Full report:", report_path)
    for record in report["prediction_examples"]["records"]:
        print(record)
        if interactive:
            display(Video(filename=record["path"], embed=True, width=420))
