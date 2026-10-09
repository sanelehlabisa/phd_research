"""Validation-only, one-factor transfer of search settings to native inputs."""

from pathlib import Path
import statistics

from .experiment_config import ExperimentConfig
from .study_matrix import (
    atomic_json,
    file_hash,
    inventory,
    read_json,
    validation_accuracy,
    value_hash,
    verify_inventory,
)


def search_proposal(selected):
    """Read verified seed-42 WD evidence, equally weighting the three sizes."""
    from .capacity_search import job_key

    group = Path(selected["source_group"])
    state = read_json(group / "progress.json")
    decisions = read_json(group / "decisions.json")
    model = selected["top3"][0]["candidate"]["name"]
    jobs = {j["job_id"]: j for j in state["jobs"]}
    options = {}
    for stage in state["stages"]:
        if stage["name"] not in {"flat", "refinement", "weight_decay"}:
            continue
        for row in stage["rows"]:
            if row["model"] == model:
                cfg = row["config"]
                options.setdefault(cfg["weight_decay"], {})[cfg["height"]] = jobs[
                    job_key(row)
                ]["result"]
    if set(options) != {0, 1e-4, 1e-3} or any(
        set(block) != {32, 48, 64} for block in options.values()
    ):
        raise ValueError(
            "Recipe transfer requires complete top-one weight-decay evidence"
        )
    wd = min(
        options,
        key=lambda value: (
            -statistics.mean(
                validation_accuracy(r, True) for r in options[value].values()
            ),
            statistics.mean(
                r["validation_metrics"]["loss"] for r in options[value].values()
            ),
            value,
        ),
    )
    return {"learning_rate": decisions["learning_rates"]["custom"], "weight_decay": wd}


def select_recipe(study, base, selected, group, split, run_trial, offset):
    """At most three fresh jobs; ties retain the incumbent, never inspect test.

    Called within the comparison runner's existing study lock/identity guard.
    Replay reconstructs every decision and verifies receipts, not just the winner.
    """
    from .capacity_search import _validate_result

    if selected.get("protocol") != "capacity_top3_v2":
        raise ValueError("Recipe transfer requires verified focused-search evidence")
    candidate = {**selected["top3"][0]["candidate"], "name": "custom_top1"}
    proposal = search_proposal(selected)
    source = read_json(split["manifest_path"])
    validation = {
        str((Path(base.dataset_dir) / r["path"]).resolve()): r["class_index"]
        for r in source["samples"]
        if r["split"] == "validation"
    }
    common = {
        **base.to_dict(),
        "epochs": 128,
        "early_stopping_patience": 24,
        "prediction_samples_per_category": 0,
        "convlstm_layers": candidate["convlstm_layers"],
        "hidden_classifier_width": candidate["hidden_classifier_width"],
    }
    jobs, by_config, steps = [], {}, []

    def trial(role, lr, wd):
        values = ExperimentConfig.from_mapping(
            {
                **common,
                "learning_rate": lr,
                "weight_decay": wd,
            }
        ).to_dict()
        key = value_hash(values)
        if key in by_config:
            return by_config[key]
        number = offset + len(jobs) + 1
        directory = group / "recipe_jobs" / f"{len(jobs) + 1:02d}_{role}"
        receipt_path = directory / "receipt.json"
        row = {"model": "custom_top1", "config": values, "minimum_epochs": 64}
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get("status") != "complete":
                raise RuntimeError(
                    "Recipe validation was interrupted; inspect evidence, no automatic retry"
                )
            verify_inventory(receipt["files"])
            result = receipt["result"]
            if receipt.get("experiment_number") != number:
                raise ValueError("Recipe experiment number changed")
            print(f"Experiment {number}: verified recipe {role} (reused)", flush=True)
        else:
            if (group / "recipe_selection.json").exists() or (
                group / "validation_frozen.json"
            ).exists():
                raise ValueError(
                    "Frozen recipe evidence is missing; no retraining after selection"
                )
            config_path = atomic_json(directory / "config.json", values)
            candidates = atomic_json(
                directory / "candidates.json",
                {
                    "screening_id": "comparison_recipe_v1",
                    "candidates": [candidate],
                },
            )
            args = [
                "--config",
                str(config_path),
                "--model",
                "custom_top1",
                "--candidates-config",
                str(candidates),
                "--minimum-epochs",
                "64",
                "--run-label",
                "recipe-validation",
                "--trial-name",
                f"{number:03d}",
                "--changed-factor",
                role,
            ]
            atomic_json(directory / "command.json", args)
            atomic_json(
                receipt_path, {"status": "running", "experiment_number": number}
            )
            print(
                f"Experiment {number}: recipe {role}, LR={lr:g}, WD={wd:g}; test locked",
                flush=True,
            )
            try:
                leaf = Path(run_trial(args)).resolve()
                results = read_json(leaf / "summary.json")["all"]
                if (
                    len(results) != 1
                    or read_json(leaf / "run.json")["status"] != "complete"
                ):
                    raise ValueError("Recipe validation leaf is incomplete")
                result = results[0]
                _validate_result(
                    result, row, split, validation, len(source["class_names"])
                )
                files = inventory(leaf)
                files.update(
                    {
                        str(p.resolve()): file_hash(p)
                        for p in directory.glob("*.json")
                        if p != receipt_path
                    }
                )
                receipt = {
                    "status": "complete",
                    "experiment_number": number,
                    "run_dir": str(leaf),
                    "result": result,
                    "files": files,
                }
                atomic_json(receipt_path, receipt)
            except BaseException as error:
                atomic_json(
                    receipt_path,
                    {
                        "status": "failed",
                        "experiment_number": number,
                        "error": str(error),
                    },
                )
                raise
        _validate_result(result, row, split, validation, len(source["class_names"]))
        job = {
            "experiment_number": number,
            "role": role,
            "run_dir": receipt["run_dir"],
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": file_hash(receipt_path),
            "result": result,
        }
        jobs.append(job)
        by_config[key] = job
        return job

    def better(incumbent, challenger):
        def score(job):
            r = job["result"]
            return (-validation_accuracy(r, True), r["validation_metrics"]["loss"])

        return challenger if score(challenger) < score(incumbent) else incumbent

    winner = trial("baseline", base.learning_rate, base.weight_decay)
    for factor in ("learning_rate", "weight_decay"):
        cfg = winner["result"]["experiment_config"]
        settings = {k: cfg[k] for k in ("learning_rate", "weight_decay")}
        settings[factor] = proposal[factor]
        challenger = trial(factor, settings["learning_rate"], settings["weight_decay"])
        incumbent = winner
        winner = better(incumbent, challenger)
        steps.append(
            {
                "factor": factor,
                "incumbent": incumbent["experiment_number"],
                "challenger": challenger["experiment_number"],
                "selected": winner["experiment_number"],
            }
        )
    chosen = {
        k: winner["result"]["experiment_config"][k]
        for k in ("learning_rate", "weight_decay")
    }
    report = {
        "protocol": "native_recipe_transfer_v1",
        "status": "complete",
        "selection_partition": "validation",
        "test_access": "locked",
        "search_selection_sha256": file_hash(study["selected_config"]),
        "source_group": selected["source_group"],
        "proposal": proposal,
        "base_config_sha256": value_hash(base.to_dict()),
        "steps": steps,
        "selected": chosen,
        "selected_experiment": winner["experiment_number"],
        "jobs": jobs,
        "note": "Shared recipe tuned on search top-one custom model, not per-family optimal tuning. Native 50-frame/50x50 inputs, seed 42; ties keep incumbent. Temporal ablations not transferred.",
    }
    path = group / "recipe_selection.json"
    if path.exists() and read_json(path) != report:
        raise ValueError("Frozen comparison recipe changed; test forbidden")
    atomic_json(path, report)
    print(
        f"Frozen shared comparison recipe: {chosen}; {len(jobs)} validation trainings",
        flush=True,
    )
    return ExperimentConfig.from_mapping({**base.to_dict(), **chosen}), report
