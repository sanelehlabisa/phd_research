"""Single-source AAD study schema and staged orchestration for src.experiments.

Old multi-file plans remain readable for historical notebooks, not this study.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

from .experiment_config import CandidateManifest, ExperimentConfig
from .metrics import rank_validation_results
from .utils import RunContext, write_json

BASELINES = ("r3d_18", "mc3_18", "r2plus1d_18", "swin3d_t")
ROOT = Path(__file__).resolve().parent.parent


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} requires exactly: {', '.join(sorted(expected))}")


def _path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("study paths must be non-empty strings")
    path = Path(value).expanduser()
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def load_study(path):
    """Validate every dimension and the additive run budget before dataset access."""
    study = json.loads(Path(path).read_text(encoding="utf-8"))
    _fields(
        study,
        {
            "schema_version",
            "dataset",
            "models",
            "custom_candidates",
            "reference_input",
            "factors",
            "training",
            "seeds",
            "runs_dir",
            "max_runs",
            "published_topology",
        },
        "study",
    )
    if type(study["schema_version"]) is not int or study["schema_version"] != 1:
        raise ValueError("unsupported study schema_version")
    data = study["dataset"]
    _fields(data, {"name", "path", "split_manifest"}, "dataset")
    if data["name"] != "aad":
        raise ValueError(
            "this controlled study supports AAD only; keep Kinetics separate"
        )
    manifest = CandidateManifest.from_mapping(
        {
            "screening_id": "aad_staged_architectures",
            "candidates": study["custom_candidates"],
        }
    )
    names = [candidate.name for candidate in manifest.candidates]
    if len(names) > 3 or set(names).intersection(
        (*BASELINES, "paper_convlstm_published")
    ):
        raise ValueError(
            "declare at most three custom candidates with distinct registry names"
        )
    models = study["models"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(name, str) for name in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("models must be a non-empty unique list")
    if set(models) - set([*names, *BASELINES]):
        raise ValueError(
            "unknown model; PaperConvLSTM belongs only in published_topology"
        )
    if not set(names) <= set(models):
        raise ValueError("every declared custom candidate must be included in models")
    seeds = study["seeds"]
    if (
        not isinstance(seeds, list)
        or not 2 <= len(seeds) <= 3
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must contain two or three unique non-negative integers")
    training = study["training"]
    _fields(
        training,
        {
            "epochs",
            "early_stopping_patience",
            "batch_size",
            "learning_rate",
            "scheduler",
            "augment",
            "num_workers",
            "pin_memory",
            "prediction_samples_per_category",
        },
        "fixed training budget",
    )
    reference = study["reference_input"]
    _fields(
        reference, {"frame_size", "sequence_length", "weight_decay"}, "reference_input"
    )
    config = ExperimentConfig.from_mapping(
        {
            **training,
            "dataset_name": "aad",
            "dataset_dir": _path(data["path"]),
            "split_manifest": _path(data["split_manifest"]),
            "runs_dir": _path(study["runs_dir"]),
            "seed": seeds[0],
            "height": reference["frame_size"],
            "width": reference["frame_size"],
            "sequence_length": reference["sequence_length"],
            "weight_decay": reference["weight_decay"],
        }
    )
    factors = study["factors"]
    _fields(
        factors,
        {"frame_sizes", "sequence_lengths", "weight_decays"},
        "one-factor stages",
    )
    for key, base in (
        ("frame_sizes", config.height),
        ("sequence_lengths", config.sequence_length),
        ("weight_decays", config.weight_decay),
    ):
        values = factors[key]
        if not isinstance(values, list) or not 1 <= len(values) <= 3:
            raise ValueError(
                f"{key} must declare one to three values; no Cartesian grid"
            )
        for value in values:
            if key == "weight_decays":
                valid = (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                )
            else:
                valid = type(value) is int and value > 0
                if key == "frame_sizes":
                    valid = valid and value >= 4 and value % 2 == 0
            if not valid:
                raise ValueError(f"invalid {key} value: {value!r}")
        if len(set(values)) != len(values) or base not in values:
            raise ValueError(f"{key} must be unique and include the reference")
    native = study["published_topology"]
    _fields(native, {"enabled", "batch_size"}, "published_topology")
    if (
        type(native["enabled"]) is not bool
        or type(native["batch_size"]) is not int
        or native["batch_size"] != 1
    ):
        raise ValueError("published_topology requires enabled boolean and batch_size=1")
    count = len(study_rows(study, config))
    if (
        type(study["max_runs"]) is not int
        or not 1 <= study["max_runs"] <= 64
        or count > study["max_runs"]
    ):
        raise ValueError(
            f"declared stages require {count} runs, exceeding max_runs (hard limit 64)"
        )
    return study, config, manifest


def study_rows(study, config, winner="VALIDATION_WINNER"):
    """Expand sums of factors, never their Cartesian product."""
    rows = []

    def row(stage, model, seed, factor, overrides):
        values = {**config.to_dict(), **overrides, "seed": seed}
        resolved = ExperimentConfig.from_mapping(values)
        return {
            "stage": stage,
            "model": model,
            "seed": seed,
            "changed_factor": factor,
            "config": resolved.to_dict(),
            "partition": "validation",
            "test_access": "locked",
        }

    for name in study["models"]:
        for seed in study["seeds"]:
            rows.append(row("screen", name, seed, "architecture", {}))
    for field, values, base in (
        ("frame_size", study["factors"]["frame_sizes"], config.height),
        (
            "sequence_length",
            study["factors"]["sequence_lengths"],
            config.sequence_length,
        ),
        ("weight_decay", study["factors"]["weight_decays"], config.weight_decay),
    ):
        for value in values:
            if value == base:
                continue
            overrides = (
                {"height": value, "width": value}
                if field == "frame_size"
                else {field: value}
            )
            for seed in study["seeds"]:
                rows.append(row("ablation", winner, seed, field, overrides))
    if study["published_topology"]["enabled"]:
        for seed in study["seeds"]:
            rows.append(
                row(
                    "native",
                    "paper_convlstm_published",
                    seed,
                    "native_protocol",
                    {"sequence_length": 50, "height": 50, "width": 50, "batch_size": 1},
                )
            )
    return rows


def print_study(path, study, config):
    """List all fixed and conditional jobs without allocating a model/dataset."""
    rows = study_rows(study, config)
    print(f"AAD staged study: {len(rows)} model runs; no Cartesian expansion.")
    print(f"Dataset={config.dataset_dir}; split={config.split_manifest}; split_seed=42")
    print(
        f"Fixed: epochs<={config.epochs}, patience={config.early_stopping_patience}, "
        f"batch={config.batch_size}, lr={config.learning_rate}, scheduler={config.scheduler}"
    )
    print("Selection: mean validation accuracy, loss, parameters; test remains locked.")
    print(
        "Ablations reuse the winner's screen runs as the unchanged reference; no combined factors."
    )
    print(
        "Command:",
        shlex.join(
            [
                sys.executable,
                "-m",
                "src.experiments",
                "--study-config",
                str(Path(path).resolve()),
            ]
        ),
    )
    for number, row in enumerate(rows, 1):
        c = row["config"]
        candidate_flag = (
            " --candidates-config <study-run>/candidates.json"
            if row["model"] not in (*BASELINES, "paper_convlstm_published")
            else ""
        )
        print(
            f"{number:02d}. {row['stage']}: {row['model']} seed={row['seed']} "
            f"T={c['sequence_length']} {c['height']}x{c['width']} wd={c['weight_decay']} "
            f"factor={row['changed_factor']} | leaf: python -m src.experiments "
            f"--config <study-run>/jobs/{number:02d}/config.json "
            f"--model {row['model']}{candidate_flag}"
        )
    print(
        "Leaf JSON/commands are generated and recorded from this one file at execution; "
        "VALIDATION_WINNER is resolved only after every screen seed finishes."
    )
    print(
        "Native PaperConvLSTM: T=50, 50x50, batch=1, separate/unranked; "
        f"enabled={study['published_topology']['enabled']} "
        f"({len(study['seeds'])} additional runs if disabled and later enabled)."
    )


def aggregate_screen(records, models, seeds):
    """Rank complete, comparable validation evidence across all planned seeds."""
    expected = {(name, seed) for name in models for seed in seeds}
    actual = {(item["name"], item["seed"]) for item in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("screen must complete every model/seed exactly once")
    splits = {item["split"]["manifest_hash"] for item in records}
    datasets = {item["dataset_dir"] for item in records}
    protocols = set()
    for item in records:
        if item["partition"] != "validation" or item["test_access"] != "locked":
            raise ValueError("screen selection requires validation-only evidence")
        cfg = {
            key: value
            for key, value in item["experiment_config"].items()
            if key
            not in {"seed", "runs_dir", "convlstm_layers", "hidden_classifier_width"}
        }
        protocols.add(json.dumps(cfg, sort_keys=True))
    if len(splits) != 1 or len(datasets) != 1 or len(protocols) != 1:
        raise ValueError(
            "screen results must share dataset, split and fixed training/input protocol"
        )
    aggregated = []
    for name in models:
        entries = [item for item in records if item["name"] == name]
        if len({item["num_params"] for item in entries}) != 1:
            raise ValueError("parameter count must agree across seeds")
        scores = {
            key: [item["validation_metrics"][key] for item in entries]
            for key in ("loss", "accuracy", "precision", "recall", "f1")
        }
        if any(
            not math.isfinite(value) for values in scores.values() for value in values
        ):
            raise ValueError("screen metrics must be finite")
        aggregated.append(
            {
                "name": name,
                "partition": "validation",
                "num_params": entries[0]["num_params"],
                "validation_metrics": {
                    key: statistics.mean(values) for key, values in scores.items()
                },
                "validation_std": {
                    key: statistics.stdev(values) for key, values in scores.items()
                },
                "seeds": seeds,
                "runs": entries,
            }
        )
    return rank_validation_results(aggregated)


def execute_study(path, study, config, manifest, run_trial):
    """Execute existing comparison leaves sequentially; never invoke test evaluation."""
    from .dataset_source import resolve_dataset

    root = resolve_dataset("aad", config.dataset_dir, ROOT)
    config = ExperimentConfig.from_mapping(
        {**config.to_dict(), "dataset_dir": str(root)}
    )
    canonical = json.dumps(study, sort_keys=True).encode()
    group = RunContext(
        config.runs_dir,
        purpose="studies",
        dataset_path=root,
        label="aad-staged",
        arguments={"study_file": str(Path(path).resolve()), "study": study},
        metadata={
            "study_sha256": hashlib.sha256(canonical).hexdigest(),
            "test_access": "locked",
        },
    )
    write_json(group.run_dir / "study.json", study)
    candidates = write_json(group.run_dir / "candidates.json", manifest.to_dict())
    completed = []

    def run(row):
        number = len(completed) + 1
        job_dir = group.run_dir / "jobs" / f"{number:02d}"
        # Keep leaf artifacts at the usual runs/experiments path, avoiding deep
        # Windows paths. The study records their exact directories, never globs.
        values = {
            **row["config"],
            "dataset_dir": str(root),
            "runs_dir": config.runs_dir,
        }
        candidate = next(
            (item for item in manifest.candidates if item.name == row["model"]), None
        )
        if candidate is not None:
            values.update(
                convlstm_layers=candidate.to_dict()["convlstm_layers"],
                hidden_classifier_width=candidate.hidden_classifier_width,
            )
        leaf = write_json(job_dir / "config.json", values)
        args = [
            "--config",
            str(leaf),
            "--model",
            row["model"],
            "--run-label",
            row["stage"],
            "--trial-name",
            f"{row['stage']}_{number:02d}",
            "--changed-factor",
            row["changed_factor"],
        ]
        if row["model"] in {candidate.name for candidate in manifest.candidates}:
            args += ["--candidates-config", str(candidates)]
        write_json(
            job_dir / "command.json", [sys.executable, "-m", "src.experiments", *args]
        )
        print(
            f"\nStudy job {number}: {row['stage']} / {row['model']} / seed {row['seed']}",
            flush=True,
        )
        leaf_run = run_trial(args)
        payload = json.loads(
            (Path(leaf_run) / "summary.json").read_text(encoding="utf-8")
        )
        if len(payload["all"]) != 1:
            raise ValueError("each study job must return exactly one candidate")
        entry = payload["all"][0]
        if entry["name"] != row["model"] or entry["seed"] != row["seed"]:
            raise ValueError("leaf evidence does not match its declared study job")
        if entry["partition"] != "validation" or entry["test_access"] != "locked":
            raise ValueError("study jobs must keep test data locked")
        if entry["experiment_config"] != values:
            raise ValueError("leaf configuration differs from the declared study job")
        if (
            completed
            and entry["split"]["manifest_hash"]
            != completed[0]["result"]["split"]["manifest_hash"]
        ):
            raise ValueError("study jobs must use the same split")
        completed.append(
            {
                "stage": row["stage"],
                "changed_factor": row["changed_factor"],
                "run_dir": str(leaf_run),
                "result": entry,
            }
        )
        write_json(group.run_dir / "progress.json", completed)
        return entry

    rows = study_rows(study, config)
    try:
        screen = [run(row) for row in rows if row["stage"] == "screen"]
        ranking = aggregate_screen(screen, study["models"], study["seeds"])
        winner = ranking[0]["name"]
        selection = {
            "selected_model": winner,
            "selection_partition": "validation",
            "ranking": ranking,
            "test_access": "locked",
            "ranking_rule": [
                "mean_validation_accuracy",
                "mean_validation_loss",
                "parameters",
            ],
            "reference_runs": [item for item in screen if item["name"] == winner],
        }
        write_json(group.run_dir / "selection.json", selection)
        print(f"Frozen validation-selected reference: {winner}", flush=True)
        for row in study_rows(study, config, winner):
            if row["stage"] != "screen":
                run(row)
        ablation_groups = {}
        for item in completed:
            if item["stage"] != "ablation":
                continue
            cfg = item["result"]["experiment_config"]
            key = (
                item["changed_factor"],
                cfg["height"],
                cfg["width"],
                cfg["sequence_length"],
                cfg["weight_decay"],
            )
            ablation_groups.setdefault(key, []).append(item["result"])
        ablation_comparisons = [
            {
                "factor": key[0],
                "height": key[1],
                "width": key[2],
                "sequence_length": key[3],
                "weight_decay": key[4],
                "summary": aggregate_screen(entries, [winner], study["seeds"])[0],
            }
            for key, entries in ablation_groups.items()
        ]
        summary = {
            **selection,
            "jobs": completed,
            "ablation_comparisons": ablation_comparisons,
            "native_comparable": False,
            "ablation_factors_combined": False,
            "note": "No test evaluation; ablations do not silently replace the frozen reference.",
        }
        summary_path = write_json(group.run_dir / "summary.json", summary)
        group.complete(
            artifacts={
                "summary": str(summary_path),
                "selection": str(group.run_dir / "selection.json"),
            },
            results={
                "selected_model": winner,
                "completed_jobs": len(completed),
                "test_access": "locked",
            },
        )
    except Exception as error:
        group.update(
            {
                "status": "failed",
                "error": str(error),
                "completed_jobs": len(completed),
                "test_access": "locked",
            }
        )
        raise
    return group.run_dir
