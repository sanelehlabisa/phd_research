"""Single-source configured-dataset study schema for src.experiments.

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
from .utils import RunContext, safe_filename, write_json

BASELINES = ("r3d_18", "mc3_18", "r2plus1d_18", "swin3d_t", "swin3d_s")
ROOT = Path(__file__).resolve().parent.parent


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} requires exactly: {', '.join(sorted(expected))}")


def _path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("study paths must be non-empty strings")
    path = Path(value).expanduser()
    return str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())


def _normalise_grid(study):
    """Convert a flat JSON grid into the shared internal study representation."""
    expected = {
        "schema_version",
        "mode",
        "profile",
        "dataset_name",
        "dataset_dir",
        "split_manifest",
        "runs_dir",
        "models",
        "custom_candidates",
        "frame_sizes",
        "num_frames",
        "epochs",
        "weight_decays",
        "seed",
        "batch_size",
        "early_stopping_patience",
        "learning_rate",
        "scheduler",
        "augment",
        "num_workers",
        "pin_memory",
        "prediction_samples_per_category",
    }
    _fields(study, expected, "flat experiment grid")
    if study["schema_version"] != 1 or study["mode"] != "grid":
        raise ValueError("flat experiment grid requires schema_version=1 and mode=grid")
    if not isinstance(study["dataset_name"], str) or not study[
        "dataset_name"
    ].strip():
        raise ValueError("dataset_name must be a non-empty string")
    if not isinstance(study["dataset_dir"], str) or not study[
        "dataset_dir"
    ].strip():
        raise ValueError("dataset_dir must be a non-empty path")
    if study["profile"] not in {"local_smoke", "colab_a100"}:
        raise ValueError("profile must be local_smoke or colab_a100")
    frame_sizes = study["frame_sizes"]
    num_frames = study["num_frames"]
    if (
        not isinstance(frame_sizes, list)
        or not frame_sizes
        or any(
            type(size) is not int or size < 4 or size % 2 for size in frame_sizes
        )
        or len(set(frame_sizes)) != len(frame_sizes)
    ):
        raise ValueError("frame_sizes must be unique, even integers of at least 4")
    if (
        not isinstance(num_frames, list)
        or not num_frames
        or any(type(frames) is not int or frames <= 0 for frames in num_frames)
        or len(set(num_frames)) != len(num_frames)
    ):
        raise ValueError("num_frames must be a non-empty list of unique positive integers")
    data_sizes = [
        [frames, size] for size in frame_sizes for frames in num_frames
    ]
    epochs = study["epochs"]
    if (
        not isinstance(epochs, list)
        or not epochs
        or any(type(value) is not int or value < (1 if study["profile"] == "local_smoke" else 8) for value in epochs)
        or len(set(epochs)) != len(epochs)
    ):
        raise ValueError("epochs must be a non-empty list of unique integers >= 8")
    if study["profile"] == "colab_a100" and min(epochs) < 16:
        raise ValueError("colab_a100 epochs must be at least 16")
    weight_decays = study["weight_decays"]
    if (
        not isinstance(weight_decays, list)
        or not weight_decays
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in weight_decays
        )
        or len(set(weight_decays)) != len(weight_decays)
    ):
        raise ValueError("weight_decays must be a non-empty list of unique non-negative numbers")
    seed = study["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(study["models"], list) or not study["models"]:
        raise ValueError("models must be a non-empty list")

    first_frames, first_size = data_sizes[0]
    training = {
        "epochs": epochs[0],
        "early_stopping_patience": study["early_stopping_patience"],
        "batch_size": study["batch_size"],
        "learning_rate": study["learning_rate"],
        "scheduler": study["scheduler"],
        "augment": study["augment"],
        "num_workers": study["num_workers"],
        "pin_memory": study["pin_memory"],
        "prediction_samples_per_category": study[
            "prediction_samples_per_category"
        ],
    }
    normalised = {
        "schema_version": 1,
        "mode": "grid",
        "profile": study["profile"],
        "dataset": {
            "name": study["dataset_name"],
            "path": study["dataset_dir"],
            "split_manifest": study["split_manifest"],
        },
        "models": study["models"],
        "custom_candidates": study["custom_candidates"],
        "reference_input": {
            "frame_size": first_size,
            "sequence_length": first_frames,
            "weight_decay": weight_decays[0],
        },
        "factors": {
            "frame_sizes": list(dict.fromkeys(size[1] for size in data_sizes)),
            "sequence_lengths": list(dict.fromkeys(size[0] for size in data_sizes)),
            "weight_decays": weight_decays,
        },
        "training": training,
        "seeds": [seed],
        "runs_dir": study["runs_dir"],
        "max_runs": len(study["models"])
        * len(data_sizes)
        * len(epochs)
        * len(weight_decays),
        "published_topology": {"enabled": False, "batch_size": 1},
        "frame_sizes": frame_sizes,
        "num_frames": num_frames,
        "data_sizes": data_sizes,
        "epoch_values": epochs,
    }
    return normalised


def load_study(path):
    """Validate every dimension and the additive run budget before dataset access."""
    study = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(study, dict) and study.get("mode") == "grid":
        study = _normalise_grid(study)
    expected_fields = {
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
    }
    if isinstance(study, dict) and "selected_config" in study:
        expected_fields.add("selected_config")
    if isinstance(study, dict) and "minimum_epochs" in study:
        expected_fields.add("minimum_epochs")
    if isinstance(study, dict) and "mode" in study:
        expected_fields.add("mode")
    if isinstance(study, dict) and study.get("mode") == "grid":
        expected_fields.update(
            {"profile", "frame_sizes", "num_frames", "data_sizes", "epoch_values"}
        )
    _fields(study, expected_fields, "study")
    if type(study["schema_version"]) is not int or study["schema_version"] != 1:
        raise ValueError("unsupported study schema_version")
    data = study["dataset"]
    _fields(data, {"name", "path", "split_manifest"}, "dataset")
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ValueError("dataset.name must be a non-empty string")
    if not isinstance(data["path"], str) or not data["path"].strip():
        raise ValueError("dataset.path must be a non-empty local path")
    if data["split_manifest"] is not None and (
        not isinstance(data["split_manifest"], str)
        or not data["split_manifest"].strip()
    ):
        raise ValueError("dataset.split_manifest must be null or a non-empty path")
    selected_config_path = study.get("selected_config")
    if "custom_selected" in study["models"]:
        if not isinstance(selected_config_path, str) or not selected_config_path.strip():
            raise ValueError(
                "model comparison requires selected_config from a completed custom search"
            )
        selected_config_path = _path(selected_config_path)
        try:
            selected = json.loads(Path(selected_config_path).read_text(encoding="utf-8"))
            selected_candidate = selected["candidate"]
            selected_layers = selected_candidate["convlstm_layers"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"selected_config is missing or invalid: {selected_config_path}"
            ) from error
        if selected.get("selection_partition") != "validation" or selected.get("test_access") != "locked":
            raise ValueError("selected_config must be based on validation while test stays locked")
        if not isinstance(selected.get("split"), dict) or not selected["split"].get("manifest_hash"):
            raise ValueError("selected_config must record its split manifest hash")
        if selected.get("seed") not in study["seeds"]:
            raise ValueError("selected_config seed must match the comparison seed")
        if "dataset_dir" in selected and _path(data["path"]) != _path(selected["dataset_dir"]):
            raise ValueError("comparison dataset path differs from the selected search run")
        if study["custom_candidates"]:
            if study["custom_candidates"][0]["convlstm_layers"] != selected_layers:
                raise ValueError("comparison candidate does not match selected_config")
        else:
            study["custom_candidates"] = [{
                "name": "custom_selected",
                "research_question": "Architecture selected using validation data in the custom search.",
                "convlstm_layers": selected_layers,
                "hidden_classifier_width": selected_candidate.get("hidden_classifier_width"),
            }]
        study["selected_config"] = selected_config_path
    elif selected_config_path is not None:
        raise ValueError("selected_config is only valid for a model comparison")

    manifest = CandidateManifest.from_mapping(
        {
            "screening_id": "aad_staged_architectures",
            "candidates": study["custom_candidates"],
        }
    )
    names = [candidate.name for candidate in manifest.candidates]
    if len(names) > 8 or set(names).intersection(
        (*BASELINES, "paper_convlstm_published")
    ):
        raise ValueError(
            "declare at most eight custom candidates with distinct registry names"
        )
    models = study["models"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(name, str) for name in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("models must be a non-empty unique list")
    if set(models) - set([*names, *BASELINES, "paper_convlstm_published"]):
        raise ValueError(
            "unknown model; use a registered candidate or model family"
        )
    if not set(names) <= set(models):
        raise ValueError("every declared custom candidate must be included in models")
    mode = study.get("mode", "staged")
    if not isinstance(mode, str) or mode not in {"staged", "grid"}:
        raise ValueError("unsupported experiment study mode")
    seeds = study["seeds"]
    if (
        not isinstance(seeds, list)
        or not (1 <= len(seeds) <= 3)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "studies require one to three unique seeds"
        )
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
            "dataset_name": data["name"],
            "dataset_dir": _path(data["path"]),
            "split_manifest": (
                _path(data["split_manifest"])
                if data["split_manifest"] is not None
                else None
            ),
            "runs_dir": _path(study["runs_dir"]),
            "seed": seeds[0],
            "height": reference["frame_size"],
            "width": reference["frame_size"],
            "sequence_length": reference["sequence_length"],
            "weight_decay": reference["weight_decay"],
        }
    )
    minimum_epochs = study.get("minimum_epochs", 1)
    if (
        type(minimum_epochs) is not int
        or minimum_epochs < 1
        or minimum_epochs > config.epochs
    ):
        raise ValueError("minimum_epochs must be between 1 and the epoch cap")
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
    """Expand either the explicit smoke grid or the legacy staged plan."""
    rows = []

    def row(stage, model, seed, factor, overrides):
        values = {**config.to_dict(), **overrides, "seed": seed}
        resolved = ExperimentConfig.from_mapping(values)
        trial_id = (
            f"{model}_f{resolved.height}_t{resolved.sequence_length}_"
            f"e{resolved.epochs}_wd{resolved.weight_decay:g}"
        )
        return {
            "stage": stage,
            "model": model,
            "trial_id": trial_id,
            "seed": seed,
            "changed_factor": factor,
            "config": resolved.to_dict(),
            "partition": "validation",
            "test_access": "locked",
            "minimum_epochs": study.get("minimum_epochs", 1),
        }

    if study.get("mode", "staged") == "grid":
        for name in study["models"]:
            for frames, frame_size in study["data_sizes"]:
                for epochs in study["epoch_values"]:
                    for weight_decay in study["factors"]["weight_decays"]:
                        for seed in study["seeds"]:
                            rows.append(
                                row(
                                    "grid",
                                    name,
                                    seed,
                                    "cartesian_grid",
                                    {
                                        "height": frame_size,
                                        "width": frame_size,
                                        "sequence_length": frames,
                                        "epochs": epochs,
                                        "weight_decay": weight_decay,
                                    },
                                )
                            )
        return rows
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
    candidate_names = {
        candidate["name"] for candidate in study["custom_candidates"]
    }
    if study.get("mode", "staged") == "grid":
        label = "local pipeline smoke" if study["profile"] == "local_smoke" else "A100 screening"
        print(
            f"AAD {label}: {len(rows)} Cartesian configurations; "
            + (
                "not paper evidence."
                if study["profile"] == "local_smoke"
                else "single-seed validation screen; confirm before paper claims."
            )
        )
    else:
        is_comparison = bool(set(study["models"]) - candidate_names)
        label = "model-family comparison" if is_comparison else "custom architecture search"
        print(f"AAD {label}: {len(rows)} declared runs; no Cartesian expansion.")
    print(f"Dataset={config.dataset_dir}; split={config.split_manifest}; split_seed=42")
    print(
        f"Fixed: epochs<={config.epochs}, patience={config.early_stopping_patience}, "
        f"batch={config.batch_size}, lr={config.learning_rate}, scheduler={config.scheduler}"
    )
    if study.get("minimum_epochs", 1) > 1:
        print(f"Minimum completed epochs before stopping: {study['minimum_epochs']}")
    print("Selection: validation accuracy, loss, parameters; test remains locked.")
    if study.get("mode", "staged") == "grid":
        if study["profile"] == "local_smoke":
            print("This short local grid checks the pipeline only.")
        else:
            print("The A100 grid ranks candidates using validation only.")
    elif "custom_selected" in study["models"]:
        print(
            "Custom architecture comes only from selected_config.json produced by the validation search; "
            "the committed profile has no fallback architecture."
        )
    else:
        print(
            "Ablations reuse the VALIDATION_WINNER screen runs as the unchanged reference; "
            "no combined factors."
        )
    print(
        "Command:",
        shlex.join(
            [
                sys.executable,
                "-m",
                "src.experiments",
                "--config",
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
    print("Leaf JSON/commands are generated and recorded from this one file at execution.")
    if "paper_convlstm_published" in study["models"]:
        print(
            "PaperConvLSTM is included at its native shared comparison input: "
            f"T={config.sequence_length}, {config.height}x{config.width}, "
            f"batch={config.batch_size}."
        )
    elif study["published_topology"]["enabled"]:
        print("Separate native PaperConvLSTM runs are enabled (not ranked).")
    else:
        print("Native PaperConvLSTM: T=50, 50x50, batch=1, separate/unranked; enabled=False.")


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
                "family": entries[0].get("family"),
                "model_class": entries[0].get("model_class"),
                "partition": "validation",
                "num_params": entries[0]["num_params"],
                "validation_metrics": {
                    key: statistics.mean(values) for key, values in scores.items()
                },
                "validation_std": {
                    key: statistics.stdev(values) if len(values) > 1 else 0.0
                    for key, values in scores.items()
                },
                "seeds": seeds,
                "runs": entries,
            }
        )
    return rank_validation_results(aggregated)


def rank_smoke_grid(records, rows):
    """Rank each completed smoke configuration using validation data only."""
    if len(records) != len(rows):
        raise ValueError("smoke grid must complete every declared configuration")
    if len({item["split"]["manifest_hash"] for item in records}) != 1:
        raise ValueError("smoke configurations must use the same fixed split")
    ranked = []
    for record, row in zip(records, rows):
        if record["partition"] != "validation" or record["test_access"] != "locked":
            raise ValueError("smoke selection requires validation-only evidence")
        if record["name"] != row["model"] or record["seed"] != row["seed"]:
            raise ValueError("smoke result does not match its declared configuration")
        if record["experiment_config"] != row["config"]:
            raise ValueError("smoke result configuration differs from the grid")
        ranked.append(
            {
                "name": row["trial_id"],
                "model": row["model"],
                "partition": "validation",
                "num_params": record["num_params"],
                "validation_metrics": record["validation_metrics"],
                "seed": row["seed"],
                "experiment_config": row["config"],
            }
        )
    return rank_validation_results(ranked)


def execute_study(path, study, config, manifest, run_trial, run_evaluation=None):
    """Run validation-only jobs, freeze comparison, then optionally evaluate test."""
    from .dataset_source import resolve_dataset

    root = resolve_dataset(config.dataset_name, config.dataset_dir, ROOT)
    config = ExperimentConfig.from_mapping(
        {**config.to_dict(), "dataset_dir": str(root)}
    )
    canonical = json.dumps(study, sort_keys=True).encode()
    candidate_names = {candidate.name for candidate in manifest.candidates}
    if study.get("mode") == "grid" and study["profile"] == "local_smoke":
        study_kind = "local_smoke"
    elif set(study["models"]) <= candidate_names:
        study_kind = "custom_architecture_search"
    else:
        study_kind = "model_family_comparison"
    group = RunContext(
        config.runs_dir,
        purpose="studies",
        dataset_path=root,
        label=(
            config.dataset_name + "-" + study["profile"]
            if study.get("mode") == "grid"
            else config.dataset_name + "-" + study_kind
        ),
        arguments={"study_file": str(Path(path).resolve()), "study": study},
        metadata={
            "study_sha256": hashlib.sha256(canonical).hexdigest(),
            "study_kind": study_kind,
            "test_access": "locked",
            "pipeline_smoke_only": study.get("mode") == "grid"
            and study["profile"] == "local_smoke",
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
        if row["minimum_epochs"] > 1:
            args += ["--minimum-epochs", str(row["minimum_epochs"])]
        if row["model"] in {candidate.name for candidate in manifest.candidates}:
            args += ["--candidates-config", str(candidates)]
        write_json(
            job_dir / "command.json", [sys.executable, "-m", "src.experiments", *args]
        )
        print(f"\nStudy job {number}: {row['stage']} / {row['model']} / seed {row['seed']}", flush=True)
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
    is_grid = study.get("mode", "staged") == "grid"
    if is_grid:
        for row in rows:
            row["config"]["dataset_dir"] = str(root)
            row["config"]["runs_dir"] = config.runs_dir
            candidate = next(
                (item for item in manifest.candidates if item.name == row["model"]),
                None,
            )
            if candidate is not None:
                row["config"] = ExperimentConfig.from_mapping(
                    {
                        **row["config"],
                        "convlstm_layers": candidate.to_dict()["convlstm_layers"],
                        "hidden_classifier_width": candidate.hidden_classifier_width,
                    }
                ).to_dict()
    try:
        if is_grid:
            results = [run(row) for row in rows]
            ranking = rank_smoke_grid(results, rows)
            winner = ranking[0]
            selected_row = next(
                row for row in rows if row["trial_id"] == winner["name"]
            )
            selected_values = json.loads(
                (
                    group.run_dir
                    / "jobs"
                    / f"{rows.index(selected_row) + 1:02d}"
                    / "config.json"
                )
                .read_text(encoding="utf-8")
            )
            selected_config_path = write_json(
                group.run_dir / "selected_config.json", selected_values
            )
            smoke_only = study["profile"] == "local_smoke"
            selection = {
                "study_kind": study_kind,
                "selected_trial": winner["name"],
                "selected_model": winner["model"],
                "selection_partition": "validation",
                "selected_config": str(selected_config_path),
                "selected_experiment_config": winner["experiment_config"],
                "ranking": ranking,
                "test_access": "locked",
                "pipeline_smoke_only": smoke_only,
                "note": (
                    "Local pipeline smoke only; not paper evidence."
                    if smoke_only
                    else "Single-seed validation screen; confirm before paper claims."
                ),
            }
            write_json(group.run_dir / "selection.json", selection)
            summary = {**selection, "jobs": completed}
            summary_path = write_json(group.run_dir / "summary.json", summary)
            group.complete(
                artifacts={
                    "summary": str(summary_path),
                    "selection": str(group.run_dir / "selection.json"),
                    "selected_config": str(selected_config_path),
                },
                results={
                    "selected_model": winner["model"],
                    "completed_jobs": len(completed),
                    "test_access": "locked",
                    "pipeline_smoke_only": smoke_only,
                },
            )
            return group.run_dir

        from tqdm import tqdm

        screen_rows = [row for row in rows if row["stage"] == "screen"]
        screen = [run(row) for row in tqdm(screen_rows, desc="Study models")]
        ranking = aggregate_screen(screen, study["models"], study["seeds"])
        winner = ranking[0]["name"]
        selection = {
            "study_kind": study_kind,
            "selected_model": winner,
            "selection_partition": "validation",
            "ranking": ranking,
            "test_access": "locked",
        "ranking_rule": [
                "mean_validation_accuracy",
                "mean_validation_loss",
            "parameters",
        ],
        "single_seed_evidence": True,
            "reference_runs": [item for item in screen if item["name"] == winner],
        }
        if study.get("selected_config"):
            selected_source = json.loads(
                Path(study["selected_config"]).read_text(encoding="utf-8")
            )
            selected_split_hash = selected_source["split"]["manifest_hash"]
            if any(item["split"]["manifest_hash"] != selected_split_hash for item in screen):
                raise ValueError("comparison does not use the selected search split manifest")
            selection["search_split_manifest_hash"] = selected_split_hash
        if study_kind == "custom_architecture_search":
            winning_row = next(row for row in screen_rows if row["model"] == winner)
            winning_result = next(item for item in screen if item["name"] == winner)
            candidate = next(item for item in manifest.candidates if item.name == winner)
            selected_config = {
                "candidate": {
                    "name": winner,
                    "convlstm_layers": candidate.to_dict()["convlstm_layers"],
                    "hidden_classifier_width": candidate.hidden_classifier_width,
                    "research_question": candidate.research_question,
                },
                "input": {
                    "sequence_length": winning_row["config"]["sequence_length"],
                    "height": winning_row["config"]["height"],
                    "width": winning_row["config"]["width"],
                },
                "seed": winning_result["seed"],
                "validation_metrics": winning_result["validation_metrics"],
                "num_params": winning_result["num_params"],
                "source_run": completed[screen_rows.index(winning_row)]["run_dir"],
                "dataset_dir": winning_result["dataset_dir"],
                "split": winning_result["split"],
                "selection_partition": "validation",
                "selection_rule": ["validation_accuracy", "validation_loss", "parameters"],
                "single_seed_evidence": True,
                "test_access": "locked",
            }
            write_json(group.run_dir / "selected_config.json", selected_config)
            selection["selected_config"] = str(group.run_dir / "selected_config.json")
            selection["selected_candidate"] = selected_config["candidate"]
            selection["selected_input"] = selected_config["input"]
        write_json(group.run_dir / "selection.json", selection)
        print(f"Frozen validation-selected reference: {winner}", flush=True)
        followup_rows = [row for row in study_rows(study, config, winner) if row["stage"] != "screen"]
        for row in tqdm(followup_rows, desc="One-factor checks"):
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
            "study_kind": study_kind,
            **selection,
            "jobs": completed,
            "ablation_comparisons": ablation_comparisons,
            "native_comparable": False,
            "ablation_factors_combined": False,
            "note": "No test evaluation; ablations do not silently replace the frozen reference.",
        }
        test_results = []
        if study_kind == "model_family_comparison" and study.get("selected_config"):
            if run_evaluation is None:
                raise ValueError("comparison requires a post-freeze test evaluator")
            write_json(group.run_dir / "validation_frozen.json", selection)
            evaluation_rows = [item for item in completed if item["stage"] == "screen"]
            for item in tqdm(evaluation_rows, desc="Frozen test evaluation"):
                result = item["result"]
                evaluation_config = {
                    **result["experiment_config"],
                    "checkpoint_path": result["selected_checkpoint"],
                }
                config_path = write_json(
                    group.run_dir / "test_configs" / f"{safe_filename(item['result']['name'])}.json",
                    evaluation_config,
                )
                evaluation_run = run_evaluation(config_path)
                report_path = Path(evaluation_run) / "metrics" / "final.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("partition") != "test":
                    raise ValueError("test evaluator did not return test-partition results")
                if report["split"]["manifest_hash"] != result["split"]["manifest_hash"]:
                    raise ValueError("test evaluation changed the frozen split manifest")
                test_results.append({
                    "model": result["name"],
                    "validation_selected_best": result["name"] == winner,
                    "run_dir": str(evaluation_run),
                    "metrics": report["metrics"],
                    "confusion_matrix": report["artifacts"]["confusion_matrix"],
                    "prediction_examples": report["prediction_examples"] if result["name"] == winner else None,
                    "partition": "test",
                })
            summary["test_evaluation"] = {
                "frozen_before_test": True,
                "selection_source": "validation",
                "best_validation_model": winner,
                "single_seed_evidence": True,
                "models": test_results,
            }
        summary_path = write_json(group.run_dir / "summary.json", summary)
        group.complete(
            artifacts={
                "summary": str(summary_path),
                "selection": str(group.run_dir / "selection.json"),
                "selected_config": (
                    str(group.run_dir / "selected_config.json")
                    if (group.run_dir / "selected_config.json").is_file()
                    else None
                ),
                "validation_frozen": (
                    str(group.run_dir / "validation_frozen.json")
                    if (group.run_dir / "validation_frozen.json").is_file()
                    else None
                ),
                "test_evaluation": test_results,
            },
            results={
                "selected_model": winner,
                "completed_jobs": len(completed),
                "test_access": "evaluated_after_validation_freeze"
                if test_results
                else "locked",
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
