"""Evidence and restart guards for the existing runner's two-stage matrix mode.

No CLI or model implementation lives here. Legacy studies remain readable.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter

from .experiment_config import ExperimentConfig
from .utils import RunContext

CUSTOM_SLOTS = ["custom_top1", "custom_top2", "custom_top3"]
COMPARISON_MODELS = CUSTOM_SLOTS + [
    "paper_convlstm_published",
    "r3d_18",
    "mc3_18",
    "swin3d_t",
    "swin3d_s",
]
ADAPTED_COMPARISON_MODELS = [
    n if n != "paper_convlstm_published" else "paper_convlstm_adapted"
    for n in COMPARISON_MODELS
]
NOTE = (
    "Single seed; resolution variation is not seed uncertainty. "
    "Finite validation search, not proof of a global optimum or unseen-data generalisation. "
    "Clip-stratified split does not establish source-group independence."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def inventory(directory):
    """Hash all completed evidence, including the selected checkpoint."""
    return {
        str(p.resolve()): file_hash(p)
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file()
    }


def verify_inventory(files):
    if not files:
        raise ValueError("missing artifact inventory")
    for path, expected in files.items():
        if not Path(path).is_file() or file_hash(path) != expected:
            raise ValueError(f"Saved evidence missing or modified: {path}")


def validate_protocol(study, config):
    factors = study["factors"]
    if study["seeds"] != [42] or "confirmation_seed" in study:
        raise ValueError("matrix protocol requires seed 42 only; no confirmation seed")
    if study["published_topology"]["enabled"]:
        raise ValueError("no extra published topology stage in the two-stage protocol")
    if factors["sequence_lengths"] != [config.sequence_length] or factors[
        "weight_decays"
    ] != [config.weight_decay]:
        raise ValueError("matrix protocol fixes frame count and weight decay")
    if study["mode"] == "multiresolution":
        if not 3 <= len(study["models"]) <= 12 or len(factors["frame_sizes"]) != 3:
            raise ValueError(
                "search requires 3-12 custom architectures at exactly three resolutions"
            )
        if set(study["models"]) != {c["name"] for c in study["custom_candidates"]}:
            raise ValueError("resolution search accepts custom architectures only")
        architectures = {
            value_hash([c["convlstm_layers"], c["hidden_classifier_width"]])
            for c in study["custom_candidates"]
        }
        if len(architectures) != len(study["models"]):
            raise ValueError(
                "search architectures must be distinct, not renamed duplicates"
            )
    else:
        adapted = study.get("comparison_protocol") == "wide_final_v1"
        if study["models"] != (
            ADAPTED_COMPARISON_MODELS if adapted else COMPARISON_MODELS
        ):
            raise ValueError(
                "comparison requires the top three and exactly five declared baselines"
            )
        expected_input = (16, 64, 64, config.batch_size) if adapted else (50, 50, 50, 1)
        if (
            config.sequence_length,
            config.height,
            config.width,
            config.batch_size,
        ) != expected_input:
            raise ValueError(
                "all comparison models require native 50-frame 50x50 input and batch 1"
            )
        if factors["frame_sizes"] != ([64] if adapted else [50]):
            raise ValueError("comparison cannot vary input resolution")


def validation_accuracy(result, exact=False):
    """New protocols rank actual counts; legacy selections retain their old rule."""
    if not exact:
        return result["validation_metrics"]["accuracy"]
    matrix = result.get("validation_confusion_matrix", {}).get("matrix")
    if matrix is not None:
        if not matrix or any(len(row) != len(matrix) for row in matrix):
            raise ValueError("invalid validation confusion matrix")
        if any(type(n) is not int or n < 0 for row in matrix for n in row):
            raise ValueError("validation counts must be nonnegative integers")
        total = sum(map(sum, matrix))
        correct = sum(row[i] for i, row in enumerate(matrix))
    else:
        path = result.get("validation_predictions")
        if not path or not Path(path).is_file():
            raise ValueError("exact ranking requires validation counts or predictions")
        records = read_json(path)
        if any(r.get("partition") != "validation" for r in records):
            raise ValueError("exact ranking requires validation-only predictions")
        total = len(records)
        correct = sum(r["target"] == r["predicted"] for r in records)
    if total <= 0:
        raise ValueError("empty validation evidence")
    accuracy = Fraction(correct, total)
    if not math.isclose(
        float(accuracy), result["validation_metrics"]["accuracy"], abs_tol=2e-6
    ):
        raise ValueError("validation accuracy differs from exact counts")
    return accuracy


def rank_resolutions(records, models, sizes, seed=42, exact=False):
    expected = {(name, size, seed) for name in models for size in sizes}
    actual = {(r["name"], r["experiment_config"]["height"], r["seed"]) for r in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            "search must complete every architecture/resolution exactly once"
        )
    protocols, splits, datasets = set(), set(), set()
    for r in records:
        cfg = r["experiment_config"]
        if r["partition"] != "validation" or r["test_access"] != "locked":
            raise ValueError("ranking requires validation-only evidence")
        if cfg["height"] != cfg["width"] or cfg["seed"] != seed:
            raise ValueError("incompatible search input/seed")
        protocols.add(
            value_hash(
                {
                    k: v
                    for k, v in cfg.items()
                    if k
                    not in {
                        "height",
                        "width",
                        "convlstm_layers",
                        "hidden_classifier_width",
                        "runs_dir",
                    }
                }
            )
        )
        splits.add(r["split"]["manifest_hash"])
        datasets.add(r["dataset_dir"])
        if any(
            not math.isfinite(r["validation_metrics"][k]) for k in ("accuracy", "loss")
        ):
            raise ValueError("non-finite validation score")
    if len(protocols) != 1 or len(splits) != 1 or len(datasets) != 1:
        raise ValueError(
            "search must share the fixed training protocol, dataset and split"
        )
    ranks = {}
    for size in sizes:
        entries = sorted(
            [r for r in records if r["experiment_config"]["height"] == size],
            key=lambda r: (
                -validation_accuracy(r, exact),
                r["validation_metrics"]["loss"],
                r["num_params"],
                r["name"],
            ),
        )
        ranks[size] = {r["name"]: i + 1 for i, r in enumerate(entries)}
    ranking = []
    for name in models:
        entries = [
            next(
                r
                for r in records
                if r["name"] == name and r["experiment_config"]["height"] == size
            )
            for size in sizes
        ]
        if (
            len({r["num_params"] for r in entries}) != 1
            or len(
                {
                    value_hash(
                        [
                            r["experiment_config"]["convlstm_layers"],
                            r["experiment_config"]["hidden_classifier_width"],
                        ]
                    )
                    for r in entries
                }
            )
            != 1
        ):
            raise ValueError("architecture/parameter count changed across resolutions")
        ranking.append(
            {
                "name": name,
                "mean_validation_accuracy": float(
                    statistics.mean(validation_accuracy(r, exact) for r in entries)
                ),
                **(
                    {
                        "accuracy_fraction": list(
                            statistics.mean(
                                validation_accuracy(r, True) for r in entries
                            ).as_integer_ratio()
                        )
                    }
                    if exact
                    else {}
                ),
                "mean_validation_loss": statistics.mean(
                    r["validation_metrics"]["loss"] for r in entries
                ),
                "worst_validation_accuracy": min(
                    r["validation_metrics"]["accuracy"] for r in entries
                ),
                "num_params": entries[0]["num_params"],
                "per_resolution": [
                    {
                        "frame_size": size,
                        "rank": ranks[size][name],
                        "validation_metrics": r["validation_metrics"],
                    }
                    for size, r in zip(sizes, entries)
                ],
            }
        )
    return sorted(
        ranking,
        key=lambda r: (
            -(
                Fraction(*r["accuracy_fraction"])
                if exact
                else r["mean_validation_accuracy"]
            ),
            r["mean_validation_loss"],
            r["num_params"],
            r["name"],
        ),
    )


def selection_bundle(study, jobs, group):
    ranking = rank_resolutions(
        [j["result"] for j in jobs], study["models"], study["factors"]["frame_sizes"]
    )
    candidates = {c["name"]: c for c in study["custom_candidates"]}
    first = jobs[0]["result"]
    return {
        "protocol": "multiresolution_top3_v1",
        "selection_partition": "validation",
        "test_access": "locked",
        "seed": 42,
        "single_seed_evidence": True,
        "note": NOTE,
        "source_group": str(Path(group).resolve()),
        "source_study_sha256": file_hash(Path(group) / "study.json"),
        "dataset_dir": first["dataset_dir"],
        "split": first["split"],
        "ranking_rule": [
            "equal_weight_mean_validation_accuracy_desc",
            "mean_validation_loss_asc",
            "parameters_asc",
            "name_asc",
        ],
        "ranking": ranking,
        "top3": [
            {
                "comparison_name": CUSTOM_SLOTS[i],
                "candidate": candidates[r["name"]],
                "sources": [
                    {
                        "run_dir": j["run_dir"],
                        "config_sha256": value_hash(j["result"]["experiment_config"]),
                        "frame_size": j["result"]["experiment_config"]["height"],
                    }
                    for j in jobs
                    if j["result"]["name"] == r["name"]
                ],
            }
            for i, r in enumerate(ranking[:3])
        ],
    }


def load_selection(path):
    """Do not accept handwritten winners, incomplete matrices or stale artifacts."""
    selected = read_json(path)
    if selected.get("protocol") in {
        "capacity_top3_v1",
        "capacity_top3_v2",
        "capacity_top3_v3",
    }:
        from .capacity_search import load_capacity_selection

        return load_capacity_selection(path)
    if selected.get("protocol") != "multiresolution_top3_v1":
        raise ValueError(
            "selected_config requires a complete multi-resolution top-three search"
        )
    group = Path(selected["source_group"])
    if read_json(group / "run.json")["status"] != "complete":
        raise ValueError("selected search is incomplete")
    study = read_json(group / "study.json")
    progress = read_json(group / "progress.json")
    for job in progress["jobs"]:
        receipt = read_json(job["receipt"])
        if receipt["status"] != "complete" or receipt["result"] != job["result"]:
            raise ValueError("search receipt does not match completed evidence")
        verify_inventory(receipt["files"])
    expected = selection_bundle(study, progress["jobs"], group)
    if selected != expected:
        raise ValueError("selected_config differs from complete validation evidence")
    if (
        file_hash(selected["split"]["manifest_path"])
        != selected["split"]["manifest_hash"]
    ):
        raise ValueError("selected split manifest has changed")
    return selected


def comparison_candidates(study):
    from .study_config import _path

    if not study.get("selected_config"):
        raise ValueError(
            "comparison requires selected_config from a completed custom search"
        )
    selected_path = _path(study["selected_config"])
    selected = load_selection(selected_path)
    if _path(study["dataset"]["path"]) != _path(selected["dataset_dir"]) or study[
        "seeds"
    ] != [selected["seed"]]:
        raise ValueError("comparison dataset/seed differs from search")
    if (
        not study["dataset"]["split_manifest"]
        or file_hash(_path(study["dataset"]["split_manifest"]))
        != selected["split"]["manifest_hash"]
    ):
        raise ValueError("comparison must use the selected search split manifest")
    if (
        selected.get("protocol") == "capacity_top3_v3"
        and study.get("comparison_protocol") != "wide_final_v1"
    ):
        raise ValueError("075 selections require the adapted final-input protocol")
    candidates = [
        {**item["candidate"], "name": item["comparison_name"]}
        for item in selected["top3"]
    ]
    if study["custom_candidates"] and study["custom_candidates"] != candidates:
        raise ValueError("comparison candidate does not match selected_config")
    study["selected_config"] = selected_path
    return candidates


def _validate_leaf(result, row, values, split_hash):
    if (result["name"], result["seed"], result["partition"], result["test_access"]) != (
        row["model"],
        values["seed"],
        "validation",
        "locked",
    ):
        raise ValueError("leaf did not return the declared validation-only job")
    if (
        result["experiment_config"] != values
        or result["split"]["manifest_hash"] != split_hash
    ):
        raise ValueError("leaf config/split differs from the fixed protocol")
    sidecar = read_json(Path(result["selected_checkpoint"]).with_suffix(".json"))
    if (
        sidecar["selection_partition"] != "validation"
        or sidecar["checkpoint_role"] != "validation_selected_lowest_loss"
    ):
        raise ValueError("checkpoint was not selected on validation loss")
    if (
        sidecar["experiment_config"] != values
        or sidecar["split_manifest_hash"] != split_hash
    ):
        raise ValueError("checkpoint/config/split mismatch")
    if sidecar["selected_epoch"] != result["checkpoint_selection"]["selected_epoch"]:
        raise ValueError("checkpoint epoch mismatch")
    if (
        sidecar.get("seed") != values["seed"]
        or sidecar.get("model_registry_entry", {}).get("name") != row["model"]
    ):
        raise ValueError("checkpoint model/seed mismatch")
    if not Path(result["selected_checkpoint"]).is_file():
        raise ValueError("selected checkpoint missing")


def _table_row(job, report, extended=False, raw=False):
    result = job["result"]
    row = {
        "experiment_number": job.get("experiment_number"),
        "model": result["name"],
        "learning_rate": result["experiment_config"]["learning_rate"],
        "weight_decay": result["experiment_config"]["weight_decay"],
        "frames": result["experiment_config"]["sequence_length"],
        "height": result["experiment_config"]["height"],
        "width": result["experiment_config"]["width"],
        "family": result["family"],
        "seed": result["seed"],
        "num_params": result["num_params"],
        "weight_tensor_bytes": result["weight_tensor_bytes"],
        "checkpoint_bytes": result["checkpoint_bytes"],
        "actual_epochs": result["early_stopping"]["actual_epochs"],
        "selected_epoch": result["checkpoint_selection"]["selected_epoch"],
        "train_seconds": result["train_time_s"],
        "validation_metric_pass_seconds": result["validation_metric_pass_seconds"],
        "test_metric_pass_seconds": report["metric_pass_seconds"],
        "split_sha256": result["split"]["manifest_hash"],
        "checkpoint": result["selected_checkpoint"],
        "training_run": job["run_dir"],
        "history": result["history"],
        "curves": result["training_curves"],
        "validation_predictions": result["validation_predictions"],
        "test_predictions": report["predictions"],
    }
    for partition, metrics in (
        ("validation", result["validation_metrics"]),
        ("test", report["metrics"]),
    ):
        for key in ("loss", "accuracy", "precision", "recall", "f1"):
            label = f"micro_{key}" if key in {"precision", "recall", "f1"} else key
            row[f"{partition}_{label}"] = metrics[key]
    if extended:
        from .study_reporting import extended_metrics

        class_count = len(read_json(result["split"]["manifest_path"])["class_names"])
        for partition, path in (
            ("validation", result["validation_predictions"]),
            ("test", report["predictions"]),
        ):
            records = read_json(path)
            metrics = extended_metrics(
                [r["predicted"] for r in records],
                [r["target"] for r in records],
                class_count,
            )
            row.update({f"{partition}_{k}": v for k, v in metrics.items()})
    if raw:
        from .wide_protocol import raw_result_fields, unaveraged_row

        row.update(raw_result_fields(result))
        row.update(raw_result_fields(result, "test", report["predictions"]))
        row.update(result.get("efficiency", {}))
        row["target_fps"] = result["experiment_config"]["target_fps"]
        row["matched_training"] = {
            k: result.get("training_evaluation", {}).get(k)
            for k in ("train_minus_validation_accuracy", "validation_minus_train_loss")
        }
        row = unaveraged_row(row)
    return row


def execute_matrix(path, study, config, manifest, run_trial, run_evaluation):
    """Run the explicit matrix, with durable receipts and no automatic repeat tests."""
    from .dataset import AHARDataset, load_split_subsets, resolve_split_manifest_path
    from .dataset_source import resolve_dataset
    from .study_config import ROOT, study_rows

    raw_presentation = study.get("comparison_protocol") == "wide_final_v1"
    root = resolve_dataset(config.dataset_name, config.dataset_dir, ROOT)
    source = AHARDataset(root, config.sequence_length, (config.width, config.height))
    split_path = resolve_split_manifest_path(
        root, config.split_manifest, config.split_seed
    )
    _, _, test_set, split = load_split_subsets(
        source,
        split_path,
        seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    # Metadata inventory only: never decode a test clip while preparing selection.
    data_identity = [
        (str(item.resolve()), item.stat().st_size, item.stat().st_mtime_ns)
        for p, _ in source.samples
        for item in (
            sorted(f for f in Path(p).rglob("*") if f.is_file())
            if Path(p).is_dir()
            else [Path(p)]
        )
    ]
    code_identity = {p.name: file_hash(p) for p in sorted((ROOT / "src").glob("*.py"))}
    code_identity["requirements.txt"] = file_hash(ROOT / "requirements.txt")
    identity = value_hash(
        {
            "study": study,
            "split": split["manifest_hash"],
            "data": data_identity,
            "code": code_identity,
        }
    )
    receipts_root = Path(config.runs_dir) / "study_receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)
    # Stable request slot: changed code/data must not silently start fresh tests.
    index_path = receipts_root / f"{value_hash(study)}.json"
    lock = index_path.with_suffix(".lock")
    try:
        handle = lock.open("x")
    except FileExistsError as error:
        raise RuntimeError(
            f"Study lock exists: {lock}. Stop concurrent workers; inspect saved progress before recovery."
        ) from error
    with handle:
        handle.write(str(Path(path).resolve()))
    started = perf_counter()
    group = None
    state = None
    try:
        if index_path.exists():
            group = Path(read_json(index_path)["run_dir"])
            saved_state = read_json(group / "progress.json")
            if saved_state["identity"] != identity:
                raise ValueError(
                    "resume code/data/split identity changed; inspect the existing study, do not silently retrain/retest"
                )
            state = saved_state
            print(
                f"Verifying saved study (no silent retraining/retesting): {group}",
                flush=True,
            )
        else:
            context = RunContext(
                config.runs_dir,
                "studies",
                root,
                study["mode"],
                {"study_file": str(Path(path).resolve()), "study": study},
                {
                    "identity": identity,
                    "test_access": "locked",
                    "single_seed_evidence": True,
                    "resume_index": str(index_path.resolve()),
                },
            )
            group = context.run_dir
            atomic_json(group / "study.json", study)
            atomic_json(group / "candidates.json", manifest.to_dict())
            archived_split = group / "split_manifest.json"
            archived_split.write_bytes(Path(split["manifest_path"]).read_bytes())
            state = {
                "identity": identity,
                "status": "running",
                "jobs": [],
                "tests": [],
                "test_access": "locked",
            }
            atomic_json(group / "progress.json", state)
            atomic_json(index_path, {"run_dir": str(group.resolve())})
        config = ExperimentConfig.from_mapping(
            {
                **config.to_dict(),
                "dataset_dir": str(Path(root).resolve()),
                "split_manifest": str((group / "split_manifest.json").resolve()),
            }
        )
        experiment_offset = 0
        recipe = None
        if study["mode"] == "top3_comparison":
            selected = load_selection(study["selected_config"])
            if split["manifest_hash"] != selected["split"]["manifest_hash"]:
                raise ValueError("comparison split differs from selected search")
            if run_evaluation is None:
                raise ValueError("comparison requires a post-freeze test evaluator")
            if selected.get("protocol") in {"capacity_top3_v2", "capacity_top3_v3"}:
                experiment_offset = len(
                    read_json(Path(selected["source_group"]) / "progress.json")["jobs"]
                )
            if raw_presentation:
                from .wide_protocol import print_final_input

                print_final_input(selected["final_input"])
            if study.get("recipe_transfer"):
                from .comparison_recipe import select_recipe

                if (group / "validation_frozen.json").exists() and not (
                    group / "recipe_selection.json"
                ).is_file():
                    raise ValueError("Frozen comparison is missing its recipe evidence")
                config, recipe = select_recipe(
                    study, config, selected, group, split, run_trial, experiment_offset
                )
                experiment_offset += len(recipe["jobs"])
                state["recipe_selection"] = str(
                    (group / "recipe_selection.json").resolve()
                )
                state["effective_config"] = config.to_dict()
                atomic_json(group / "progress.json", state)
        rows = study_rows(study, config)
        candidates = {c.name: c for c in manifest.candidates}
        jobs = []
        for number, row in enumerate(rows, 1):
            experiment_number = experiment_offset + number
            jobdir = group / "jobs" / f"{number:02d}"
            receipt_path = jobdir / "receipt.json"
            values = dict(row["config"])
            if row["model"] in candidates:
                c = candidates[row["model"]]
                values.update(
                    convlstm_layers=c.to_dict()["convlstm_layers"],
                    hidden_classifier_width=c.hidden_classifier_width,
                )
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if receipt["status"] != "complete":
                    raise RuntimeError(
                        f"Job {number} was interrupted/failed. Evidence preserved at {jobdir}; review before an explicit retry, never silently retrain."
                    )
                verify_inventory(receipt["files"])
                result = receipt["result"]
                leaf_run = receipt["run_dir"]
                if (
                    receipt.get("experiment_number", experiment_number)
                    != experiment_number
                ):
                    raise ValueError("Saved comparison experiment number changed")
                print(
                    f"Experiment {experiment_number}: verified {row['model']} (reused)",
                    flush=True,
                )
            else:
                if (
                    study.get("recipe_transfer")
                    and (group / "validation_frozen.json").exists()
                ):
                    raise ValueError(
                        "Frozen comparison receipt missing; no retraining after test selection"
                    )
                leaf = atomic_json(jobdir / "config.json", values)
                args = [
                    "--config",
                    str(leaf),
                    "--model",
                    row["model"],
                    "--run-label",
                    study["mode"],
                    "--trial-name",
                    f"{experiment_number:03d}",
                    "--changed-factor",
                    row["changed_factor"],
                ]
                if raw_presentation:
                    args += [
                        "--matched-training-evaluation",
                        "--per-class-reporting",
                        "--campaign-number",
                        str(experiment_number),
                        "--campaign-maximum",
                        "142",
                        "--campaign-stage",
                        "final_comparison",
                    ]
                if row["minimum_epochs"] > 1:
                    args += ["--minimum-epochs", str(row["minimum_epochs"])]
                if row["model"] in candidates:
                    args += ["--candidates-config", str(group / "candidates.json")]
                atomic_json(
                    jobdir / "command.json",
                    [sys.executable, "-m", "src.experiments", *args],
                )
                atomic_json(
                    receipt_path,
                    {
                        "status": "running",
                        "config_sha256": value_hash(values),
                        "experiment_number": experiment_number,
                    },
                )
                print(
                    f"Experiment {experiment_number}/{142 if raw_presentation else experiment_offset + len(rows)} maximum | {study['mode']} {number}/{len(rows)}: {row['model']}, {values['height']}x{values['width']}",
                    flush=True,
                )
                before = set((Path(config.runs_dir) / "experiments").glob("*"))
                try:
                    leaf_run = Path(run_trial(args)).resolve()
                    payload = read_json(leaf_run / "summary.json")
                    if len(payload["all"]) != 1:
                        raise ValueError("one result required per job")
                    result = payload["all"][0]
                    _validate_leaf(result, row, values, split["manifest_hash"])
                    if read_json(leaf_run / "run.json")["status"] != "complete":
                        raise ValueError("leaf lifecycle incomplete")
                    atomic_json(
                        receipt_path,
                        {
                            "status": "complete",
                            "experiment_number": experiment_number,
                            "run_dir": str(leaf_run),
                            "result": result,
                            "files": inventory(leaf_run),
                        },
                    )
                except BaseException as error:
                    partial = (
                        set((Path(config.runs_dir) / "experiments").glob("*")) - before
                    )
                    for folder in partial:
                        lifecycle = folder / "run.json"
                        if lifecycle.exists():
                            atomic_json(
                                lifecycle,
                                {
                                    **read_json(lifecycle),
                                    "status": "failed",
                                    "error": str(error),
                                },
                            )
                    atomic_json(
                        receipt_path,
                        {
                            "status": "failed",
                            "error": str(error),
                            "partial_runs": [str(p.resolve()) for p in partial],
                        },
                    )
                    raise
            _validate_leaf(result, row, values, split["manifest_hash"])
            if study["mode"] == "top3_comparison" and selected.get("protocol") in {
                "capacity_top3_v2",
                "capacity_top3_v3",
            }:
                actual = result["early_stopping"]["actual_epochs"]
                if not row["minimum_epochs"] <= actual <= values["epochs"]:
                    raise ValueError(
                        "comparison checkpoint does not satisfy the longer-training budget"
                    )
                if not 1 <= result["checkpoint_selection"]["selected_epoch"] <= actual:
                    raise ValueError(
                        "comparison selected epoch exceeds completed training"
                    )
            if raw_presentation:
                from .capacity_search import validate_matched_training

                validate_matched_training(
                    result, {"config": values}, source.num_classes
                )
            jobs.append(
                {
                    "experiment_number": experiment_number,
                    "run_dir": str(leaf_run),
                    "receipt": str(receipt_path.resolve()),
                    "result": result,
                }
            )
            state.update(jobs=jobs)
            atomic_json(group / "progress.json", state)
        if study["mode"] == "multiresolution":
            bundle = selection_bundle(study, jobs, group)
            atomic_json(group / "selected_config.json", bundle)
            atomic_json(group / "selection.json", bundle)
            summary = {**bundle, "jobs": jobs}
            print(f"Top-three selection: {group / 'selected_config.json'}", flush=True)
        else:

            exact = selected.get("protocol") in {"capacity_top3_v2", "capacity_top3_v3"}

            def order(j):
                r = j["result"]
                return (
                    -validation_accuracy(r, exact),
                    r["validation_metrics"]["loss"],
                    r["num_params"],
                    r["name"],
                )

            best_custom = min(
                [j for j in jobs if j["result"]["name"] in CUSTOM_SLOTS], key=order
            )["result"]["name"]
            frozen = {
                "selection_partition": "validation",
                "best_custom": best_custom,
                "seed": 42,
                "split_sha256": split["manifest_hash"],
                "note": NOTE,
                **(
                    {
                        "recipe_selection_sha256": file_hash(
                            group / "recipe_selection.json"
                        )
                    }
                    if recipe
                    else {}
                ),
                **(
                    {
                        "ranking_version": "exact_counts_v1",
                        "search_selection_sha256": file_hash(study["selected_config"]),
                        "source_audit_sha256": selected["source_audit_sha256"],
                    }
                    if exact
                    else {}
                ),
                "models": [
                    {
                        "model": j["result"]["name"],
                        "checkpoint": j["result"]["selected_checkpoint"],
                        "checkpoint_sha256": file_hash(
                            j["result"]["selected_checkpoint"]
                        ),
                        "config_sha256": value_hash(j["result"]["experiment_config"]),
                    }
                    for j in jobs
                ],
            }
            freeze_path = group / "validation_frozen.json"
            if freeze_path.exists() and read_json(freeze_path) != frozen:
                raise ValueError(
                    "frozen validation models/configs changed; test forbidden"
                )
            atomic_json(freeze_path, frozen)
            expected_test_sources = {
                str(Path(source.samples[i][0]).resolve()) for i in test_set.indices
            }
            table, reports = [], []
            for job, model in zip(jobs, frozen["models"]):
                name = model["model"]
                print(
                    f"Test {len(reports) + 1}/{len(jobs)} | experiment {job['experiment_number']}: {name}",
                    flush=True,
                )
                receipt_path = group / "test_receipts" / f"{name}.json"
                evaluation_config = {
                    **job["result"]["experiment_config"],
                    "checkpoint_path": model["checkpoint"],
                    "prediction_samples_per_category": 3 if name == best_custom else 0,
                }
                if receipt_path.exists():
                    receipt = read_json(receipt_path)
                    if receipt["status"] != "complete":
                        raise RuntimeError(
                            f"Test for {name} was already attempted; inspect {receipt_path}. Automatic repeat is forbidden."
                        )
                    verify_inventory(receipt["files"])
                    evaluation_run = Path(receipt["run_dir"])
                else:
                    # Durable marker before any test decoding, even if the evaluator crashes.
                    atomic_json(
                        receipt_path,
                        {"status": "started", "freeze_sha256": file_hash(freeze_path)},
                    )
                    state["test_access"] = "started_after_validation_freeze"
                    atomic_json(group / "progress.json", state)
                    test_config = atomic_json(
                        group / "test_configs" / f"{name}.json", evaluation_config
                    )
                    before_test = set((Path(config.runs_dir) / "evaluate").glob("*"))
                    try:
                        evaluation_run = Path(run_evaluation(test_config)).resolve()
                        report = read_json(evaluation_run / "metrics" / "final.json")
                        _validate_test(
                            report, evaluation_config, split, expected_test_sources
                        )
                        if (
                            read_json(evaluation_run / "run.json")["status"]
                            != "complete"
                        ):
                            raise ValueError("test lifecycle incomplete")
                    except BaseException as error:
                        partial = (
                            set((Path(config.runs_dir) / "evaluate").glob("*"))
                            - before_test
                        )
                        for folder in partial:
                            lifecycle = folder / "run.json"
                            if lifecycle.exists():
                                atomic_json(
                                    lifecycle,
                                    {
                                        **read_json(lifecycle),
                                        "status": "failed",
                                        "error": str(error),
                                    },
                                )
                        atomic_json(
                            receipt_path,
                            {
                                "status": "failed",
                                "error": str(error),
                                "freeze_sha256": file_hash(freeze_path),
                                "partial_runs": [str(p.resolve()) for p in partial],
                            },
                        )
                        raise
                    atomic_json(
                        receipt_path,
                        {
                            "status": "complete",
                            "run_dir": str(evaluation_run),
                            "freeze_sha256": file_hash(freeze_path),
                            "files": inventory(evaluation_run),
                        },
                    )
                if read_json(receipt_path)["freeze_sha256"] != file_hash(freeze_path):
                    raise ValueError("test receipt belongs to another freeze")
                report = read_json(evaluation_run / "metrics" / "final.json")
                _validate_test(report, evaluation_config, split, expected_test_sources)
                reports.append(
                    {"model": name, "run_dir": str(evaluation_run), "report": report}
                )
                table.append(
                    _table_row(job, report, extended=exact, raw=raw_presentation)
                )
                state["tests"] = reports
                atomic_json(group / "progress.json", state)
            summary = {
                "validation_frozen": frozen,
                **(
                    {
                        "recipe_selection": str(
                            (group / "recipe_selection.json").resolve()
                        ),
                        "shared_recipe": recipe["selected"],
                        "training_counts": {
                            "search": experiment_offset - len(recipe["jobs"]),
                            "recipe_validation": len(recipe["jobs"]),
                            "comparison": len(jobs),
                            "total": experiment_offset + len(jobs),
                        },
                    }
                    if recipe
                    else {}
                ),
                "jobs": jobs,
                "tests": reports,
                "table": table,
                "best_custom_examples": next(
                    r["report"]["prediction_examples"]
                    for r in reports
                    if r["model"] == best_custom
                ),
                "timing_note": "Training includes epoch train/validation and checkpoint writes; metric passes include decoding/loading, not pure inference latency.",
                "size_note": "weight_tensor_bytes: state_dict tensors including buffers only; checkpoint_bytes: serialized checkpoint including optimizer/provenance.",
                "note": NOTE,
                "single_seed_evidence": True,
            }
            atomic_json(group / "comparison.json", summary)
            if exact:
                print(
                    "Final frozen comparison (all models; no test-based selection):",
                    flush=True,
                )
                for record in table:
                    if raw_presentation:
                        print(
                            f"{record['model']}: test accuracy={record['test_accuracy']:.4f}, "
                            f"loss={record['test_loss']:.4f}, parameters={record['num_params']:,}",
                            flush=True,
                        )
                        for item in record["test_per_class"]:
                            print(item, flush=True)
                        continue
                    print(
                        f"{record['model']}: test accuracy={record['test_accuracy']:.4f}, "
                        f"macro precision={record['test_macro_precision']:.4f}, "
                        f"recall={record['test_macro_recall']:.4f}, "
                        f"F1={record['test_macro_f1']:.4f}, parameters={record['num_params']:,}",
                        flush=True,
                    )
                print(
                    f"Validation-selected custom examples: {best_custom}; see comparison.json and predictions in the ZIP",
                    flush=True,
                )
            with (group / "comparison.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
            state["test_access"] = "evaluated_after_validation_freeze"
        atomic_json(group / "summary.json", summary)
        state["status"] = "complete"
        atomic_json(group / "progress.json", state)
        atomic_json(
            group / "run.json",
            {
                **read_json(group / "run.json"),
                "status": "complete",
                "test_access": state["test_access"],
                "completed_jobs": len(jobs),
                "last_invocation_seconds": perf_counter() - started,
            },
        )
        return group
    except BaseException as error:
        if group is not None and state is not None:
            state.update(status="incomplete", error=str(error))
            atomic_json(group / "progress.json", state)
            atomic_json(
                group / "run.json",
                {
                    **read_json(group / "run.json"),
                    "status": "incomplete",
                    "error": str(error),
                    "test_access": state["test_access"],
                },
            )
        raise
    finally:
        lock.unlink(missing_ok=True)


def _validate_test(report, expected_config, split, expected_sources):
    if (
        report["partition"] != "test"
        or report["split"]["manifest_hash"] != split["manifest_hash"]
    ):
        raise ValueError("test report changed partition/split")
    if (
        Path(report["checkpoint"]["path"]).resolve()
        != Path(expected_config["checkpoint_path"]).resolve()
    ):
        raise ValueError("test used a different checkpoint")
    if report["experiment_config"] != {
        k: v for k, v in expected_config.items() if k != "checkpoint_path"
    }:
        raise ValueError("test input/config differs from freeze")
    records = read_json(report["predictions"])
    if (
        len(records) != len(expected_sources)
        or {str(Path(r["source"]).resolve()) for r in records} != expected_sources
    ):
        raise ValueError("test predictions do not cover the full unique partition")
    if any(r["partition"] != "test" for r in records):
        raise ValueError("non-test prediction record")
    accuracy = statistics.mean(r["target"] == r["predicted"] for r in records)
    loss = statistics.mean(r["loss"] for r in records)
    for key in ("accuracy", "precision", "recall", "f1"):
        if not math.isclose(report["metrics"][key], accuracy, abs_tol=2e-6):
            raise ValueError("test micro metrics do not match full prediction records")
    if not math.isclose(report["metrics"]["loss"], loss, rel_tol=1e-5, abs_tol=2e-6):
        raise ValueError("test loss does not match full prediction records")
