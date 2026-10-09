"""Guarded staged capacity search, driven exclusively by validation evidence."""

from pathlib import Path
import re
import math
import statistics
import sys

from .experiment_config import ExperimentConfig
from . import capacity_config as plan
from .study_matrix import (
    atomic_json,
    file_hash,
    inventory,
    read_json,
    value_hash,
    verify_inventory,
    _validate_leaf,
    CUSTOM_SLOTS,
)
from .study_reporting import (
    extended_metrics,
    result_row,
    write_full_table,
    search_plots,
)
from .utils import RunContext


def job_key(row):
    return value_hash(
        {
            "model": row["model"],
            "config": row["config"],
            "minimum_epochs": row["minimum_epochs"],
        }
    )[:16]


def schedule(study, config, completed):
    """Resolve the next stage only after prior evidence is complete.

    Used both to execute and to independently reconstruct a saved selection.
    Completed is a live job-key dictionary, updated by the runner after each yield.
    """

    def evidence(rows):
        return [completed[job_key(r)] for r in rows]

    calibration = plan.calibration_rows(study, config)
    yield "calibration", calibration
    lrs = plan.select_learning_rates(study, evidence(calibration))
    flat_specs = plan.flats(study)
    flat = plan.matrix_rows(study, config, flat_specs, lrs, "flat")
    yield "flat", flat
    references = plan.matrix_rows(study, config, plan.REFERENCES, lrs, "references")
    yield "references", references
    triplets = [c["name"] for c in flat_specs if len(c["convlstm_layers"]) == 3]
    flat_ranking = plan.architecture_ranking(
        evidence(flat), triplets, study["frame_sizes"], exact=plan.focused(study)
    )
    w = int(flat_ranking[0]["name"].split("_")[-1])
    refined_specs = (
        plan.shape_candidates()
        if plan.focused(study)
        else plan.refinement_candidates(w, flat_specs)
    )
    refinement = plan.matrix_rows(study, config, refined_specs, lrs, "refinement")
    yield "refinement", refinement
    custom_specs = {c["name"]: c for c in flat_specs + refined_specs}
    reference_jobs = evidence(flat + refinement)
    ranking = plan.architecture_ranking(
        reference_jobs,
        list(custom_specs),
        study["frame_sizes"],
        exact=plan.focused(study),
    )
    shortlist = [custom_specs[r["name"]] for r in ranking[:3]]
    confirming = plan.confirmation_specs(study, shortlist, w)
    confirmation = plan.matrix_rows(
        study, config, confirming, lrs, "confirmation", 2026
    )
    yield "confirmation", confirmation
    confirmed = plan.architecture_ranking(
        reference_jobs + evidence(confirmation),
        [c["name"] for c in shortlist],
        study["frame_sizes"],
        study["seeds"],
        exact=plan.focused(study),
    )
    shortlist = [custom_specs[r["name"]] for r in confirmed]
    wd = plan.ablation_rows(study, config, shortlist, lrs, "weight_decay")
    yield "weight_decay", wd
    temporal = plan.ablation_rows(study, config, shortlist, lrs, "temporal")
    yield "temporal", temporal
    reference_confirmation = {
        name: plan.architecture_ranking(
            evidence(references + confirmation),
            [name],
            study["frame_sizes"],
            study["seeds"],
            exact=plan.focused(study),
        )[0]
        for name in plan.REFERENCES
    }
    # Sensitivity evidence does not replace a member of the frozen shortlist.
    sensitivity = {}
    for spec in confirming:
        if isinstance(spec, dict) and spec["name"] not in {
            c["name"] for c in shortlist
        }:
            sensitivity[spec["name"]] = plan.architecture_ranking(
                reference_jobs + evidence(confirmation),
                [spec["name"]],
                study["frame_sizes"],
                study["seeds"],
                exact=plan.focused(study),
            )[0]
    yield "decisions", {
        "learning_rates": lrs,
        "three_layer_flat_ranking": flat_ranking,
        "flat_peak_width": w,
        "peak_at_search_boundary": w
        in (
            min(int(n.split("_")[-1]) for n in triplets),
            max(int(n.split("_")[-1]) for n in triplets),
        ),
        "preliminary_custom_ranking": ranking,
        "confirmed_custom_ranking": confirmed,
        "reference_confirmation": reference_confirmation,
        "peak_neighbour_sensitivity": sensitivity,
        "top3_candidates": shortlist,
        "ablation_note": "Single-seed one-factor evidence; do not silently combine winners.",
    }


def audit_split(source, split_path, review_path=None):
    """File hashes/source metadata only; test frames and predictions stay untouched."""
    manifest = read_json(split_path)
    by_hash, by_source = {}, {}
    files = []
    missing_sources = 0
    root = source.dataset_dir.resolve()
    for sample in manifest["samples"]:
        path = (root / sample["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                "capacity split must point to existing original video files"
            )
        digest = file_hash(path)
        record = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": digest,
            "partition": sample["split"],
        }
        files.append(record)
        by_hash.setdefault(digest, set()).add(sample["split"])
        source_id = sample.get("source_id") or sample.get("source_video_id")
        if source_id is None:
            missing_sources += 1
        else:
            by_source.setdefault(str(source_id), set()).add(sample["split"])
    collisions = {
        "cross_split_file_hashes": [
            k for k, splits in by_hash.items() if len(splits) > 1
        ],
        "cross_split_source_ids": [
            k for k, splits in by_source.items() if len(splits) > 1
        ],
    }
    audit = {
        **collisions,
        "files": files,
        "missing_source_ids": missing_sources,
        "source_independence_verified": missing_sources == 0
        and not collisions["cross_split_source_ids"]
        and not collisions["cross_split_file_hashes"],
        "note": "Exact file duplicates only; near-duplicates/unrecorded source groups remain unresolved. No test decoding.",
    }
    if review_path is not None:
        review = read_json(review_path)
        if (
            review.get("schema_version") != 1
            or review.get("inventory_hash") != manifest.get("inventory_hash")
            or review.get("split_content_sha256") != value_hash(manifest)
            or review.get("dataset_name") != manifest.get("dataset_name")
            or review.get("disposition") != "independent_recordings_user_confirmed"
        ):
            raise ValueError(
                "Source review does not cover this dataset/split; review before continuing"
            )
        suspected = {}
        for row in manifest["samples"]:
            stem = re.sub(r"_\d+$", "", Path(row["path"]).stem)
            suspected.setdefault(stem, set()).add(row["split"])
        audit.update(
            source_review=review,
            source_review_sha256=file_hash(review_path),
            filename_only_cross_split_groups=sorted(
                k for k, v in suspected.items() if len(v) > 1
            ),
            filename_overlap_disposition="user confirmed different recordings; suffixes are not source IDs",
            independence_limit="User-attested recording independence; subject/scene independence and near-duplicate absence are not established",
        )
    return audit


def _validate_result(result, row, split, sources, class_count):
    _validate_leaf(result, row, row["config"], split["manifest_hash"])
    metrics = result["validation_metrics"]
    required = (
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    )
    if any(
        not isinstance(metrics.get(k), (int, float)) or not math.isfinite(metrics[k])
        for k in required
    ):
        raise ValueError("missing/non-finite validation metrics")
    if metrics["loss"] < 0 or any(
        not 0 <= metrics[k] <= 1 for k in required if k != "loss"
    ):
        raise ValueError("validation metric outside its valid range")
    actual = result["early_stopping"]["actual_epochs"]
    if not row["minimum_epochs"] <= actual <= row["config"]["epochs"]:
        raise ValueError("leaf did not respect the common epoch budget")
    if not 1 <= result["checkpoint_selection"]["selected_epoch"] <= actual:
        raise ValueError("invalid selected epoch")
    if type(result["num_params"]) is not int or result["num_params"] <= 0:
        raise ValueError("invalid parameter count")
    predictions = read_json(result["validation_predictions"])
    paths = [str(Path(r["source"]).resolve()) for r in predictions]
    if set(paths) != set(sources) or len(paths) != len(sources):
        raise ValueError(
            "predictions do not cover the complete validation partition exactly once"
        )
    if any(
        r["partition"] != "validation" or r["target"] != sources[p]
        for r, p in zip(predictions, paths)
    ):
        raise ValueError("prediction partition/target differs from the split")
    checks = extended_metrics(
        [r["predicted"] for r in predictions],
        [r["target"] for r in predictions],
        class_count,
    )
    checks.update(
        accuracy=checks["micro_f1"],
        precision=checks["micro_f1"],
        recall=checks["micro_f1"],
        f1=checks["micro_f1"],
        loss=statistics.mean(r["loss"] for r in predictions),
    )
    if any(
        not math.isclose(metrics[k], v, abs_tol=1e-5, rel_tol=1e-5)
        for k, v in checks.items()
    ):
        raise ValueError("reported metrics do not reconcile with complete predictions")


def archive_stage(group, runs_dir):
    """Replace one verified ZIP at boundaries; retry never invokes training."""
    from notebooks.utils.study_archive import make_archive

    root = Path(runs_dir).resolve().parent
    if Path(runs_dir).resolve() != root / "runs":
        raise ValueError("staged ZIP delivery requires a runs directory named runs")
    archive = root / "runs" / "capacity_archives" / group.name / "artifacts.zip"
    return make_archive(root, archive, group / "progress.json", {group}, ())


def _report(group, state, print_rows=False, plots=False):
    jobs = {j["job_id"]: j for j in state["jobs"]}
    rows = []
    seen = set()
    for stage in state["stages"]:
        for spec in stage.get("rows", []):
            current = jobs.get(job_key(spec), {})
            context = {
                "stage": stage["name"],
                "job_id": job_key(spec),
                "reused_evidence": job_key(spec) in seen,
            }
            seen.add(job_key(spec))
            if current.get("status") == "complete":
                rows.append(result_row(current["result"], **context))
            else:
                cfg = spec["config"]
                rows.append(
                    {
                        **context,
                        "model": spec["model"],
                        "status": current.get("status", "pending"),
                        "error": current.get("error"),
                        "filters": "-".join(
                            str(x[0])
                            for x in (spec.get("candidate") or {}).get(
                                "convlstm_layers", []
                            )
                        ),
                        "depth": len(
                            (spec.get("candidate") or {}).get("convlstm_layers", [])
                        )
                        or None,
                        "height": cfg["height"],
                        "width": cfg["width"],
                        "frames": cfg["sequence_length"],
                        "fps": cfg["target_fps"],
                        "seed": cfg["seed"],
                        "weight_decay": cfg["weight_decay"],
                        "learning_rate": cfg["learning_rate"],
                    }
                )
    write_full_table(group, rows, print_rows=print_rows)
    if print_rows and state.get("pending_stages"):
        print(
            "Pending dependent stages (not yet resolved): "
            + ", ".join(state["pending_stages"]),
            flush=True,
        )
    if plots:
        search_plots(group, rows)
    return rows


def _ablation_recommendations(rows):
    """Export best tested single-factor settings, not an untested combined recipe."""
    report = {}
    for model in sorted({r["model"] for r in rows if r.get("status") == "complete"}):
        options = []
        wd = [
            r for r in rows if r["model"] == model and r.get("stage") == "weight_decay"
        ]
        temporal = [
            r for r in rows if r["model"] == model and r.get("stage") == "temporal"
        ]
        if not wd and not temporal:
            continue  # No ablation recommendation for an untested architecture.
        for value in sorted({r["weight_decay"] for r in wd}):
            block = [r for r in wd if r["weight_decay"] == value]
            options.append(
                {
                    "factor": "weight_decay",
                    "value": value,
                    "mean_validation_accuracy": statistics.mean(
                        r["validation_accuracy"] for r in block
                    ),
                    "mean_validation_loss": statistics.mean(
                        r["validation_loss"] for r in block
                    ),
                }
            )
        # Baseline WD=0 comes from the common-reference matrix; resolution averaged.
        baseline = [
            r
            for r in rows
            if r["model"] == model
            and r.get("stage") in {"flat", "refinement", "references"}
        ]
        if baseline:
            options.append(
                {
                    "factor": "weight_decay",
                    "value": 0,
                    "mean_validation_accuracy": statistics.mean(
                        r["validation_accuracy"] for r in baseline
                    ),
                    "mean_validation_loss": statistics.mean(
                        r["validation_loss"] for r in baseline
                    ),
                }
            )
        temporal += [r for r in baseline if r["height"] == 48]
        temporal_options = [
            {
                "frames": r["frames"],
                "fps": r["fps"],
                "validation_accuracy": r["validation_accuracy"],
                "validation_loss": r["validation_loss"],
            }
            for r in temporal
        ]
        report[model] = {
            "weight_decay_options": options,
            "weight_decay_recommendation": (
                min(
                    options,
                    key=lambda r: (
                        -r["mean_validation_accuracy"],
                        r["mean_validation_loss"],
                        r["value"],
                    ),
                )
                if wd
                else None
            ),
            "temporal_options": temporal_options,
            "temporal_recommendation": min(
                temporal_options,
                key=lambda r: (
                    -r["validation_accuracy"],
                    r["validation_loss"],
                    (r["frames"], r["fps"]) != (8, 16),
                    r["frames"],
                    -r["fps"],
                ),
            ),
        }
    return {
        "selection_partition": "validation",
        "seed": 42,
        "note": "Single-seed, separate one-factor settings. No combined winner is inferred.",
        "recommendation_rule": "accuracy then loss; WD ties prefer lower decay; temporal ties prefer the common reference, then fewer frames/higher FPS",
        "models": report,
    }


def selection_bundle(study, state, group, decisions):
    jobs = {j["job_id"]: j for j in state["jobs"]}
    specs = decisions["top3_candidates"]
    top3 = []
    config = ExperimentConfig.from_mapping(state["effective_config"])
    for i, spec in enumerate(specs):
        sources = []
        for seed in study["seeds"]:
            for row in plan.matrix_rows(
                study, config, [spec], decisions["learning_rates"], "confirmation", seed
            ):
                saved = jobs[job_key(row)]
                sources.append(
                    {
                        "run_dir": saved["run_dir"],
                        "receipt": saved["receipt"],
                        "receipt_sha256": file_hash(saved["receipt"]),
                        "config_sha256": value_hash(row["config"]),
                        "frame_size": row["config"]["height"],
                        "seed": seed,
                    }
                )
        top3.append(
            {"comparison_name": CUSTOM_SLOTS[i], "candidate": spec, "sources": sources}
        )
    return {
        "protocol": "capacity_top3_v2" if plan.focused(study) else "capacity_top3_v1",
        "selection_partition": "validation",
        "test_access": "locked",
        "seed": 42,
        "seeds": study["seeds"],
        "single_seed_evidence": False,
        "note": plan.protocol_note(study),
        "source_group": str(group.resolve()),
        "source_study_sha256": file_hash(group / "study.json"),
        "source_progress_sha256": file_hash(group / "progress.json"),
        "code_identity": state["code_identity"],
        "dataset_dir": config.dataset_dir,
        "split": state["split"],
        "ranking": decisions["confirmed_custom_ranking"],
        "ranking_rule": "equal mean over three resolutions and two seeds within predeclared shortlist; loss/parameters/name ties",
        **(
            {
                "ranking_version": "exact_counts_v1",
                "source_audit_sha256": file_hash(group / "split_audit.json"),
            }
            if plan.focused(study)
            else {}
        ),
        "top3": top3,
        "ablation_recommendations": str(group / "ablation_recommendations.json"),
        "ablation_recommendations_sha256": file_hash(
            group / "ablation_recommendations.json"
        ),
        "decisions_sha256": file_hash(group / "decisions.json"),
    }


def load_capacity_selection(path):
    """Rebuild every adaptive decision; no handwritten or partial winners."""
    selected = read_json(path)
    group = Path(selected["source_group"])
    state = read_json(group / "progress.json")
    if (
        state["status"] != "complete"
        or read_json(group / "run.json")["status"] != "complete"
    ):
        raise ValueError("capacity search is incomplete")
    verify_inventory(state["preparation_files"])
    source_audit = read_json(group / "split_audit.json")
    if (
        source_audit["cross_split_file_hashes"]
        or source_audit["cross_split_source_ids"]
    ):
        raise ValueError("selection has known cross-split duplication")
    verify_inventory({r["path"]: r["sha256"] for r in source_audit["files"]})
    study, _, _ = plan.load_capacity(read_json(group / "study.json"))
    if plan.focused(study):
        review = source_audit.get("source_review", {})
        if review.get(
            "disposition"
        ) != "independent_recordings_user_confirmed" or review.get(
            "split_content_sha256"
        ) != value_hash(
            read_json(group / "split_manifest.json")
        ):
            raise ValueError(
                "Missing compatible user source review; final testing forbidden"
            )
    config = ExperimentConfig.from_mapping(state["effective_config"])
    if file_hash(config.split_manifest) != state["split"]["manifest_hash"]:
        raise ValueError("selected split changed")
    manifest = read_json(config.split_manifest)
    sources = {
        str((Path(config.dataset_dir) / r["path"]).resolve()): r["class_index"]
        for r in manifest["samples"]
        if r["split"] == "validation"
    }
    jobs = {j["job_id"]: j for j in state["jobs"]}
    if len(jobs) != len(state["jobs"]) or any(
        j["status"] != "complete" for j in jobs.values()
    ):
        raise ValueError("duplicate/incomplete capacity jobs")
    visited, expected_stages = set(), []
    decisions = None
    for stage, rows in schedule(study, config, jobs):
        if stage == "decisions":
            decisions = rows
            break
        expected_stages.append({"name": stage, "status": "complete", "rows": rows})
        for row in rows:
            key = job_key(row)
            saved = jobs[key]
            receipt = read_json(saved["receipt"])
            if (
                receipt.get("status") != "complete"
                or receipt.get("result") != saved["result"]
            ):
                raise ValueError("capacity receipt/result mismatch")
            verify_inventory(receipt["files"])
            _validate_result(
                saved["result"],
                row,
                state["split"],
                sources,
                len(manifest["class_names"]),
            )
            visited.add(key)
    if set(jobs) != visited or state["stages"] != expected_stages:
        raise ValueError(
            "capacity stage matrix differs from approved adaptive protocol"
        )
    expected = selection_bundle(study, state, group, decisions)
    if selected != expected:
        raise ValueError("selected_config differs from verified two-seed evidence")
    return selected


def execute_capacity(path, study, config, manifest, run_trial):
    from .dataset import AHARDataset, load_split_subsets, resolve_split_manifest_path
    from .dataset_source import resolve_dataset
    from .study_config import ROOT
    from .study_resources import capacity_preflight

    root = Path(
        resolve_dataset(config.dataset_name, config.dataset_dir, ROOT)
    ).resolve()
    split_path = resolve_split_manifest_path(root, config.split_manifest, 42)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if not split_path.is_file():
        raise ValueError(
            "Ticket 069 requires the existing AAD split: finish/export 068 and supply its split_manifest. No new split was created."
        )
    source = AHARDataset(root, 8, (48, 48), sampling_version="timestamps_v1")
    train, validation, _, split = load_split_subsets(source, split_path)
    audit = audit_split(
        source,
        split_path,
        (
            ROOT / "configs/experiments/aad_source_review.json"
            if plan.focused(study)
            else None
        ),
    )
    source_identity = audit["files"]
    code = {p.name: file_hash(p) for p in sorted((ROOT / "src").glob("*.py"))}
    code["requirements.txt"] = file_hash(ROOT / "requirements.txt")
    if plan.focused(study):
        code["aad_source_review.json"] = audit["source_review_sha256"]
    code.update(
        {
            "notebooks/utils/" + name: file_hash(ROOT / "notebooks/utils" / name)
            for name in ("aad_study.py", "study_archive.py")
        }
    )
    identity = value_hash(
        {
            "study": study,
            "split": split["manifest_hash"],
            "data": source_identity,
            "code": code,
        }
    )
    receipts = Path(config.runs_dir) / "study_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    index = receipts / f"capacity_{value_hash(study)}.json"
    lock = index.with_suffix(".lock")
    try:
        handle = lock.open("x")
    except FileExistsError as error:
        raise RuntimeError(
            f"Study lock exists: {lock}; inspect active workers before recovery"
        ) from error
    with handle:
        handle.write(str(Path(path).resolve()))
    group, state = None, None
    try:
        if index.exists():
            group = Path(read_json(index)["run_dir"])
            saved = read_json(group / "progress.json")
            if saved["identity"] != identity:
                raise ValueError(
                    "resume code/data/split identity changed; no retraining or overwrite"
                )
            state = saved
            if state["status"] == "complete":
                load_capacity_selection(group / "selected_config.json")
                print(
                    f"Verified completed capacity study: {group}; no training/test repeated.",
                    flush=True,
                )
                return group
        else:
            context = RunContext(
                config.runs_dir,
                "studies",
                root,
                "capacity",
                {"study_file": str(Path(path).resolve()), "study": study},
                {"resume_index": str(index.resolve()), "test_access": "locked"},
            )
            group = context.run_dir.resolve()
            atomic_json(group / "study.json", study)
            (group / "split_manifest.json").write_bytes(split_path.read_bytes())
            atomic_json(group / "split_audit.json", audit)
            state = {
                "status": "preparing",
                "identity": identity,
                "code_identity": code,
                "test_access": "locked",
                "jobs": [],
                "stages": [],
                "pending_stages": list(plan.STAGE_LIMITS),
                "split": {**split, "manifest_path": str(group / "split_manifest.json")},
            }
            atomic_json(group / "progress.json", state)
            atomic_json(index, {"run_dir": str(group)})
        if audit["cross_split_file_hashes"] or audit["cross_split_source_ids"]:
            raise ValueError(
                "Known source/file duplication crosses splits; inspect split_audit.json. No automatic resplit."
            )
        if "effective_config" not in state:
            resource = (
                capacity_preflight(config, [*study["widths"], 64], max_depth=3)
                if plan.focused(study)
                else capacity_preflight(config, study["widths"])
            )
            atomic_json(group / "preflight.json", resource)
            if resource["selected_batch_size"] is None:
                raise RuntimeError(
                    "No common memory-safe batch; inspect preflight.json"
                )
            safe_config = ExperimentConfig.from_mapping(
                {
                    **config.to_dict(),
                    "dataset_dir": str(root),
                    "split_manifest": str(group / "split_manifest.json"),
                    "batch_size": resource["selected_batch_size"],
                }
            )
            timings = [
                r["training_step_seconds"]
                for r in resource["results"]
                if r.get("batch_size") == safe_config.batch_size
                and r.get("memory_safe")
            ]
            estimates = {
                "maximum_jobs": study["max_runs"],
                "train_samples": len(train),
                "batch_size": safe_config.batch_size,
                "estimated_training_only_upper_seconds": (
                    max(timings)
                    * math.ceil(len(train) / safe_config.batch_size)
                    * config.epochs
                    * study["max_runs"]
                    if timings
                    else None
                ),
                "note": "Conservative largest-input training-step extrapolation, not a wall-clock guarantee; excludes validation/decoding/ZIP. One selected checkpoint per job.",
            }
            atomic_json(group / "resource_estimates.json", estimates)
            state.update(effective_config=safe_config.to_dict(), status="running")
            state["preparation_files"] = inventory(group)
            state["preparation_files"].pop(
                str((group / "progress.json").resolve()), None
            )
            state["preparation_files"].pop(str((group / "run.json").resolve()), None)
            atomic_json(group / "progress.json", state)
        verify_inventory(state["preparation_files"])
        config = ExperimentConfig.from_mapping(state["effective_config"])
        sources = {
            str(source.samples[i][0].resolve()): source.samples[i][1]
            for i in validation.indices
        }
        completed = {j["job_id"]: j for j in state["jobs"] if j["status"] == "complete"}
        stages_seen = []
        decisions = None
        for stage, rows in schedule(study, config, completed):
            if stage == "decisions":
                decisions = rows
                break
            if len(rows) > plan.stage_limits(study)[stage]:
                raise ValueError("adaptive stage exceeded its declared bound")
            entry = next((s for s in state["stages"] if s["name"] == stage), None)
            if entry is not None and entry["rows"] != rows:
                raise ValueError(
                    "saved adaptive stage differs from complete prior evidence"
                )
            if entry is None:
                entry = {"name": stage, "status": "pending", "rows": rows}
                state["stages"].append(entry)
            stages_seen.append(stage)
            state["pending_stages"] = [
                s for s in plan.STAGE_LIMITS if s not in stages_seen
            ]
            atomic_json(group / "progress.json", state)
            _report(group, state)
            for row in rows:
                key = job_key(row)
                jobdir = group / "jobs" / key
                receipt_path = jobdir / "receipt.json"
                if receipt_path.exists():
                    receipt = read_json(receipt_path)
                    if receipt.get("status") != "complete":
                        raise RuntimeError(
                            f"Job was interrupted/failed: {receipt_path}. Evidence preserved; no automatic retry."
                        )
                    verify_inventory(receipt["files"])
                    result, leaf_run = receipt["result"], receipt["run_dir"]
                else:
                    if len(state["jobs"]) >= study["max_runs"]:
                        raise ValueError("maximum unique job budget reached")
                    leaf = atomic_json(jobdir / "config.json", row["config"])
                    args = [
                        "--config",
                        str(leaf),
                        "--model",
                        row["model"],
                        "--run-label",
                        "capacity",
                        "--trial-name",
                        key,
                        "--changed-factor",
                        row["changed_factor"],
                        "--minimum-epochs",
                        str(row["minimum_epochs"]),
                    ]
                    if row["candidate"]:
                        cp = atomic_json(
                            jobdir / "candidate.json",
                            {
                                "screening_id": "capacity_trial",
                                "candidates": [row["candidate"]],
                            },
                        )
                        args += ["--candidates-config", str(cp)]
                    atomic_json(
                        jobdir / "command.json",
                        [sys.executable, "-m", "src.experiments", *args],
                    )
                    atomic_json(receipt_path, {"status": "running", "job_id": key})
                    current = {
                        "job_id": key,
                        "status": "running",
                        "receipt": str(receipt_path),
                    }
                    state["jobs"].append(current)
                    atomic_json(group / "progress.json", state)
                    print(
                        f"Capacity {stage} [{len(state['jobs'])}/{study['max_runs']} max]: "
                        f"{row['model']} {row['config']['height']}px, "
                        f"{row['config']['sequence_length']}f/{row['config']['target_fps']}fps, seed {row['seed']}",
                        flush=True,
                    )
                    before = set((Path(config.runs_dir) / "experiments").glob("*"))
                    try:
                        leaf_run = str(Path(run_trial(args)).resolve())
                        payload = read_json(Path(leaf_run) / "summary.json")
                        if len(payload["all"]) != 1:
                            raise ValueError("one validation result required per job")
                        result = payload["all"][0]
                        _validate_result(
                            result, row, state["split"], sources, source.num_classes
                        )
                        if (
                            read_json(Path(leaf_run) / "run.json")["status"]
                            != "complete"
                        ):
                            raise ValueError("leaf lifecycle incomplete")
                        files = inventory(leaf_run)
                        files.update(
                            {
                                str(f.resolve()): file_hash(f)
                                for f in jobdir.glob("*.json")
                                if f != receipt_path
                            }
                        )
                        atomic_json(
                            receipt_path,
                            {
                                "status": "complete",
                                "run_dir": leaf_run,
                                "result": result,
                                "files": files,
                            },
                        )
                    except BaseException as error:
                        partial = (
                            set((Path(config.runs_dir) / "experiments").glob("*"))
                            - before
                        )
                        current.update(
                            status="failed",
                            error=str(error),
                            partial_runs=[str(p.resolve()) for p in partial],
                        )
                        atomic_json(receipt_path, current)
                        atomic_json(group / "progress.json", state)
                        raise
                _validate_result(
                    result, row, state["split"], sources, source.num_classes
                )
                saved_job = {
                    "job_id": key,
                    "status": "complete",
                    "receipt": str(receipt_path),
                    "run_dir": str(leaf_run),
                    "result": result,
                }
                state["jobs"] = [j for j in state["jobs"] if j["job_id"] != key] + [
                    saved_job
                ]
                completed[key] = saved_job
                atomic_json(group / "progress.json", state)
                _report(group, state)
            entry["status"] = "complete"
            atomic_json(group / "progress.json", state)
            _report(group, state, print_rows=True, plots=True)
            archive_stage(group, config.runs_dir)
        state["status"] = "complete"
        state["pending_stages"] = []
        atomic_json(group / "decisions.json", decisions)
        rows = _report(group, state, print_rows=True, plots=True)
        atomic_json(
            group / "ablation_recommendations.json", _ablation_recommendations(rows)
        )
        atomic_json(group / "progress.json", state)
        selected = selection_bundle(study, state, group, decisions)
        atomic_json(group / "selected_config.json", selected)
        atomic_json(group / "selection.json", selected)
        atomic_json(
            group / "summary.json",
            {**selected, "jobs": state["jobs"], "decisions": decisions},
        )
        atomic_json(
            group / "run.json",
            {
                **read_json(group / "run.json"),
                "status": "complete",
                "test_access": "locked",
                "completed_jobs": len(state["jobs"]),
            },
        )
        load_capacity_selection(group / "selected_config.json")
        archive_stage(group, config.runs_dir)
        print(
            f"Capacity search complete. Validation-only top three: {group / 'selected_config.json'}",
            flush=True,
        )
        return group
    except BaseException as error:
        if state is not None and group is not None:
            state.update(status="incomplete", error=str(error))
            atomic_json(group / "progress.json", state)
            atomic_json(
                group / "run.json",
                {
                    **read_json(group / "run.json"),
                    "status": "incomplete",
                    "error": str(error),
                    "test_access": "locked",
                },
            )
            _report(group, state, print_rows=True)
            try:
                archive_stage(group, config.runs_dir)
            except Exception as archive_error:
                print(
                    f"Evidence preserved at {group}; ZIP retry required: {archive_error}",
                    flush=True,
                )
        raise
    finally:
        lock.unlink(missing_ok=True)
