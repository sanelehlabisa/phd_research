"""077/076: balanced input search and a validation-frozen five-custom handoff.

Protocol decisions only; execution, receipts and test guards use existing runners.
"""

from copy import deepcopy
import statistics

from .experiment_config import CandidateManifest, ExperimentConfig
from .study_matrix import file_hash, value_hash, validation_accuracy

STACKS = [
    [96, 64, 64],
    [32, 64, 64],
    [64, 96, 64],
    [48, 48, 48],
    [64, 128, 64],
    [64, 64, 128],
    [128, 64, 64],
    [64, 64, 32],
    [64, 64, 64, 64, 64],
    [96, 96, 96],
    [16, 16, 16],
    [64],
    [64, 64],
    [64, 64, 64],
    [64, 64, 64, 64],
    [128, 128, 128],
]
SLOTS = [f"custom_top{i}" for i in range(1, 6)]
BASELINES = ["paper_convlstm_adapted", "r3d_18", "mc3_18", "swin3d_t", "swin3d_s"]
CONDITIONS = [(size, fps) for size in (32, 64) for fps in (8, 16)]
LIMITS = {f"input_{size}px_{fps}fps": 16 for size, fps in CONDITIONS}
ANCHOR = "custom_96_64_64"
NOTE = (
    "Evidence-guided single-seed search; input sensitivity is not seed uncertainty "
    "or source-independent generalisation. No guaranteed optimum/turning point. "
    "The historical anchor has a disclosed reserved slot, not a forced winning score. "
    "Shared input is custom-selected; fixed LR/WD are not independently tuned per family."
)


def candidates():
    from .capacity_config import candidate

    return [candidate(s) for s in STACKS]


def validation_diagnostics(result):
    from collections import Counter
    from .study_matrix import read_json

    records = read_json(result["validation_predictions"])
    manifest = read_json(result["split"]["manifest_path"])
    counts = Counter(r["predicted"] for r in records)
    train_counts = Counter(
        s["class_index"] for s in manifest["samples"] if s["split"] == "train"
    )
    majority = min(train_counts, key=lambda c: (-train_counts[c], c))
    baseline = sum(r["target"] == majority for r in records) / len(records)
    warnings = []
    if len(counts) == 1:
        warnings.append("single-class predictions")
    if float(validation_accuracy(result, True)) <= baseline:
        warnings.append("at/below training-majority-label validation baseline")
    return {
        "validation_warnings": warnings,
        "training_majority_label": majority,
        "validation_majority_baseline": baseline,
        "diagnostic_note": "Possible optimization limitation; not proof of a capacity boundary. No automatic exclusion/retraining.",
    }


def load_focused(study):
    from .study_config import _fields, _path

    study = deepcopy(study)
    _fields(
        study,
        {
            "schema_version",
            "mode",
            "dataset",
            "architectures",
            "frame_sizes",
            "target_fps",
            "seeds",
            "training",
            "runs_dir",
            "max_runs",
        },
        "focused search",
    )
    for key, expected in {
        "schema_version": 5,
        "mode": "capacity_search",
        "architectures": STACKS,
        "frame_sizes": [32, 64],
        "target_fps": [8, 16],
        "seeds": [42],
        "max_runs": 64,
    }.items():
        if value_hash(study[key]) != value_hash(expected):
            raise ValueError(f"077 requires {key}={expected}")
    _fields(study["dataset"], {"name", "path", "split_manifest"}, "dataset")
    if study["dataset"]["name"] != "aad":
        raise ValueError("077 requires AAD and its reviewed existing split")
    training = study["training"]
    _fields(
        training,
        {
            "epochs",
            "minimum_epochs",
            "early_stopping_patience",
            "batch_size",
            "num_workers",
            "pin_memory",
            "cache_dataset",
            "prediction_samples_per_category",
        },
        "focused training",
    )
    for key, expected in {
        "epochs": 256,
        "minimum_epochs": 64,
        "early_stopping_patience": 24,
        "cache_dataset": True,
        "prediction_samples_per_category": 0,
    }.items():
        if value_hash(training[key]) != value_hash(expected):
            raise ValueError(f"077 requires {key}={expected}")
    data = study["dataset"]
    config = ExperimentConfig.from_mapping(
        {
            **{k: v for k, v in training.items() if k != "minimum_epochs"},
            "dataset_name": "aad",
            "dataset_dir": _path(data["path"]),
            "split_manifest": (
                _path(data["split_manifest"]) if data["split_manifest"] else None
            ),
            "runs_dir": _path(study["runs_dir"]),
            "seed": 42,
            "sequence_length": 8,
            "height": 64,
            "width": 64,
            "learning_rate": 0.001,
            "weight_decay": 0,
            "scheduler": "reduce_on_plateau",
            "augment": False,
            "sampling_version": "timestamps_v1",
            "target_fps": 16,
        }
    )
    return (
        study,
        config,
        CandidateManifest.from_mapping(
            {
                "screening_id": "focused_inputs_v1",
                "candidates": candidates(),
            }
        ),
    )


def blocks(config):
    from .capacity_config import job

    for size, fps in CONDITIONS:
        stage = f"input_{size}px_{fps}fps"
        rows = []
        for spec in candidates():
            row = job(
                "flat",
                spec["name"],
                config,
                spec,
                height=size,
                width=size,
                target_fps=fps,
            )
            row.update(
                stage=stage,
                changed_factor="architecture_by_size_fps",
                matched_checkpoint_metrics=True,
            )
            rows.append(row)
        yield stage, rows


def decisions(jobs):
    """Equal condition weight; exact counts, not rounded console maxima."""
    records = [j["result"] for j in jobs]
    expected = {(c["name"], s, f) for c in candidates() for s, f in CONDITIONS}
    indexed = {}
    for result in records:
        cfg = result["experiment_config"]
        key = (result["name"], cfg["height"], cfg["target_fps"])
        if key not in expected or key in indexed:
            raise ValueError("focused search requires exactly one result per condition")
        if (result["partition"], result["test_access"], result["seed"]) != (
            "validation",
            "locked",
            42,
        ):
            raise ValueError("focused ranking requires locked validation evidence")
        indexed[key] = result
    if set(indexed) != expected:
        raise ValueError("incomplete focused condition coverage")
    ranks = []
    exact = {}
    for spec in candidates():
        rows = [indexed[(spec["name"], s, f)] for s, f in CONDITIONS]
        if len({r["num_params"] for r in rows}) != 1:
            raise ValueError("custom parameter count changed across conditions")
        scores = [validation_accuracy(r, True) for r in rows]
        mean, minimum = sum(scores) / 4, min(scores)
        loss = statistics.mean(r["validation_metrics"]["loss"] for r in rows)
        exact[spec["name"]] = (
            -mean,
            -minimum,
            loss,
            rows[0]["num_params"],
            spec["name"],
        )
        ranks.append(
            {
                "name": spec["name"],
                "mean_validation_accuracy": float(mean),
                "minimum_validation_accuracy": float(minimum),
                "accuracy_fraction": [mean.numerator, mean.denominator],
                "mean_validation_loss": loss,
                "parameters": rows[0]["num_params"],
                "conditions": [
                    {
                        "frame_size": s,
                        "target_fps": f,
                        "sequence_length": 8,
                        "sampling_span_seconds": 7 / f,
                        "nominal_duration_seconds": 8 / f,
                        "accuracy": float(validation_accuracy(r, True)),
                        "loss": r["validation_metrics"]["loss"],
                    }
                    for (s, f), r in zip(CONDITIONS, rows)
                ],
            }
        )
    ranks.sort(key=lambda r: exact[r["name"]])
    ranks = [{"rank": i + 1, **r} for i, r in enumerate(ranks)]
    names = [r["name"] for r in ranks[:5]]
    reserved = ANCHOR not in names
    if reserved:
        names = names[:4] + [ANCHOR]
    conditions = []
    for s, f in CONDITIONS:
        rows = [indexed[(name, s, f)] for name in names]
        score = sum(validation_accuracy(r, True) for r in rows) / 5
        conditions.append(
            (
                score,
                statistics.mean(r["validation_metrics"]["loss"] for r in rows),
                s,
                f,
            )
        )
    score, loss, size, fps = min(conditions, key=lambda c: (-c[0], c[1], c[2], -c[3]))
    return {
        "ranking": ranks,
        "numerical_top5": [r["name"] for r in ranks[:5]],
        "finalist_names": names,
        "anchor_reserved_slot": reserved,
        "anchor_rank": next(r["rank"] for r in ranks if r["name"] == ANCHOR),
        "final_input": {
            "sequence_length": 8,
            "height": size,
            "width": size,
            "target_fps": fps,
            "sampling_version": "timestamps_v1",
            "mean_validation_accuracy": float(score),
            "mean_validation_loss": loss,
        },
        "shared_recipe": {
            "learning_rate": 0.001,
            "weight_decay": 0,
            "scheduler": "reduce_on_plateau",
            "augment": False,
        },
        "note": NOTE,
    }


def schedule(study, config, completed):
    from .capacity_search import job_key

    rows_all = []
    for stage, rows in blocks(config):
        rows_all.extend(rows)
        yield stage, rows
    yield "decisions", decisions([completed[job_key(r)] for r in rows_all])


def selection_bundle(study, state, group, decision):
    config = ExperimentConfig.from_mapping(state["effective_config"])
    specs = {c["name"]: c for c in candidates()}
    return {
        "protocol": "focused_top5_v1",
        "selection_partition": "validation",
        "test_access": "locked",
        "seed": 42,
        "single_seed_evidence": True,
        "source_group": str(group.resolve()),
        "source_study_sha256": file_hash(group / "study.json"),
        "source_progress_sha256": file_hash(group / "progress.json"),
        "source_audit_sha256": file_hash(group / "split_audit.json"),
        "code_identity": state["code_identity"],
        "dataset_dir": config.dataset_dir,
        "split": state["split"],
        **decision,
        "ranking_rule": "equal-condition mean accuracy, minimum accuracy, mean loss, parameters, name",
        "top5": [
            {
                "comparison_name": SLOTS[i],
                "candidate": specs[name],
                "sources": [
                    {
                        "run_dir": j["run_dir"],
                        "receipt": j["receipt"],
                        "receipt_sha256": file_hash(j["receipt"]),
                        "config_sha256": value_hash(j["result"]["experiment_config"]),
                    }
                    for j in state["jobs"]
                    if j["result"]["name"] == name
                ],
            }
            for i, name in enumerate(decision["finalist_names"])
        ],
    }


def preflight(config, *, final=False, device=None, probe=None):
    """Bound both wide and deep customs; final phase includes every baseline."""
    import torch
    from .study_resources import training_probe

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and probe is None:
        return {
            "status": "cpu_only_gpu_safety_unverified",
            "selected_batch_size": config.batch_size,
            "results": [],
        }
    probe = probe or (lambda spec: training_probe(spec, config, 128, 3, device))
    trials = []
    for batch in (32, 16, 8, 4, 2, 1):
        if batch > config.batch_size:
            continue
        for name in ["custom_bound", "custom_depth5"] + (BASELINES if final else []):
            spec = dict(
                model=name,
                frames=8,
                size=config.height if final else 64,
                batch_size=batch,
                phase="comparison" if final else "search",
            )
            try:
                trial = {**spec, **probe(spec)}
            except torch.cuda.OutOfMemoryError:
                trial = {**spec, "memory_safe": False, "reason": "CUDA OOM"}
            trials.append(trial)
            if not trial["memory_safe"]:
                break
        else:
            return {
                "status": "complete",
                "selected_batch_size": batch,
                "results": trials,
                "note": "Full Adam steps; <=80% peak and free >=max(1 GiB, 10%). Synthetic timing, not a wall-clock guarantee.",
            }
    return {"status": "failed", "selected_batch_size": None, "results": trials}


def print_plan(study, config):
    print(
        "077 focused search: 64 jobs; 16 architectures x 32/64px x 8/16 FPS; test locked."
    )
    print(
        "8 frames; seed 42; Adam LR=0.001 WD=0; 256-cap/64-minimum/24-patience; no extra recipe/seed sweep."
    )
    for number, row in enumerate((r for _, rows in blocks(config) for r in rows), 1):
        c = row["config"]
        print(
            f"Experiment {number}/64 | {row['model']} | {c['height']}px | {c['target_fps']} FPS"
        )
    print(
        "Comparison adds ten fresh 512-cap trainings; caps are not runtime guarantees."
    )
    print(NOTE)


def matched_summaries(directory, rows):
    """Condition-separated evidence tables; no invented capacity diagnosis."""
    from .study_matrix import atomic_json
    from .study_reporting import write_full_table
    from .capacity_config import candidate

    groups = {
        "depth_at_width64": [[64] * d for d in range(1, 6)],
        "width_at_depth3": [[w] * 3 for w in (16, 48, 64, 96, 128)],
        "position_widen128": [
            [64, 64, 64],
            [128, 64, 64],
            [64, 128, 64],
            [64, 64, 128],
        ],
        "position_widen96": [[64, 64, 64], [96, 64, 64], [64, 96, 64], [64, 64, 96]],
        "position_narrow32": [[64, 64, 64], [32, 64, 64], [64, 32, 64], [64, 64, 32]],
    }
    index = {
        (r["model"], r["height"], r["fps"]): r
        for r in rows
        if r.get("status") == "complete"
    }
    records, summaries = [], []
    for group, stacks in groups.items():
        names = [candidate(s)["name"] for s in stacks]
        for size, fps in CONDITIONS:
            found = [index[(n, size, fps)] for n in names if (n, size, fps) in index]
            missing = [n for n in names if (n, size, fps) not in index]
            summaries.append(
                dict(
                    group=group,
                    frame_size=size,
                    target_fps=fps,
                    missing=missing,
                    rows=found,
                )
            )
            records.extend({**r, "ablation_group": group} for r in found)
    atomic_json(
        directory / "matched_capacity_groups.json", {"note": NOTE, "groups": summaries}
    )
    write_full_table(
        directory, records, stem="matched_capacity", print_rows=False, raw=True
    )
