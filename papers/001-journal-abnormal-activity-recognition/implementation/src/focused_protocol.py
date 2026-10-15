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
FRAME_NOTE = (
    "Evidence-guided uneven depth sampling (5/6/13/4/4), one seed; best tested "
    "accuracy-size trade-off, not a global/depth-three optimum. Equal-condition "
    "means are not macro class averages. FPS changes coverage, not tensor size. "
    "Use matched training/validation histories; accuracy alone cannot prove "
    "overfitting. No subject/scene independence or runtime guarantee. "
    "New top-five selection; no reserved anchor. Comparison recipe approval "
    "and notebook integration remain separate (089)."
)
FAST_NOTE = (
    "Single-seed, evidence-guided 32px/128-epoch screening, not a global optimum "
    "or proof of convergence. 8/16 frames and 8/16 FPS vary temporal coverage. "
    "Only five shortlisted models receive 64px validation checks; all five advance "
    "to fresh 256-cap comparisons. Do not claim exhaustive 64px architecture search "
    "or source-group independence. Pooled P/R/F1 are not macro class averages."
)


def expanded(study):
    return study is not None and study.get("schema_version") in (6, 7)


def fast(study):
    return study is not None and study.get("schema_version") == 7


def input_conditions(study=None):
    if expanded(study):
        return [
            (n, s, f)
            for n in study["num_frames"]
            for s in study["frame_sizes"]
            for f in study["target_fps"]
        ]
    return [(8, s, f) for s, f in CONDITIONS]


def stage_name(condition, study=None):
    frames, size, fps = condition
    return (
        f"input_{frames}f_{size}px_{fps}fps"
        if expanded(study)
        else f"input_{size}px_{fps}fps"
    )


def candidates(study=None):
    from .capacity_config import candidate

    return [
        candidate(s) for s in (study["architectures"] if expanded(study) else STACKS)
    ]


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
        }
        | ({"num_frames", "backup"} if expanded(study) else set()),
        "focused search",
    )
    for key, expected in {
        "schema_version": 7 if fast(study) else (6 if expanded(study) else 5),
        "mode": "capacity_search",
        **({} if expanded(study) else {"architectures": STACKS}),
        "frame_sizes": [32] if fast(study) else ([64] if expanded(study) else [32, 64]),
        "target_fps": [8, 16],
        "seeds": [42],
        "max_runs": 128 if expanded(study) else 64,
        **({"num_frames": [8, 16]} if expanded(study) else {}),
    }.items():
        if value_hash(study[key]) != value_hash(expected):
            raise ValueError(
                f"{'087' if expanded(study) else '077'} requires {key}={expected}"
            )
    if expanded(study):
        from notebooks.utils.study_archive import backup_paths

        stacks = study["architectures"]
        # The JSON is the ordered manifest; constrain it to the approved controls.
        expected = (
            {tuple(s) for s in STACKS}
            | {(w,) * d for w in (32, 48, 64, 96) for d in range(1, 6)}
            | {(128,), (32, 64), (64, 32)}
        )
        if (
            len(stacks) != 32
            or any(
                not isinstance(s, list) or any(type(w) is not int for w in s)
                for s in stacks
            )
            or {tuple(s) for s in stacks} != expected
        ):
            raise ValueError(
                "087 requires exactly the approved 32 unique architectures"
            )
        backup_paths(study["backup"])
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
        "epochs": 128 if fast(study) else 256,
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
            "height": study["frame_sizes"][-1],
            "width": study["frame_sizes"][-1],
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
                "screening_id": (
                    "frame_depth_inputs_v1" if expanded(study) else "focused_inputs_v1"
                ),
                "candidates": candidates(study),
            }
        ),
    )


def blocks(config, study=None):
    from .capacity_config import job

    for condition in input_conditions(study):
        frames, size, fps = condition
        stage = stage_name(condition, study)
        rows = []
        for spec in candidates(study):
            row = job(
                "flat",
                spec["name"],
                config,
                spec,
                height=size,
                width=size,
                target_fps=fps,
                sequence_length=frames,
            )
            row.update(
                stage=stage,
                changed_factor=(
                    "architecture_by_frames_fps"
                    if expanded(study)
                    else "architecture_by_size_fps"
                ),
                matched_checkpoint_metrics=True,
            )
            rows.append(row)
        yield stage, rows


def decisions(jobs, study=None):
    """Equal condition weight; exact counts, not rounded console maxima."""
    if expanded(study) and any(j.get("status", "complete") != "complete" for j in jobs):
        raise ValueError("incomplete/failed frame-depth condition coverage")
    records = [j["result"] for j in jobs]
    conditions_in = input_conditions(study)
    expected = {
        (c["name"], n, s, f) for c in candidates(study) for n, s, f in conditions_in
    }
    indexed = {}
    for result in records:
        cfg = result["experiment_config"]
        key = (result["name"], cfg["sequence_length"], cfg["height"], cfg["target_fps"])
        if key not in expected or key in indexed:
            raise ValueError("focused search requires exactly one result per condition")
        if (result["partition"], result["test_access"], result["seed"]) != (
            "validation",
            "locked",
            42,
        ):
            raise ValueError("focused ranking requires locked validation evidence")
        indexed[key] = result
        if expanded(study):
            from .study_reporting import POOLED_REPORTING, reporting_protocol

            if (
                reporting_protocol(result) != POOLED_REPORTING
                or cfg["width"] != cfg["height"]
            ):
                raise ValueError(
                    "087 requires pooled evidence at the declared square input"
                )
    if set(indexed) != expected:
        raise ValueError("incomplete focused condition coverage")
    ranks = []
    exact = {}
    for spec in candidates(study):
        rows = [indexed[(spec["name"], n, s, f)] for n, s, f in conditions_in]
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
                        "sequence_length": n,
                        "sampling_span_seconds": (n - 1) / f,
                        "nominal_duration_seconds": n / f,
                        "accuracy": float(validation_accuracy(r, True)),
                        "loss": r["validation_metrics"]["loss"],
                    }
                    for (n, s, f), r in zip(conditions_in, rows)
                ],
            }
        )
    ranks.sort(key=lambda r: exact[r["name"]])
    ranks = [{"rank": i + 1, **r} for i, r in enumerate(ranks)]
    names = [r["name"] for r in ranks[:5]]
    reserved = not expanded(study) and ANCHOR not in names
    if reserved:
        names = names[:4] + [ANCHOR]
    conditions = []
    for n, s, f in conditions_in:
        rows = [indexed[(name, n, s, f)] for name in names]
        score = sum(validation_accuracy(r, True) for r in rows) / 5
        conditions.append(
            (
                score,
                statistics.mean(r["validation_metrics"]["loss"] for r in rows),
                s,
                f,
                n,
            )
        )
    score, loss, size, fps, frames = min(
        conditions, key=lambda c: (-c[0], c[1], c[2], c[4], -c[3])
    )
    return {
        "ranking": ranks,
        "numerical_top5": [r["name"] for r in ranks[:5]],
        "finalist_names": names,
        "anchor_reserved_slot": reserved,
        "anchor_rank": next(r["rank"] for r in ranks if r["name"] == ANCHOR),
        "final_input": {
            "sequence_length": frames,
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
        "note": FAST_NOTE if fast(study) else (FRAME_NOTE if expanded(study) else NOTE),
    }


def schedule(study, config, completed):
    from .capacity_search import job_key

    rows_all = []
    for stage, rows in blocks(config, study):
        rows_all.extend(rows)
        yield stage, rows
    yield "decisions", decisions([completed[job_key(r)] for r in rows_all], study)


def selection_bundle(study, state, group, decision):
    config = ExperimentConfig.from_mapping(state["effective_config"])
    specs = {c["name"]: c for c in candidates(study)}
    return {
        "protocol": (
            "fast_frame_depth_top5_v1"
            if fast(study)
            else ("frame_depth_top5_v1" if expanded(study) else "focused_top5_v1")
        ),
        **(
            {
                "reporting_protocol": "pooled_per_class_v2",
                "comparison_gate": "owner-approved recipe and transformer learnability checks required (089)",
                "input_tie_rule": "mean accuracy, mean loss, smaller size, fewer frames, higher FPS",
            }
            if expanded(study)
            else {}
        ),
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
    if expanded(study):
        print(
            f"{'089 fast' if fast(study) else '087'} frame/depth search: 32 architectures, depths 5/6/13/4/4; 128 jobs, fixed {config.height}x{config.width}; test locked."
        )
        print(
            f"8/16 frames x 8/16 FPS; seed 42; Adam LR=.001 WD=0; {config.epochs}-cap/64-minimum/24-patience; no augmentation."
        )
        for number, row in enumerate(
            (r for _, rows in blocks(config, study) for r in rows), 1
        ):
            c = row["config"]
            print(
                f"Experiment {number}/128 | {row['model']} | {c['sequence_length']}f/{c['height']}px/{c['target_fps']}fps | nominal {c['sequence_length']/c['target_fps']:g}s"
            )
        print(
            f"Results-only Drive backup: {study['backup']['directory']}/<study-id>/results.zip; authorize/mount before launch."
        )
        print(
            f"Full-step 16f/{config.height}px CUDA preflight on all 32 actual stacks precedes training. No overnight guarantee. The 089 notebook adds five validation-only 64px checks and ten fresh comparisons; no tests run by this search command."
        )
        print(FAST_NOTE if fast(study) else FRAME_NOTE)
        return
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


def frame_depth_summaries(directory, rows, study):
    """All measured points; controlled slices keep frame count/FPS distinct."""
    from .study_matrix import atomic_json
    from .study_reporting import write_full_table
    from .capacity_config import candidate
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    complete = [r for r in rows if r.get("status") == "complete"]
    index = {(r["model"], r["frames"], r["height"], r["fps"]): r for r in complete}
    if len(index) != len(complete):
        raise ValueError("Duplicate frame-count/FPS plot condition")
    specs = candidates(study)
    groups = []
    recipe = {
        "seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0,
        "augment": False,
        "scheduler": "reduce_on_plateau",
        "epochs_cap": 256,
        "minimum_epochs": 64,
        "patience": 24,
    }

    def add(label, changed, keys, controls):
        found = [index[k] for k in keys if k in index]
        groups.append(
            {
                "group": label,
                "changed_factor": changed,
                "fixed": {**recipe, **controls},
                "missing": [list(k) for k in keys if k not in index],
                "rows": found,
            }
        )

    for n, s, f in input_conditions(study):
        fixed = {
            "frames": n,
            "size": s,
            "fps": f,
            "seed": 42,
            "learning_rate": 0.001,
            "weight_decay": 0,
            "augment": False,
        }
        for width in (32, 48, 64, 96):
            add(
                f"depth_at_width{width}",
                "depth",
                [(candidate([width] * d)["name"], n, s, f) for d in range(1, 6)],
                {**fixed, "flat_width": width},
            )
        for depth in range(1, 6):
            flat = [
                c
                for c in specs
                if len(c["convlstm_layers"]) == depth
                and len({l[0] for l in c["convlstm_layers"]}) == 1
            ]
            add(
                f"width_at_depth{depth}",
                "flat_width",
                [(c["name"], n, s, f) for c in flat],
                {**fixed, "depth": depth},
            )
        for width in (32, 96, 128):
            stacks = [[width if i == pos else 64 for i in range(3)] for pos in range(3)]
            add(
                f"position_{width}_in_64x3",
                "width_placement",
                [(candidate(v)["name"], n, s, f) for v in stacks],
                {**fixed, "depth": 3, "filter_multiset": sorted(stacks[0])},
            )
        add(
            "two_layer_placement",
            "width_placement",
            [(candidate(v)["name"], n, s, f) for v in ([32, 64], [64, 32])],
            {**fixed, "depth": 2, "filter_multiset": [32, 64]},
        )
    for spec in specs:
        name = spec["name"]
        for n in study["num_frames"]:
            add(
                f"fps_{name}_{n}f",
                "fps_and_coverage",
                [(name, n, 64, f) for f in study["target_fps"]],
                {"model": name, "frames": n, "size": 64, "seed": 42},
            )
        for f in study["target_fps"]:
            add(
                f"frames_{name}_{f}fps",
                "frames_and_coverage",
                [(name, n, 64, f) for n in study["num_frames"]],
                {"model": name, "fps": f, "size": 64, "seed": 42},
            )
    atomic_json(
        directory / "matched_capacity_groups.json",
        {"note": FRAME_NOTE, "groups": groups},
    )
    write_full_table(
        directory,
        [
            {**r, "ablation_group": g["group"], "changed_factor": g["changed_factor"]}
            for g in groups
            for r in g["rows"]
        ],
        stem="matched_capacity",
        print_rows=False,
        raw=True,
        pooled_only=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    markers = ["o", "s", "^", "D", "P"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    panels = []
    for ax, (n, s, f) in zip(axes.flat, input_conditions(study)):
        measured = [
            r for r in complete if (r["frames"], r["height"], r["fps"]) == (n, s, f)
        ]
        for depth in range(1, 6):
            points = [r for r in measured if r["depth"] == depth]
            ax.scatter(
                [r["parameters"] for r in points],
                [100 * r["validation_accuracy"] for r in points],
                marker=markers[depth - 1],
                color=colors[depth - 1],
                alpha=0.75,
                edgecolors="black",
                linewidths=0.3,
            )
        # Only identify a few points; the complete table retains all identities.
        if measured:
            labels = {
                min(measured, key=lambda r: r["parameters"])["model"],
                max(measured, key=lambda r: r["parameters"])["model"],
                max(measured, key=lambda r: r["validation_accuracy"])["model"],
                "custom_48_48_48",
            }
            for r in measured:
                if r["model"] in labels:
                    ax.annotate(
                        r["filters"],
                        (r["parameters"], 100 * r["validation_accuracy"]),
                        xytext=(3, 5),
                        textcoords="offset points",
                        fontsize=7,
                    )
        ax.set_title(f"{n} frames / {f} FPS / {s}px ({len(measured)}/32 measured)")
        ax.set_xscale("log")
        ax.set_ylim(-2, 104)
        ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_ylabel("Validation accuracy (%)")
        ax.grid(alpha=0.2)
        panels.append(
            {
                "frames": n,
                "size": s,
                "fps": f,
                "points": [
                    {
                        "model": r["model"],
                        "depth": r["depth"],
                        "parameters": r["parameters"],
                        "validation_accuracy": r["validation_accuracy"],
                    }
                    for r in measured
                ],
            }
        )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=markers[d - 1],
                color=colors[d - 1],
                linestyle="",
                label=f"{d} layers",
            )
            for d in range(1, 6)
        ],
        loc="lower center",
        ncol=5,
    )
    fig.suptitle("Measured capacity: all frame-count/FPS conditions (validation only)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(directory / "frame_depth_capacity.png", dpi=160)
    plt.close(fig)
    atomic_json(
        directory / "frame_depth_plot_points.json",
        {"note": FRAME_NOTE, "measured_points": len(complete), "panels": panels},
    )
