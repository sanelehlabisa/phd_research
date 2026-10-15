"""Full result tables shared by leaf experiments and the capacity search."""

import csv
import math
import statistics
from pathlib import Path

from .study_matrix import atomic_json


POOLED_REPORTING = "pooled_per_class_v2"
LEGACY_REPORTING = "legacy_extended_v1"


def reporting_protocol(report):
    """Missing markers identify historical evidence, never a new pooled receipt."""
    protocol = report.get("reporting_protocol", LEGACY_REPORTING)
    if protocol not in (POOLED_REPORTING, LEGACY_REPORTING):
        raise ValueError("Unsupported reporting protocol")
    return protocol


def pooled_metrics(records, class_names):
    """Full-partition pooled scores and mean per-sample loss; no class averaging."""
    from .metrics import classification_metrics
    import torch

    if not records or len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("Nonempty predictions and unique class mapping required")
    for record in records:
        if any(
            type(record.get(key)) is not int or not 0 <= record[key] < len(class_names)
            for key in ("target", "predicted")
        ):
            raise ValueError("prediction or target outside class mapping")
        loss = record.get("loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError("Prediction loss must be finite and non-negative")
        for key, index in (("true_class", "target"), ("predicted_class", "predicted")):
            if key in record and record[key] != class_names[record[index]]:
                raise ValueError("Prediction class label differs from class mapping")
        if "correct" in record and record["correct"] != (
            record["target"] == record["predicted"]
        ):
            raise ValueError("Prediction correctness disagrees with class indices")
    return {
        "loss": statistics.mean(r["loss"] for r in records),
        **classification_metrics(
            torch.tensor([r["predicted"] for r in records]),
            torch.tensor([r["target"] for r in records]),
            len(class_names),
        ),
    }


def confusion_counts(records, class_names):
    matrix = [[0] * len(class_names) for _ in class_names]
    for record in records:
        matrix[record["target"]][record["predicted"]] += 1
    return dict(class_names=list(class_names), matrix=matrix, sample_count=len(records))


def validate_pooled_evidence(metrics, records, class_names, per_class, confusion):
    """Reconcile the versioned report, including saved confusion JSON when linked."""
    from .wide_protocol import class_counts
    from .study_matrix import read_json

    expected = pooled_metrics(records, class_names)
    if (
        not isinstance(metrics, dict)
        or set(metrics) != set(expected)
        or any(
            isinstance(metrics[k], bool)
            or not isinstance(metrics[k], (int, float))
            or not math.isfinite(metrics[k])
            or not math.isclose(metrics[k], v, rel_tol=1e-5, abs_tol=2e-6)
            for k, v in expected.items()
        )
    ):
        raise ValueError("Pooled metrics do not reconcile with complete predictions")
    if not isinstance(per_class, list) or any(
        not isinstance(row, dict)
        or any(type(row.get(k)) is not int for k in ("support", "tp", "fp", "fn"))
        or any(isinstance(row.get(k), bool) for k in ("precision", "recall", "f1"))
        for row in per_class
    ):
        raise ValueError("Invalid per-class metric/count fields")
    normalized = [
        {("class_name" if k == "class" else k): v for k, v in row.items()}
        for row in per_class
    ]
    if normalized != class_counts(records, class_names):
        raise ValueError("Per-class metrics/counts do not reconcile with predictions")
    counts = confusion_counts(records, class_names)
    if not isinstance(confusion, dict) or any(
        confusion.get(k) != v for k, v in counts.items()
    ):
        raise ValueError("Confusion counts do not reconcile with predictions")
    if confusion.get("json") and read_json(confusion["json"]) != counts:
        raise ValueError("Saved confusion counts disagree with report")


def matched_training_evaluation(
    model, loader, device, class_names, checkpoint, epoch, output, validation
):
    """Unaugmented training pass after loading the validation-selected checkpoint."""
    from time import perf_counter
    from .evaluate import prediction_records, per_class_metrics
    from .study_matrix import file_hash

    started = perf_counter()
    records = prediction_records(model, loader, device, class_names, partition="train")
    metrics = pooled_metrics(records, class_names)
    path = Path(output) / "selected_training_predictions.json"
    atomic_json(path, records)
    return {
        "protocol": "matched_checkpoint_eval_v2",
        "reporting_protocol": POOLED_REPORTING,
        "partition": "train",
        "model_mode": "eval",
        "augmentation": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "selected_epoch": epoch,
        "samples": len(records),
        "metrics": metrics,
        "predictions": str(path),
        "per_class": per_class_metrics(records, class_names),
        "confusion_matrix": confusion_counts(records, class_names),
        "seconds": perf_counter() - started,
        "train_minus_validation_accuracy": metrics["accuracy"] - validation["accuracy"],
        "validation_minus_train_loss": validation["loss"] - metrics["loss"],
        "diagnosis": "Not inferred automatically; inspect matched metrics and complete learning curves.",
    }


def extended_metrics(predictions, targets, num_classes):
    """Legacy-only verification: never use to produce new active run metrics.

    Historical macro uses all declared classes; balanced accuracy uses support.
    """
    import torch

    p = torch.as_tensor(predictions, dtype=torch.long)
    t = torch.as_tensor(targets, dtype=torch.long)
    if p.ndim != 1 or t.shape != p.shape or not len(t):
        raise ValueError("nonempty matching predictions/targets required")
    if ((p < 0) | (p >= num_classes) | (t < 0) | (t >= num_classes)).any():
        raise ValueError("prediction or target outside class mapping")
    cm = (
        torch.bincount(t * num_classes + p, minlength=num_classes**2)
        .reshape(num_classes, num_classes)
        .double()
    )
    tp = cm.diag()
    support, predicted = cm.sum(1), cm.sum(0)
    precision = tp / predicted.clamp(min=1)
    recall = tp / support.clamp(min=1)
    f1 = 2 * tp / (support + predicted).clamp(min=1)
    micro = float(tp.sum() / cm.sum())
    return {
        "micro_precision": micro,
        "micro_recall": micro,
        "micro_f1": micro,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
    }


def result_row(result, raw=False, pooled_only=None, **context):
    if pooled_only is None:
        pooled_only = reporting_protocol(result) == POOLED_REPORTING
    config = result.get("experiment_config", {})
    metrics = result.get("validation_metrics", {})
    layers = (
        config.get("convlstm_layers", [])
        if result.get("name", "").startswith("custom")
        else []
    )
    row = {
        **context,
        "reporting_protocol": reporting_protocol(result),
        "model": result.get("name"),
        "family": result.get("family"),
        "status": "complete",
        "filters": "-".join(str(x[0]) for x in layers),
        "depth": len(layers) if layers else None,
        "height": config.get("height"),
        "width": config.get("width"),
        "frames": config.get("sequence_length"),
        "fps": config.get("target_fps", 16),
        "weight_decay": config.get("weight_decay"),
        "learning_rate": config.get("learning_rate"),
        "seed": result.get("seed"),
        "parameters": result.get("num_params"),
        "total_parameters": result.get("total_parameters", result.get("num_params")),
        "weight_tensor_bytes": result.get("weight_tensor_bytes"),
        "checkpoint_bytes": result.get("checkpoint_bytes"),
        "train_seconds": result.get("train_time_s"),
        "actual_epochs": result.get("early_stopping", {}).get("actual_epochs"),
        "selected_epoch": result.get("checkpoint_selection", {}).get("selected_epoch"),
        "checkpoint": result.get("selected_checkpoint"),
    }
    validation_keys = (
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
    )
    if not pooled_only:
        validation_keys += (
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
        )
    for key in validation_keys:
        label = "micro_" + key if key in {"precision", "recall", "f1"} else key
        row["validation_" + label] = metrics.get(key)
    for key in ("latency_ms_per_batch", "samples_per_second", "peak_cuda_memory_bytes"):
        row[key] = result.get("efficiency", {}).get(key)
    training = result.get("training_evaluation", {})
    for key in (
        ("loss", "accuracy", "precision", "recall", "f1")
        if pooled_only
        else ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")
    ):
        label = "micro_" + key if key in {"precision", "recall", "f1"} else key
        row["selected_training_" + label] = training.get("metrics", {}).get(key)
    row["matched_accuracy_gap"] = training.get("train_minus_validation_accuracy")
    row["matched_loss_gap"] = training.get("validation_minus_train_loss")
    if raw or pooled_only:
        from .wide_protocol import raw_result_fields, unaveraged_row

        if result.get("validation_predictions"):
            row.update(raw_result_fields(result))
        else:
            row["validation_per_class"] = result.get("validation_per_class", [])
        row["validation_confusion_matrix"] = result.get("validation_confusion_matrix")
        if training.get("predictions"):
            row.update(raw_result_fields(result, "train", training["predictions"]))
        if not pooled_only:
            row = unaveraged_row(row)
    if pooled_only:
        row = {
            key: value
            for key, value in row.items()
            if "macro" not in key and "balanced_accuracy" not in key
        }
    return row


def write_full_table(
    directory, rows, stem="results", print_rows=True, raw=False, pooled_only=None
):
    if pooled_only is None:
        pooled_only = all(
            r.get("reporting_protocol") == POOLED_REPORTING
            for r in rows
            if r.get("status") == "complete"
        )
    if raw and not pooled_only:
        from .wide_protocol import unaveraged_row

        rows = [unaveraged_row(r) for r in rows]
    if pooled_only:
        rows = [
            {
                key: value
                for key, value in row.items()
                if "macro" not in key and "balanced_accuracy" not in key
            }
            for row in rows
        ]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    complete = [r for r in rows if r.get("status") == "complete"]
    pending = [r for r in rows if r.get("status") != "complete"]
    complete.sort(
        key=lambda r: (
            -(r.get("validation_accuracy") or 0),
            (
                r.get("validation_loss")
                if r.get("validation_loss") is not None
                else math.inf
            ),
            r.get("parameters") or 0,
            r.get("model") or "",
            str(r.get("job_id", "")),
        )
    )
    ranked = [{**r, "rank": i + 1} for i, r in enumerate(complete)]
    ordered = ranked + [{**r, "rank": None} for r in pending]
    atomic_json(
        directory / f"{stem}.json",
        {
            "rank_scope": "individual validation configurations; not architecture/top-three selection",
            "ranked": ranked,
            "incomplete": pending,
            "all": ordered,
        },
    )
    fields = list(dict.fromkeys(k for row in ordered for k in row)) or ["status"]
    with (directory / f"{stem}.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    columns = [
        "rank",
        "stage",
        "model",
        "status",
        "filters",
        "depth",
        "height",
        "frames",
        "fps",
        "weight_decay",
        "learning_rate",
        "seed",
        "validation_accuracy",
        "validation_micro_precision",
        "validation_micro_recall",
        "validation_micro_f1",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
        "validation_balanced_accuracy",
        "validation_loss",
        "parameters",
        "train_seconds",
    ]
    if raw:
        if not pooled_only:
            columns = [
                k
                for k in columns
                if "macro" not in k
                and "micro" not in k
                and "balanced_accuracy" not in k
            ]
        columns += [
            "selected_training_accuracy",
            "matched_accuracy_gap",
            "matched_loss_gap",
        ]
    if pooled_only:
        columns = [
            key
            for key in columns
            if "macro" not in key and "balanced_accuracy" not in key
        ]
    lines = [
        "# Full validation results",
        "",
        "All configurations are shown. Test stays locked; raw configuration rank is not architecture selection.",
        "Reporting: "
        + ", ".join(
            sorted({r.get("reporting_protocol", LEGACY_REPORTING) for r in complete})
        ),
        (
            "Pooled P/R/F1 equal accuracy for single-label multiclass classification; per-class counts retain class-specific evidence."
            if pooled_only
            else "Historical metrics retain their original definitions; no scores are relabelled."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    def cell(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "/").replace("\n", " ")

    lines.extend(
        "| " + " | ".join(cell(r.get(k)) for k in columns) + " |" for r in ordered
    )
    if raw or pooled_only:
        lines += [
            "",
            "## Unaveraged per-class validation metrics",
            "",
            "| Experiment | Model | Class | Precision | Recall | F1 | Support | TP | FP | FN |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        per_class = []
        for row in ordered:
            for c in row.get("validation_per_class", []):
                item = {
                    "experiment": row.get("experiment_number"),
                    "model": row["model"],
                    **c,
                }
                per_class.append(item)
                lines.append("| " + " | ".join(cell(v) for v in item.values()) + " |")
        if per_class:
            with (directory / f"{stem}_per_class.csv").open(
                "w", encoding="utf-8", newline=""
            ) as out:
                writer = csv.DictWriter(out, fieldnames=list(per_class[0]))
                writer.writeheader()
                writer.writerows(per_class)
    (directory / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if print_rows:
        print(
            f"\nFull results: {len(complete)} completed; {len(pending)} failed/pending",
            flush=True,
        )
        for r in ordered:
            scores = "" if raw else f"macro_f1={cell(r.get('validation_macro_f1'))} "
            if pooled_only:
                scores = (
                    f"precision={cell(r.get('validation_micro_precision'))} "
                    f"recall={cell(r.get('validation_micro_recall'))} "
                    f"f1={cell(r.get('validation_micro_f1'))} "
                )
            print(
                f"{r.get('rank') or '-':>3} {r.get('stage', '')} {r.get('model', '')} "
                f"{r.get('status')} | {r.get('height')}px {r.get('frames')}f/{r.get('fps')}fps "
                f"seed={r.get('seed')} wd={r.get('weight_decay')} lr={r.get('learning_rate')} "
                f"val_acc={cell(r.get('validation_accuracy'))} val_loss={cell(r.get('validation_loss'))} "
                + scores
                + f"params={r.get('parameters')}",
                flush=True,
            )
    return ordered


def search_plots(directory, rows, raw=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    flat = [
        r for r in rows if r.get("stage") == "flat" and r.get("status") == "complete"
    ]
    if not flat:
        return
    fig, axes = plt.subplots(
        1, len({r["height"] for r in flat}), figsize=(12, 4), squeeze=False
    )
    for ax, size in zip(axes[0], sorted({r["height"] for r in flat})):
        for depth in sorted({r["depth"] for r in flat}):
            group = sorted(
                [r for r in flat if r["height"] == size and r["depth"] == depth],
                key=lambda r: int(r["filters"].split("-")[0]),
            )
            ax.plot(
                [int(r["filters"].split("-")[0]) for r in group],
                [r["validation_accuracy"] for r in group],
                marker="o",
                label=f"{depth} layers",
            )
        ax.set(
            title=f"{size}x{size}",
            xlabel="Flat filters per layer",
            ylabel="Validation accuracy",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "width_depth_resolution.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    complete = [
        r
        for r in rows
        if r.get("status") == "complete"
        and r.get("stage") in {"flat", "references", "refinement"}
    ]
    for ax, field in zip(axes, ("parameters", "train_seconds")):
        for family in sorted({str(r.get("family")) for r in complete}):
            group = [r for r in complete if str(r.get("family")) == family]
            ax.scatter(
                [r[field] for r in group],
                [r["validation_accuracy"] for r in group],
                label=family,
                s=16,
            )
        ax.set(xlabel=field, ylabel="Validation accuracy")
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "accuracy_efficiency.png")
    plt.close(fig)
    # Pooled P/R/F1 equal accuracy for this single-label task; one score axis suffices.
    metrics = ("validation_accuracy",)
    fig, axes = plt.subplots(
        len(metrics), 3, figsize=(15, 4 * len(metrics)), squeeze=False
    )
    for axis_row, metric in zip(axes, metrics):
        for ax, cost in zip(
            axis_row, ("parameters", "train_seconds", "latency_ms_per_batch")
        ):
            for size in sorted({r["height"] for r in complete}):
                points = [
                    r
                    for r in complete
                    if r["height"] == size
                    and r.get(cost) is not None
                    and r.get(metric) is not None
                ]
                if points:
                    ax.scatter(
                        [r[cost] for r in points],
                        [r[metric] for r in points],
                        label=f"{size}px",
                        s=16,
                    )
            ax.set(xlabel=cost, ylabel=metric)
            if ax.collections:
                ax.legend()
    fig.suptitle(
        "Reference-recipe validation; configurations are not independent datasets"
    )
    fig.tight_layout()
    fig.savefig(directory / "accuracy_cost.png")
    plt.close(fig)


def matched_capacity_summaries(directory, rows):
    """Transparent matched groups, not automatic under/overfitting diagnoses."""
    from .capacity_config import candidate

    groups = {}
    for depth in (1, 2, 3, 4, 5):
        widths = {1: (32, 64, 128), 3: (8, 16, 32, 48, 64, 80, 96, 128), 5: (64,)}.get(
            depth, (32, 64, 96, 128)
        )
        groups[f"width_at_depth_{depth}"] = [[w] * depth for w in widths]
    for width in (32, 64, 96, 128):
        groups[f"depth_at_width_{width}"] = [
            [width] * depth
            for depth in (
                (1, 2, 3, 4, 5)
                if width == 64
                else (2, 3, 4) if width == 96 else (1, 2, 3, 4)
            )
        ]
    groups.update(
        position_widen_96=[[64, 64, 64], [96, 64, 64], [64, 96, 64], [64, 64, 96]],
        position_narrow_32=[[64, 64, 64], [32, 64, 64], [64, 32, 64], [64, 64, 32]],
        position_widen_128=[[64, 64, 64], [128, 64, 64], [64, 128, 64], [64, 64, 128]],
        two_layer_orientation=[[32, 32], [32, 64], [64, 32], [64, 64]],
    )
    matched = []
    index = {
        (r["model"], r["height"], r["seed"]): r
        for r in rows
        if r.get("stage") in {"flat", "refinement", "confirmation"}
        and r.get("status") == "complete"
        and r.get("weight_decay") == 0
        and r.get("frames") == 8
        and r.get("fps") == 16
    }
    for group, specs in groups.items():
        names = [candidate(s)["name"] for s in specs]
        for seed in (42, 2026):
            for size in (32, 64):
                found = [
                    index[(n, size, seed)] for n in names if (n, size, seed) in index
                ]
                if len({r["learning_rate"] for r in found}) > 1:
                    raise ValueError("Matched capacity group mixes learning rates")
                matched.append(
                    dict(
                        group=group,
                        seed=seed,
                        pixels=size,
                        complete=len(found) == len(names),
                        missing=[n for n in names if (n, size, seed) not in index],
                        rows=found,
                    )
                )
    note = (
        "Validation only. Compare within each size/seed/common recipe. Groups overlap; do not pool them. "
        "Width/depth changes also change parameter count. Gaps use the same selected checkpoint in eval mode. "
        "Missing evidence is explicit; no automatic underfitting/overfitting or optimum claim."
    )
    directory = Path(directory)
    atomic_json(directory / "matched_capacity.json", {"note": note, "groups": matched})
    columns = (
        "group",
        "pixels",
        "seed",
        "group_complete",
        "model",
        "parameters",
        "validation_accuracy",
        "selected_training_accuracy",
        "matched_accuracy_gap",
        "matched_loss_gap",
        "train_seconds",
        "latency_ms_per_batch",
        "peak_cuda_memory_bytes",
    )
    lines = [
        "# Matched width, depth and layer-position evidence",
        "",
        note,
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    table = []
    for group in matched:
        for row in group["rows"]:
            item = {
                **row,
                "group": group["group"],
                "pixels": group["pixels"],
                "group_complete": group["complete"],
            }
            table.append({k: item.get(k) for k in columns})
            lines.append(
                "| "
                + " | ".join(
                    str(item.get(k)) if item.get(k) is not None else "not measured"
                    for k in columns
                )
                + " |"
            )
    (directory / "matched_capacity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (directory / "matched_capacity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    return matched
