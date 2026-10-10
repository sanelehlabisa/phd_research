"""075-only final-input decisions and unaveraged presentation; no test selection."""

from collections import Counter
from pathlib import Path
import statistics

from .study_matrix import read_json, validation_accuracy


def class_counts(records, class_names):
    """Unaveraged class scores with confusion counts; zero division is zero."""
    output = []
    for index, name in enumerate(class_names):
        tp = sum(r["target"] == index and r["predicted"] == index for r in records)
        fp = sum(r["target"] != index and r["predicted"] == index for r in records)
        fn = sum(r["target"] == index and r["predicted"] != index for r in records)
        output.append(
            dict(
                class_name=name,
                precision=tp / (tp + fp) if tp + fp else 0,
                recall=tp / (tp + fn) if tp + fn else 0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                support=tp + fn,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )
    return output


def raw_result_fields(result, partition="validation", predictions=None):
    manifest = read_json(result["split"]["manifest_path"])
    records = read_json(predictions or result["validation_predictions"])
    classes = class_counts(records, manifest["class_names"])
    return {
        f"{partition}_per_class": classes,
        f"{partition}_prediction_counts": {
            str(k): v for k, v in Counter(r["predicted"] for r in records).items()
        },
    }


def unaveraged_row(row):
    return {
        k: v
        for k, v in row.items()
        if "macro" not in k and "micro" not in k and "balanced_accuracy" not in k
    }


def select_final_input(jobs, shortlist, batch_size):
    """Every model/FPS required; equal custom weights, exact count accuracy."""
    names = [c["name"] for c in shortlist]
    references = ["r3d_18", "swin3d_t"]
    expected = {(n, f) for n in names + references for f in (4, 8, 16)}
    actual = [
        (j["result"]["name"], j["result"]["experiment_config"]["target_fps"])
        for j in jobs
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Final-input FPS requires all fifteen declared validations")
    rows = []
    for job in jobs:
        r = job["result"]
        cfg = r["experiment_config"]
        if (
            cfg["sequence_length"],
            cfg["height"],
            cfg["width"],
            cfg["batch_size"],
            cfg["seed"],
            cfg["weight_decay"],
            cfg["epochs"],
            cfg["early_stopping_patience"],
        ) != (16, 64, 64, batch_size, 42, 0, 256, 24):
            raise ValueError("FPS evidence changed the fixed final-input recipe")
        records = read_json(r["validation_predictions"])
        manifest = read_json(r["split"]["manifest_path"])
        majority = Counter(
            s["class_index"] for s in manifest["samples"] if s["split"] == "train"
        )
        majority_class = min(majority, key=lambda c: (-majority[c], c))
        baseline = sum(p["target"] == majority_class for p in records) / len(records)
        counts = {
            str(k): v for k, v in Counter(p["predicted"] for p in records).items()
        }
        warnings = []
        if len(counts) == 1:
            warnings.append("single-class predictions")
        if float(validation_accuracy(r, True)) <= baseline:
            warnings.append("at/below training-majority-label validation baseline")
        rows.append(
            dict(
                model=r["name"],
                fps=cfg["target_fps"],
                experiment_number=job["experiment_number"],
                job_id=job["job_id"],
                validation_accuracy=float(validation_accuracy(r, True)),
                validation_loss=r["validation_metrics"]["loss"],
                training_accuracy=r["training_evaluation"]["metrics"]["accuracy"],
                training_loss=r["training_evaluation"]["metrics"]["loss"],
                prediction_counts=counts,
                majority_label=majority_class,
                majority_baseline=baseline,
                warnings=warnings,
                **raw_result_fields(r),
            )
        )
    scores = []
    for fps in (4, 8, 16):
        custom = [
            j["result"]
            for j in jobs
            if j["result"]["name"] in names
            and j["result"]["experiment_config"]["target_fps"] == fps
        ]
        scores.append(
            (
                -statistics.mean(validation_accuracy(r, True) for r in custom),
                statistics.mean(r["validation_metrics"]["loss"] for r in custom),
                -fps,
            )
        )
    fps = -min(scores)[2]
    top_one = next(
        row for row in rows if row["model"] == names[0] and row["fps"] == fps
    )
    return dict(
        sequence_length=16,
        height=64,
        width=64,
        target_fps=fps,
        batch_size=batch_size,
        selected_top1_job_id=top_one["job_id"],
        rows=rows,
        rule="Equal top-three custom exact validation accuracy, loss, then higher FPS.",
        note="Custom-tuned shared FPS, not per-family optimal tuning. Warnings suggest possible optimization failure, not proven capacity limits.",
    )


def print_final_input(report):
    print(
        "\nFinal-input FPS checks (validation only; matched selected checkpoint):",
        flush=True,
    )
    for row in report["rows"]:
        print(
            f"{row['model']} | {row['fps']} FPS | train/val accuracy="
            f"{row['training_accuracy']:.4f}/{row['validation_accuracy']:.4f} | "
            f"train/val loss={row['training_loss']:.4f}/{row['validation_loss']:.4f} | "
            f"prediction counts={row['prediction_counts']} | warnings={row['warnings']}",
            flush=True,
        )
    print(f"Selected common FPS: {report['target_fps']}. {report['note']}", flush=True)
