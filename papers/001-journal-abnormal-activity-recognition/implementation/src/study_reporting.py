"""Full result tables shared by leaf experiments and the capacity search."""

import csv
import math
from pathlib import Path

from .study_matrix import atomic_json


def extended_metrics(predictions, targets, num_classes):
    """Macro uses every declared class; balanced accuracy uses supported classes."""
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


def result_row(result, **context):
    config = result.get("experiment_config", {})
    metrics = result.get("validation_metrics", {})
    layers = (
        config.get("convlstm_layers", [])
        if result.get("name", "").startswith("custom")
        else []
    )
    row = {
        **context,
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
    for key in (
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    ):
        label = "micro_" + key if key in {"precision", "recall", "f1"} else key
        row["validation_" + label] = metrics.get(key)
    for key in ("latency_ms_per_batch", "samples_per_second", "peak_cuda_memory_bytes"):
        row[key] = result.get("efficiency", {}).get(key)
    return row


def write_full_table(directory, rows, stem="results", print_rows=True):
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
    lines = [
        "# Full validation results",
        "",
        "All configurations are shown. Test stays locked; raw configuration rank is not architecture selection.",
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
    (directory / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if print_rows:
        print(
            f"\nFull results: {len(complete)} completed; {len(pending)} failed/pending",
            flush=True,
        )
        for r in ordered:
            print(
                f"{r.get('rank') or '-':>3} {r.get('stage', '')} {r.get('model', '')} "
                f"{r.get('status')} | {r.get('height')}px {r.get('frames')}f/{r.get('fps')}fps "
                f"seed={r.get('seed')} wd={r.get('weight_decay')} lr={r.get('learning_rate')} "
                f"val_acc={cell(r.get('validation_accuracy'))} val_loss={cell(r.get('validation_loss'))} "
                f"macro_f1={cell(r.get('validation_macro_f1'))} params={r.get('parameters')}",
                flush=True,
            )
    return ordered


def search_plots(directory, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    flat = [
        r for r in rows if r.get("stage") == "flat" and r.get("status") == "complete"
    ]
    if not flat:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for ax, size in zip(axes[0], sorted({r["height"] for r in flat})):
        for depth in (1, 2, 3):
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
