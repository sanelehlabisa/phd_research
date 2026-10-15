"""Rebuild this review from preserved inputs and the imported 64-run evidence.

Full source audit (requires imported runs): python build_preview.py
Visible saved-results report (no runs needed):
    python build_preview.py --search-visibility --output-dir ../2026-10-15

Run with Python and matplotlib. No training, checkpoint loading or test access.
Outputs are derived report files only; source ZIP and console log are read-only.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
EXPECTED_HASHES = {
    "colab_console.raw.txt": "c7609a5ec25cf93399de2f951ad93cb878212c99db88e3356da98d187e2cf98a",
    "aad_results_essential.zip": "f4086085850f1bce64aa3abaf7a968cebc6166d4e5cd304afb6b955bc2081384",
}
CONDITIONS = [(32, 8), (32, 16), (64, 8), (64, 16)]
VISIBILITY_HASHES = {
    **EXPECTED_HASHES,
    "search_results.csv": "6424afd0d7d86943bffdd354c33eddc6e7c90776f7713ad77880b02973f7a78e",
    "architecture_ranking.csv": "3fe8a203f166d258086060d8e06d13f393e730a27bbf9bff979411f90066700d",
    "preview_results.json": "a4c4a93fb38fc6c9c01ac450a2b5270942ae7d89a19a01c085292ac876052d68",
    "search_evidence_check.json": "318536cbe2660efcc4060f735142ef5194b0b04b1e14a31d3ba47cd236dbd6a2",
}


def checked_metrics(classes):
    assert len(classes) == 11
    assert len({c["class_name"] for c in classes}) == 11
    n = sum(c["support"] for c in classes)
    correct = sum(c["tp"] for c in classes)
    assert n == 160
    assert sum(c["fp"] for c in classes) == n - correct
    for c in classes:
        tp, fp, fn = (c[k] for k in ("tp", "fp", "fn"))
        assert tp + fn == c["support"]
        expected = {
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        }
        for k, value in expected.items():
            assert math.isclose(c[k], value, abs_tol=1e-10)
    return {
        "accuracy": correct / n,
        "correct": correct,
        "support": n,
        **{f"macro_{k}": statistics.mean(c[k] for c in classes)
           for k in ("precision", "recall", "f1")},
    }


def parse_search(raw):
    header = re.compile(
        r"^Experiment (\d+)/74 campaign maximum \| input_(\d+)px_(\d+)fps: "
        r"(custom_[\d_]+) (\d+)px, (\d+)f/(\d+)fps, seed (\d+)", re.M
    )
    headers = list(header.finditer(raw))
    assert len(headers) == 64
    ranked = {}
    for match in re.finditer(
        r"^\s*\d+ input_(\d+)px_(\d+)fps (custom_[\d_]+) complete \| "
        r".*?val_acc=([\d.]+) val_loss=([\d.]+) params=(\d+)", raw, re.M
    ):
        size, fps, name, acc, loss, params = match.groups()
        key = (name, int(size), int(fps))
        value = (float(acc), float(loss), int(params))
        assert key not in ranked or ranked[key] == value
        ranked[key] = value
    assert len(ranked) == 64
    rows = []
    for i, h in enumerate(headers):
        block = raw[h.end():headers[i + 1].start() if i + 1 < 64 else raw.index("Search complete.")]
        job, size, fps, name, size2, frames, fps2, seed = h.groups()
        assert size == size2 and fps == fps2 and frames == "8" and seed == "42"
        classes = [ast.literal_eval(line) for line in block.splitlines()
                   if line.startswith("{'class_name':")]
        metrics = checked_metrics(classes)
        acc, loss, params = ranked[(name, int(size), int(fps))]
        assert math.isclose(metrics["accuracy"], acc, abs_tol=1e-8)
        printed = re.search(r"val_acc=([\d.]+) val_loss=([\d.]+) selected_epoch=(\d+) time=(\d+)s", block)
        assert printed
        assert abs(float(printed[1]) - acc) <= 0.000051
        assert abs(float(printed[2]) - loss) <= 0.000051
        assert int(re.search(r"params=([\d,]+)", block)[1].replace(",", "")) == params
        run = re.search(r"^Run directory: (.+)$", block, re.M)[1]
        widths = [int(v) for v in name.removeprefix("custom_").split("_")]
        rows.append({
            "experiment": int(job), "name": name, "architecture": "-".join(map(str, widths)),
            "layers": len(widths), "params": params, "frame_size": int(size),
            "fps": int(fps), "frames": 8, "seed": 42, "learning_rate": 0.001,
            "weight_decay": 0, **metrics, "loss": loss,
            "selected_epoch": int(printed[3]), "training_seconds_rounded": int(printed[4]),
            "source": "preserved console; loss rounded to printed ranking precision",
            "source_run": run, "per_class": classes,
        })
    assert [r["experiment"] for r in rows] == list(range(1, 65))
    assert len({(r["name"], r["frame_size"], r["fps"]) for r in rows}) == 64
    ranking = []
    for name in dict.fromkeys(r["name"] for r in rows):
        group = [r for r in rows if r["name"] == name]
        assert {(r["frame_size"], r["fps"]) for r in group} == set(CONDITIONS)
        assert len({r["params"] for r in group}) == 1
        ranking.append({
            "name": name, "architecture": group[0]["architecture"],
            "layers": group[0]["layers"], "params": group[0]["params"],
            "mean_accuracy": statistics.mean(r["accuracy"] for r in group),
            "worst_accuracy": min(r["accuracy"] for r in group),
            "mean_loss_console": statistics.mean(r["loss"] for r in group),
            **{f"accuracy_{r['frame_size']}px_{r['fps']}fps": r["accuracy"] for r in group},
        })
    ranking.sort(key=lambda r: (-r["mean_accuracy"], -r["worst_accuracy"], r["mean_loss_console"], r["params"], r["name"]))
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
    logged = ast.literal_eval(re.search(r"Search complete: numerical top five: (.+)", raw)[1])
    assert [r["name"] for r in ranking[:5]] == logged
    return rows, ranking


def parse_archive(search, ranking):
    with zipfile.ZipFile(ROOT / "aad_results_essential.zip") as z:
        assert z.testzip() is None
        assert len(z.namelist()) == 12
        config = json.loads(z.read(next(n for n in z.namelist() if n.endswith("resolved_model_comparison.json"))))
        summaries = {n: json.loads(z.read(n)) for n in z.namelist() if n.endswith("summary.json")}
    mapping = {c["name"]: "-".join(str(layer[0]) for layer in c["convlstm_layers"])
               for c in config["custom_candidates"]}
    assert list(mapping.values()) == [r["architecture"] for r in ranking[:5]]
    comparisons = []
    for member, summary in summaries.items():
        assert len(summary["all"]) == 1
        r = summary["all"][0]
        metrics = checked_metrics(r["validation_per_class"])
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            assert math.isclose(metrics[k], r["validation_metrics"][k], abs_tol=1e-7)
        cm = r["validation_confusion_matrix"]
        assert len(cm["matrix"]) == 11 and all(len(row) == 11 for row in cm["matrix"])
        assert cm["class_names"] == [c["class_name"] for c in r["validation_per_class"]]
        for i, c in enumerate(r["validation_per_class"]):
            assert sum(cm["matrix"][i]) == c["support"]
            assert cm["matrix"][i][i] == c["tp"]
            assert sum(row[i] for row in cm["matrix"]) - c["tp"] == c["fp"]
        assert r["test_access"] == "locked"
        assert r["seed"] == 42
        assert r["split"]["manifest_hash"] == "aada503d63d4433b272237db6cf27d7b1c0eaeb0091d5ba8289343a6111454d8"
        assert r["split"]["counts"] == {"train": 748, "validation": 160, "test": 161, "total": 1069}
        if "_capacity/" in member:
            last = search[-1]
            assert r["name"] == last["name"]
            assert abs(metrics["accuracy"] - last["accuracy"]) < 1e-8
            assert abs(r["validation_metrics"]["loss"] - last["loss"]) < 5e-6
            assert r["validation_per_class"] == last["per_class"]
            continue
        assert r["input_dimensions"] == {"sequence_length": 8, "height": 64, "width": 64, "channels": 3}
        e = r["efficiency"]
        comparisons.append({
            "name": r["name"], "architecture": mapping.get(r["name"], r["name"]),
            "family": r["family"], "params": r["num_params"], **metrics,
            "loss": r["validation_metrics"]["loss"],
            "train_accuracy": r["training_evaluation"]["metrics"]["accuracy"],
            "train_loss": r["training_evaluation"]["metrics"]["loss"],
            "selected_epoch": r["early_stopping"]["selected_epoch"],
            "actual_epochs": r["early_stopping"]["actual_epochs"],
            "training_seconds": r["train_time_s"],
            "weight_tensor_bytes": r["weight_tensor_bytes"], "checkpoint_bytes": r["checkpoint_bytes"],
            "latency_ms": e["latency_ms_per_batch"], "clips_per_second": e["samples_per_second"],
            "peak_cuda_bytes": e["peak_cuda_memory_bytes"],
            "selected_checkpoint": r["selected_checkpoint"],
            "selected_checkpoint_metadata": r["selected_checkpoint_metadata"],
            "history_reference": r["history"],
            "validation_predictions_reference": r["validation_predictions"],
            "checkpoint_sha256_reported": r["training_evaluation"]["checkpoint_sha256"],
            "checkpoint_dataset_label": r["checkpoint_selection"]["dataset_name"],
            "per_class": r["validation_per_class"], "validation_confusion_matrix": cm["matrix"],
            "class_names": cm["class_names"], "source_zip_member": member,
            "test_accuracy": None, "test_status": "not evaluated; first model failed compatibility guard",
        })
    comparisons.sort(key=lambda r: config["models"].index(r["name"]))
    assert [r["name"] for r in comparisons] == config["models"] and len(comparisons) == 10
    assert len(summaries) == 11
    return comparisons, config


def csv_output(name, rows):
    fields = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
    with (ROOT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_search_artifacts(search, ranking):
    """Match every console run to its imported summary, history and two images."""
    implementation = ROOT.parent.parent
    study = implementation / "runs/studies/20261014_055748_815241_abnormal-activities-dataset_capacity"
    hashes = {}

    def read_bytes(path):
        # Imported run names can exceed Windows' legacy 260-character limit.
        native = Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path
        return native.read_bytes()

    def read_json(path):
        data = read_bytes(path)
        hashes[path.relative_to(implementation).as_posix()] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    results = read_json(study / "results.json")["all"]
    assert len(results) == 64
    by_key = {(r["model"], r["height"], r["fps"]): r for r in results}
    assert len(by_key) == 64
    selection = read_json(study / "selection.json")
    expected = [r["name"] for r in ranking[:5]]
    assert selection["numerical_top5"] == selection["finalist_names"] == expected
    assert selection["selection_partition"] == "validation" and selection["test_access"] == "locked"
    gallery = ["# All 64 search runs: original images", "",
               "Every run has its original training curves and validation confusion matrix below.",
               "Expand a run to view both images. Report images are byte-identical copies of the original artifacts.",
               "These are validation-only runs; this is not a final test gallery.", "",
               "[Back to the overview](README.md#2-search-all-64-runs-grouped-into-four-panels)", ""]
    image_count = 0
    plotted_keys = []
    for row in search:
        key = (row["name"], row["frame_size"], row["fps"])
        result = by_key[key]
        assert result["status"] == "complete" and result["experiment_number"] == row["experiment"]
        assert result["parameters"] == row["params"]
        assert result["frames"] == row["frames"] and result["seed"] == row["seed"]
        assert abs(result["validation_accuracy"] - row["accuracy"]) < 1e-7
        assert abs(result["validation_loss"] - row["loss"]) < 5e-6
        relative = Path("runs") / row["source_run"].split("/runs/", 1)[1]
        run = (implementation / relative).resolve()
        assert run.is_relative_to((implementation / "runs/experiments").resolve())
        summary = read_json(run / "summary.json")
        assert len(summary["all"]) == 1
        record = summary["all"][0]
        assert record["name"] == row["name"] and record["seed"] == row["seed"]
        assert record["num_params"] == row["params"]
        assert record["validation_per_class"] == row["per_class"]
        assert record["split"]["manifest_hash"] == selection["split"]["manifest_hash"]
        assert record["input_dimensions"]["height"] == row["frame_size"]
        assert record["input_dimensions"]["sequence_length"] == row["frames"]
        metrics_dir = run / "models" / row["name"] / "metrics"
        history = read_json(metrics_dir / "history.json")
        assert history["selection"]["selected_epoch"] == row["selected_epoch"]
        epochs = history["epochs"]
        assert len(epochs) == record["early_stopping"]["actual_epochs"]
        selected = next(e for e in epochs if e["epoch"] == row["selected_epoch"])
        assert abs(selected["validation_metrics"]["loss"] - result["validation_loss"]) < 1e-7
        row["artifact_validation_loss"] = result["validation_loss"]
        row["matched_train_accuracy"] = result["selected_training_accuracy"]
        row["actual_epochs"] = record["early_stopping"]["actual_epochs"]
        row["source"] = "console cross-checked against imported study result, summary and history"
        row["local_summary"] = "../../" + relative.as_posix() + "/summary.json"
        if (row["experiment"] - 1) % 16 == 0:
            gallery += [f"## {row['frame_size']}px / {row['fps']} FPS (16 runs)", ""]
        gallery += ["<details>", f"<summary>Run {row['experiment']:02d}/64: {row['architecture']} | "
                    f"{row['params']:,} parameters | validation {100 * row['accuracy']:.3f}%</summary>", ""]
        for filename, title in [("training_curves.png", "Training curves"),
                                ("validation_confusion_matrix.png", "Validation confusion matrix")]:
            path = metrics_dir / filename
            data = read_bytes(path)
            assert data.startswith(b"\x89PNG\r\n\x1a\n"), path
            rel = path.relative_to(implementation).as_posix()
            hashes[rel] = hashlib.sha256(data).hexdigest()
            report_image = Path("search_images") / f"{row['experiment']:02d}" / filename
            target = ROOT / report_image
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == data, "Existing report image differs from original evidence"
            else:
                target.write_bytes(data)
            gallery += [f"![Run {row['experiment']}: {title}]({report_image.as_posix()})", ""]
            image_count += 1
        gallery += [f"Original summary (local run archive): `{relative.as_posix()}/summary.json`", "", "</details>", ""]
        plotted_keys.append(list(key))
    assert image_count == 128
    (ROOT / "search_run_gallery.md").write_text("\n".join(gallery), encoding="utf-8")
    audit = {"search_runs": 64, "architectures": 16, "conditions": 4, "runs_per_panel": 16,
             "summaries_checked": 64, "histories_checked": 64, "original_images_linked": image_count,
             "plot_keys": plotted_keys, "source_sha256": hashes,
             "scope": "Cross-checks search artifacts; does not verify checkpoint weights or test performance."}
    (ROOT / "search_evidence_check.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in audit.items() if k not in ("source_sha256", "plot_keys")}


def save_plot(fig, stem, output_dir=ROOT):
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    svg = output_dir / f"{stem}.svg"
    fig.savefig(svg, metadata={"Date": None})
    # Matplotlib emits trailing spaces in SVG path data; normalize only derived output.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8", newline="\n")


def accuracy_axis_forward(values):
    """Compress accuracy below 90 percent without removing or changing data."""
    values = np.asarray(values)
    return np.where(values <= 90, values * 0.05, 4.5 + values - 90)


def accuracy_axis_inverse(values):
    """Restore actual percentage labels for the compressed accuracy axis."""
    values = np.asarray(values)
    return np.where(values <= 4.5, values / 0.05, 90 + values - 4.5)


def search_plot(search, ranking, output_dir=ROOT, annotate_examples=False):
    """Plot every measured model, with optional role rings and a scaled y-axis."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "aad-2026-10-14"})
    colors = {1: "#0072B2", 2: "#009E73", 3: "#D55E00", 4: "#CC79A7", 5: "#6A5ACD"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}
    finalists = {r["name"] for r in ranking[:5]}
    fig, original_axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = list(original_axes.flat)
    plotted = []
    axis_limits = []
    annotations = []
    role_colors = {"best": "#006D2C", "worst": "#B2182B", "median": "#2166AC"}
    for ax, (size, fps) in zip(axes, CONDITIONS):
        group = [r for r in search if r["frame_size"] == size and r["fps"] == fps]
        assert len(group) == 16 and len({r["name"] for r in group}) == 16
        for r in group:
            depth = r["layers"]
            point = ax.scatter(r["params"], r["accuracy"] * 100, color=colors[depth], marker=markers[depth],
                               s=65, edgecolors="black" if not annotate_examples and r["name"] in finalists else "white",
                               linewidths=1.1, alpha=0.8, zorder=3 + depth if annotate_examples else 3)
            point.set_gid(f"search-run-{r['experiment']:02d}")
            plotted.append(r["experiment"])
        # Keep the original review labels when rebuilding the historical figures.
        labels = [
            ("custom_16_16_16", "Smallest: 16-16-16", (0.03, 0.43)),
            ("custom_128_128_128", "Largest: 128-128-128", (0.70, 0.61)),
            ("custom_48_48_48", "Compact: 48-48-48", (0.15, 0.68)),
            ("custom_64_96_64", "Mean-rank #1: 64-96-64", (0.43, 0.51)),
            ("custom_96_64_64", "Peak-score arch: 96-64-64", (0.59, 0.83)),
        ]
        if (size, fps) == (64, 16):
            labels.append(("custom_64_64_64_64_64", "Deep: 64 x 5", (0.65, 0.22)))
        for name, title, location in ([] if annotate_examples else labels):
            r = next(r for r in group if r["name"] == name)
            label = f"{title}\n{100 * r['accuracy']:.3f}%"
            ax.annotate(label, (r["params"], r["accuracy"] * 100), xytext=location,
                        textcoords="axes fraction", fontsize=9, va="center",
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.93},
                        arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.8})
        ax.set_title(f"{size} x {size} pixels | {fps} FPS | 16/16 runs", fontsize=12)
        ax.set_xscale("log")
        x_limits = (min(r["params"] for r in search) * 0.72,
                    max(r["params"] for r in search) * 1.52) if annotate_examples else (35000, 4500000)
        ax.set_xlim(*x_limits)
        if annotate_examples:
            ax.set_yscale("function", functions=(accuracy_axis_forward, accuracy_axis_inverse))
            ax.set_ylim(0, 100.5)
            ax.set_yticks([0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 100])
            ax.axhline(90, color="0.5", linestyle="--", linewidth=0.8)
            ax.text(0.98, 90.3, "compressed below 90%", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color="0.35")
        else:
            ax.set_ylim(0, 104)
        axis_limits.append({"frame_size": size, "fps": fps, "x": list(x_limits),
                            "y": [0, 100.5] if annotate_examples else [0, 104]})
        ax.set_xticks([50000, 100000, 500000, 1000000, 3000000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:g}M" if x >= 1e6 else f"{x/1000:g}k"))
        ax.grid(alpha=0.2)
        if annotate_examples:
            # Highlight actual observations without changing their layer markers.
            middle_score = statistics.median(r["accuracy"] for r in group)
            examples = {
                "best": min(group, key=lambda r: (-r["accuracy"], r["params"], r["experiment"])),
                "worst": min(group, key=lambda r: (r["accuracy"], r["params"], r["experiment"])),
                "median": min(group, key=lambda r: (round(abs(r["accuracy"] - middle_score), 12),
                                                    r["params"], r["experiment"])),
            }
            panel_handles = []
            for role in ("best", "median", "worst"):
                row = examples[role]
                title = "Middle" if role == "median" else role.capitalize()
                ring = ax.scatter(row["params"], 100 * row["accuracy"], s=195,
                                  facecolors="none", edgecolors=role_colors[role],
                                  linewidths=1.7, zorder=12)
                ring.set_gid(f"example-{role}-{row['experiment']:02d}")
                panel_handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                                            markeredgecolor=role_colors[role], markeredgewidth=1.7,
                                            markersize=9,
                                            label=f"{title}: {row['architecture']} ({100 * row['accuracy']:.2f}%)"))
                annotations.append({"frame_size": size, "fps": fps, "role": role,
                                    "experiment": row["experiment"], "architecture": row["architecture"],
                                    "accuracy": row["accuracy"]})

            ax.legend(handles=panel_handles, loc="lower left", bbox_to_anchor=(0.02, 0.35),
                      fontsize=9, framealpha=0.94, edgecolor="0.85", title="Highlighted models",
                      title_fontsize=9)
    for ax in axes[-2:]:
        ax.set_xlabel("Trainable parameters (log scale)")
    for ax in axes[::2]:
        ax.set_ylabel("Validation accuracy (%)")
    handles = [Line2D([], [], color=colors[d], marker=markers[d], linestyle="", label=f"{d} layer{'s' if d > 1 else ''}") for d in range(1, 6)]
    assert sorted(plotted) == list(range(1, 65)), "Every search run must appear exactly once."
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    subtitle = "lower y-range compressed; all measurements retained" if annotate_examples else "black outlines = five selected architectures"
    fig.suptitle(f"64/64 search runs | 8 frames, seed 42 | {subtitle}", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_plot(fig, "search_capacity", output_dir)
    plt.close(fig)
    return {"plotted_experiments": plotted, "axis_limits": axis_limits,
            "annotations": annotations, "zoom_panels": [],
            "highlight_style": "ring_legend" if annotate_examples else "historical_labels",
            "y_axis": {"scale": "piecewise_linear", "threshold": 90, "lower_compression": 0.05,
                       "note": "Below 90% uses one twentieth of the vertical distance per percentage point; no values removed."}
            if annotate_examples else {"scale": "linear"}}


def plots(search, ranking, comparison):
    search_plot(search, ranking)

    # Two comparison panels expose accuracy and parameter counts separately.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = [r["architecture"] for r in comparison]
    labels = [s.replace("paper_convlstm_adapted", "Paper ConvLSTM (adapted)").replace("swin3d_", "Swin3D-") for s in labels]
    palette = ["#D55E00"] * 5 + ["#7C5A00"] + ["#0072B2"] * 2 + ["#6A5ACD"] * 2
    y = list(range(10))
    axes[0].barh(y, [100 * r["accuracy"] for r in comparison], color=palette)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0, 114)
    axes[0].set_xlabel("Validation accuracy (%) | 160 clips")
    axes[1].barh(y, [r["params"] / 1e6 for r in comparison], color=palette)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.7, 280)
    axes[1].set_yticks(y, [""] * 10)
    axes[1].set_xlabel("Trainable parameters (millions, log scale)")
    for i, r in enumerate(comparison):
        axes[0].text(100 * r["accuracy"] + 1, i, f"{100 * r['accuracy']:.3f}", va="center", fontsize=9)
        axes[1].text(r["params"] / 1e6 * 1.07, i, f"{r['params']/1e6:.3f}M", va="center", fontsize=9)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Fresh ten-model comparison | validation only; test evaluation failed")
    fig.tight_layout()
    save_plot(fig, "comparison_validation")
    plt.close(fig)


def visibility_inputs():
    """Check preserved report inputs without requiring the lost Colab run tree."""
    hashes = {}
    for name, expected in VISIBILITY_HASHES.items():
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if hashes[name] != expected:
            raise ValueError(f"Preserved evidence changed: {name}")
    payload = json.loads((ROOT / "preview_results.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "search_evidence_check.json").read_text(encoding="utf-8"))
    rows, ranking = payload["search"], payload["architecture_ranking"]
    raw = (ROOT / "colab_console.raw.txt").read_text(encoding="utf-8-sig")
    console, console_ranking = parse_search(raw)
    _, config = parse_archive(console, console_ranking)
    assert ranking == console_ranking
    assert payload["resolved_comparison_config"] == config
    assert [r["experiment"] for r in rows] == list(range(1, 65))
    assert previous["plot_keys"] == [[r["name"], r["frame_size"], r["fps"]] for r in rows]
    for row, printed in zip(rows, console, strict=True):
        for key, value in printed.items():
            if key != "source":
                assert row[key] == value, (row["experiment"], key)

    # The CSV and portable JSON must describe the exact same scalar values.
    for name, expected_rows in [("search_results.csv", rows), ("architecture_ranking.csv", ranking)]:
        with (ROOT / name).open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        expected = [{k: "" if v is None else str(v) for k, v in row.items()
                     if not isinstance(v, (dict, list))} for row in expected_rows]
        assert saved == expected, name

    for row in rows:
        source_run = "runs/" + row["source_run"].split("/runs/", 1)[1]
        for filename in ("training_curves.png", "validation_confusion_matrix.png"):
            relative = f"search_images/{row['experiment']:02d}/{filename}"
            image = (ROOT / relative).read_bytes()
            assert image.startswith(b"\x89PNG\r\n\x1a\n"), relative
            digest = hashlib.sha256(image).hexdigest()
            source_image = f"{source_run}/models/{row['name']}/metrics/{filename}"
            assert digest == previous["source_sha256"][source_image], relative
            hashes[relative] = digest

    # The old receipt is historical evidence, not proof the originals exist here.
    missing = []
    checked = 0
    implementation = ROOT.parent.parent
    for relative, expected_hash in previous["source_sha256"].items():
        source = (implementation / relative).resolve()
        assert source.is_relative_to((implementation / "runs").resolve()), relative
        if source.is_file():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash, relative
            checked += 1
        else:
            missing.append(relative)
    audit = {
        "source_mode": "preserved_report", "source_sha256": hashes,
        "original_source_files_checked": checked,
        "original_source_files_unavailable": missing,
        "historical_selection_sha256": {k: v for k, v in previous["source_sha256"].items()
                                        if k.endswith("/selection.json")},
        "scope": "Reconciles saved report values, raw console, archived comparison selection and image copies. "
                 "Missing original summaries/histories/selection cannot be freshly verified; "
                 "available originals are hash-checked only. No checkpoint loading or test access.",
    }
    return rows, ranking, audit


def build_search_visibility(output_dir: Path) -> dict:
    """Render all 64 saved configurations in a separate report without training."""
    output_dir = Path(output_dir).resolve()
    source_dir = ROOT.resolve()
    if output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
        raise ValueError("Output must be separate from the original report and its parents.")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Output must be a directory; existing file was preserved.")
    files = {"README.md", "search_run_gallery.md", "search_capacity.png", "search_capacity.svg"}
    if output_dir.exists() and any(output_dir.iterdir()):
        receipt_path = output_dir / "visibility_check.json"
        if not receipt_path.is_file():
            raise ValueError("Output contains unrelated files; existing work was preserved.")
        saved = json.loads(receipt_path.read_text(encoding="utf-8"))
        if saved.get("report_type") != "search_visibility_v1" or set(saved.get("output_sha256", {})) != files:
            raise ValueError("Output is not a verified visibility report.")
        if {p.name for p in output_dir.iterdir()} != files | {"visibility_check.json"}:
            raise ValueError("Output contains unrelated files; existing work was preserved.")
        for name, digest in saved["output_sha256"].items():
            if hashlib.sha256((output_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Output was edited; existing work was preserved: {name}")
    try:
        rows, ranking, audit = visibility_inputs()
    except (AssertionError, KeyError, OSError, IndexError, TypeError, zipfile.BadZipFile) as error:
        raise ValueError(f"Preserved search evidence failed verification: {error}") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    source_link = Path(os.path.relpath(ROOT, output_dir)).as_posix()
    lines = [
        "# AAD search: all 64 configurations", "",
        "Review: 15 October 2026. Search executed 14 October 2026 (timezone unverified).",
        f"Reformatted from the unchanged [original review]({source_link}/README.md).",
        "**Validation-only, single-seed results. No training or test evaluation was run.**", "",
        "16 architectures x two resolutions x two sampling rates = **64 configurations**, not 64 architectures.",
        "Each panel plots all 16 measured models; the failed-learning result is retained.", "",
        "![All 64 search measurements, with role highlights and a compressed lower accuracy range](search_capacity.png)", "",
        "[Vector figure](search_capacity.svg). The bottom legend shows only the five original layer-count colours/shapes.",
        "Each panel's small legend names the highlighted models in order: Best, Middle, Worst, with accuracy to two decimals.",
        "No arrows, inset zooms or performance tables.", "",
        "## Reading the graphs", "",
        "Coloured rings highlight the Best, Middle and Worst **observed model within each panel**.",
        "This keeps different FPS/resolution conditions separate. Ties choose fewer parameters, then run ID;",
        "these illustrative highlights do not change the across-condition ranking or claim a global optimum.",
        "Middle means the actual model closest to that panel's median accuracy, not a fabricated average point.", "",
        "Every marker keeps its measured coordinates; all 64 unique observations remain, including the 12.5% failure.",
        "The **y-axis is not uniformly spaced**: below 90%, a percentage point uses one twentieth of the",
        "vertical distance used above 90%. The dashed 90% line marks the transition; tick labels show true accuracy.",
        "This stretches the high-score region without dropping low values or adding extra zoom plots.",
        "The same scale is used in all four panels; nearby models can still overlap. Rings repeat selected",
        "markers as highlights, not new data. Raw scores and the original uniformly scaled figure remain in the prior review.", "",
        "## Evidence and limits", "",
        "The saved CSV/portable JSON reconcile with the raw console and archived finalist configuration.",
        "All 128 report image copies match the hashes in the earlier verification receipt.",
        f"Original run files available/hash-checked here: {audit['original_source_files_checked']}; "
        f"unavailable: {len(audit['original_source_files_unavailable'])}.",
        "The earlier summary/history/selection audit remains historical: this rebuild does not claim",
        "fresh verification of missing originals, checkpoints, dataset membership or final test performance.",
        "See [current verification](visibility_check.json) for source hashes and exact unavailable paths.", "",
        "Nothing here requires repeating search just to display the results. Recover original evidence",
        "for a full provenance audit; transformer learning and final testing remain separate tasks.",
        "One seed/input-condition sweep does not establish a global optimum or independent generalisation.", "",
        "## Fixed settings and unchanged shortlist", "",
        "All runs use eight RGB frames, seed 42, Adam LR 0.001, WD 0 and no augmentation.",
        "AAD has 748 training / 160 validation / 161 locked test clips across 11 classes.",
        "Search cap 256 epochs, minimum 64, patience 24; lowest validation-loss checkpoint.",
        "Ranking uses equal-condition mean accuracy, worst-condition accuracy, loss, parameters and name.",
        "The saved top five remain: " + ", ".join(f"[{r['architecture']}]" for r in ranking[:5]) + ".", "",
        f"[Unchanged full-precision CSV]({source_link}/search_results.csv) · "
        f"[Unchanged architecture ranking]({source_link}/architecture_ranking.csv) · "
        "[All 128 original-image links](search_run_gallery.md)", "",
    ]
    gallery = ["# All 64 runs: curves and confusion matrices", "",
               "Validation-only original-image links; no images or scores regenerated.",
               "Every run is listed openly; click either image link to inspect it.",
               "[Back to the graphs](README.md#reading-the-graphs)", ""]

    for size, fps in CONDITIONS:
        group = [r for r in rows if r["frame_size"] == size and r["fps"] == fps]
        assert len(group) == 16
        for row in group:
            number = row["experiment"]
            gallery += [f"## Run {number:02d}: {row['architecture']} | {size}px / {fps} FPS", "",
                        f"Eight frames | seed {row['seed']}", "",
                        f"[Original training curves]({source_link}/search_images/{number:02d}/training_curves.png) · "
                        f"[Original validation confusion matrix]({source_link}/search_images/{number:02d}/validation_confusion_matrix.png)", ""]
    lines += ["## Next work", "",
              "This is reporting, not a new result. Diagnose the collapsed transformers before final comparison.",
              "The earlier ten comparison trainings finished, but test failed before scoring; no final test metrics are supplied here.",
              "Use the complete evidence in "
              "[080: manuscript figure/appendix](../../../../../agents/work/080-aad-search-figure-and-appendix/prompt.md) "
              "after approval; the manuscript is unchanged.", "",
              "## Rebuild", "", "From the implementation directory (Python + matplotlib only):", "",
              "```bash", ".venv/bin/python reports/2026-10-14/build_preview.py --search-visibility --output-dir reports/2026-10-15",
              "```", "", "Rebuilding verifies source hashes and refuses to overwrite unrelated or edited output files.", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "search_run_gallery.md").write_text("\n".join(gallery), encoding="utf-8")
    plot_check = search_plot(rows, ranking, output_dir, annotate_examples=True)
    audit.update({"report_type": "search_visibility_v1", "search_runs": 64, "architectures": 16,
                  "runs_per_panel": 16, "original_images_linked": 128,
                  "plot_keys": [[r["name"], r["frame_size"], r["fps"]] for r in rows],
                  **plot_check,
                  "output_sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                                    for name in sorted(files)}})
    (output_dir / "visibility_check.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-visibility", action="store_true", help="Render saved results without imported runs.")
    parser.add_argument("--output-dir", type=Path, help="Separate output directory for --search-visibility.")
    args = parser.parse_args()
    if args.search_visibility:
        output = args.output_dir or ROOT.parent / "2026-10-15"
        audit = build_search_visibility(output)
        print(f"Verified 64 measured plot points and 128 unchanged image copies. Report: {output.resolve()}")
        print(f"Unavailable original run files: {len(audit['original_source_files_unavailable'])}; no training or test access.")
        return
    if args.output_dir:
        parser.error("--output-dir requires --search-visibility")
    for name, expected in EXPECTED_HASHES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    raw = (ROOT / "colab_console.raw.txt").read_text(encoding="utf-8-sig")
    assert "Test 1/10" in raw and "checkpoint dataset does not match the requested dataset" in raw
    assert "Test 2/10" not in raw
    search, ranking = parse_search(raw)
    comparison, config = parse_archive(search, ranking)
    search_audit = audit_search_artifacts(search, ranking)
    for name, rows in [("search_results.csv", search), ("architecture_ranking.csv", ranking),
                       ("comparison_validation.csv", comparison)]:
        csv_output(name, rows)
    csv_output("comparison_per_class.csv", [{"model": r["name"], **c} for r in comparison for c in r["per_class"]])
    payload = {
        "review_date": "2026-10-14", "execution_date_as_logged": "2026-10-14",
        "execution_timezone": "unverified", "code_revision_as_logged": "e269ae5",
        "source_sha256": EXPECTED_HASHES, "test_status": "failed before scoring first model",
        "search_artifact_check": search_audit,
        "evidence_limit": "All 64 search summaries/histories cross-checked against newly imported runs. Checkpoint weights and final comparison freeze/test evidence remain unverified.",
        "search": search, "architecture_ranking": ranking, "comparison": comparison,
        "resolved_comparison_config": config,
    }
    (ROOT / "preview_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plots(search, ranking, comparison)
    # Generated fragments make the manually written interpretation auditable.
    lines = ["# Generated numeric tables", "", "## Search (validation accuracy, %)", "",
             "| Rank | Architecture | Layers | Parameters | 32px/8FPS | 32px/16FPS | 64px/8FPS | 64px/16FPS | Mean | Worst |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ranking:
        scores = [r[f"accuracy_{s}px_{f}fps"] for s, f in CONDITIONS] + [r["mean_accuracy"], r["worst_accuracy"]]
        lines.append(f"| {r['rank']} | {r['architecture']} | {r['layers']} | {r['params']:,} | " + " | ".join(f"{v*100:.3f}" for v in scores) + " |")
    lines += ["", "## Comparison (validation only; P/R/F1 are macro, %)", "",
              "| Model | Parameters | Correct/160 | Accuracy | Loss | P | R | F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {r['name']} ({r['architecture']}) | {r['params']:,} | {r['correct']} | {r['accuracy']*100:.3f} | {r['loss']:.4f} | {r['macro_precision']*100:.3f} | {r['macro_recall']*100:.3f} | {r['macro_f1']*100:.3f} |")
    lines += ["", "## Training and efficiency", "",
              "| Model | Train acc % | Selected/actual epochs | Train min | Tensor MB | Latency ms | Clips/s | Peak CUDA MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        lines.append(f"| {r['name']} | {r['train_accuracy']*100:.3f} | {r['selected_epoch']}/{r['actual_epochs']} | {r['training_seconds']/60:.2f} | {r['weight_tensor_bytes']/1e6:.2f} | {r['latency_ms']:.2f} | {r['clips_per_second']:.1f} | {r['peak_cuda_bytes']/2**20:.1f} |")
    (ROOT / "numeric_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified: source hashes, ZIP CRC, 64 unique search runs, 16 complete four-condition groups,")
    print("top-five handoff, 11 archived summaries, 10 comparison confusion matrices and metric identities.")
    print("Imported evidence: 64/64 search summaries and histories matched; 128 original images linked.")
    print("Generated validation-only CSV/JSON, tables and PNG/SVG previews. No test scores fabricated.")
    print("Final comparison checkpoint bytes:", sum(r["checkpoint_bytes"] for r in comparison))


if __name__ == "__main__":
    main()
