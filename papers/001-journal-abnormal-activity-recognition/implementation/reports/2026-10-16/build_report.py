"""Audit the supplied results-only archive and rebuild this review's tables/plots.

Run with Python and matplotlib. No training, dataset access, or model imports.
The original archive/logs are read-only; generated tables/figures are reproducible.
"""

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
EXPECTED_SHA = "3042f4512825c93cfc831f0f591d0ecd37c51c85dcdfbd058290d5153cbefe6f"
CONDITIONS = [(8, 8), (8, 16), (16, 8), (16, 16)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_csv(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected):
    assert math.isclose(actual, expected, abs_tol=1e-6), (actual, expected)


def member(path):
    return path.split("/runs/", 1)[1]


def audit_predictions(z, path, manifest, partition, dataset, metrics, cm, per_class):
    records = json.loads(z.read(member(path)))
    expected = {s["path"]: s for s in manifest["samples"] if s["split"] == partition}
    classes = manifest["class_names"]
    matrix = [[0] * len(classes) for _ in classes]
    by_path = {}
    for row in records:
        assert row["source"].startswith(dataset + "/")
        relative = row["source"][len(dataset) + 1:]
        assert relative not in by_path and relative in expected
        target, predicted = row["target"], row["predicted"]
        assert target == expected[relative]["class_index"]
        assert row["partition"] == partition
        assert row["true_class"] == classes[target]
        assert row["predicted_class"] == classes[predicted]
        assert row["correct"] == (target == predicted)
        assert math.isfinite(row["loss"]) and row["loss"] >= 0
        assert 0 <= row["confidence"] <= 1
        matrix[target][predicted] += 1
        by_path[relative] = row
    assert set(by_path) == set(expected)
    assert cm["matrix"] == matrix and cm["class_names"] == classes
    assert cm["sample_count"] == len(records)
    correct = sum(matrix[i][i] for i in range(len(classes)))
    accuracy = correct / len(records)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        close(metrics[metric], accuracy)
    close(metrics["loss"], mean(r["loss"] for r in records))
    assert len(per_class) == len(classes)
    for i, row in enumerate(per_class):
        assert row.get("class", row.get("class_name")) == classes[i]
        tp = matrix[i][i]
        support = sum(matrix[i])
        fp = sum(r[i] for r in matrix) - tp
        fn = support - tp
        assert [row[k] for k in ["tp", "fp", "fn", "support"]] == [tp, fp, fn, support]
        close(row["precision"], tp / (tp + fp) if tp + fp else 0)
        close(row["recall"], tp / support if support else 0)
        close(row["f1"], 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return correct, accuracy, by_path


def wilson(correct, n):
    p, z = correct / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - radius, center + radius


def analyze():
    archive = HERE / "evidence/results.zip"
    assert sha(archive.read_bytes()) == EXPECTED_SHA
    rows, raw_results = [], {}
    checked_records = 0
    with ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) == 310 and z.testzip() is None
        assert all(not n.startswith("/") and ".." not in n.split("/") for n in names)
        inventory = json.loads(z.read("archive_inventory.json"))
        assert set(inventory) == set(names) - {"archive_inventory.json"}
        for name, entry in inventory.items():
            data = z.read(name)
            assert len(data) == entry["bytes"] and sha(data) == entry["sha256"]
        search, final = json.loads(z.read("essential_results.json"))["studies"]
        assert search["status"] == final["status"] == "complete"
        assert search["test_access"] == "locked"
        assert final["test_access"] == "evaluated_after_validation_freeze"
        assert [len(search["jobs"]), len(final["transfer_jobs"]), len(final["jobs"]), len(final["tests"])] == [128, 5, 10, 10]
        manifest = search["split_manifest"]
        assert manifest == final["split_manifest"]
        assert manifest["counts"] == {"train": 748, "validation": 160, "test": 161, "total": 1069}
        assert len({s["path"] for s in manifest["samples"]}) == 1069
        freeze, selection = final["validation_frozen"], search["selected_config"]
        revision = search["run"]["git_revision"]
        assert revision == final["run"]["git_revision"]
        assert final["transformer_checks"]["status"] == "passed"
        assert set(final["transformer_checks"]["models"]) == {"swin3d_t", "swin3d_s"}
        assert all(r["passed"] for r in final["transformer_checks"]["models"].values())
        assert final["transformer_checks"]["validation_clips_opened"] == final["transformer_checks"]["test_clips_opened"] == 0
        split_sha = sha(z.read(search["study_id"] + "/split_manifest.json"))
        assert split_sha == freeze["split_sha256"]
        for study, filename, expected in [
            (search, "selected_config.json", freeze["search_selection_sha256"]),
            (final, "resolution_checks.json", freeze["resolution_checks_sha256"]),
            (final, "transformer_checks.json", freeze["transformer_checks"]["sha256"]),
        ]:
            assert sha(z.read(study["study_id"] + "/" + filename)) == expected
        for stage, jobs in [("search", search["jobs"]), ("resolution_check", final["transfer_jobs"]), ("comparison", final["jobs"])]:
            for job in jobs:
                r = job["result"]
                cfg, train, sel = r["experiment_config"], r["training_evaluation"], r["checkpoint_selection"]
                assert r["reporting_protocol"] == "pooled_per_class_v2"
                assert sel["selection_partition"] == "validation" and sel["selection_metric"] == "loss"
                assert sel["split_manifest_hash"] == split_sha
                assert train["checkpoint"] == r["selected_checkpoint"]
                assert train["protocol"] == "matched_checkpoint_eval_v2" and not train["augmentation"]
                assert cfg["seed"] == 42 and cfg["batch_size"] == 32
                assert cfg["weight_decay"] == 0 and not cfg["augment"]
                assert cfg["optimizer"] == "adam" and cfg["sampling_version"] == "timestamps_v1"
                assert cfg["epochs"] == (256 if stage == "comparison" else 128)
                assert cfg["width"] == cfg["height"] == (32 if stage == "search" else 64)
                assert cfg["learning_rate"] == (1e-5 if r["name"].startswith("swin") else .001)
                _, train_acc, _ = audit_predictions(z, train["predictions"], manifest, "train", cfg["dataset_dir"], train["metrics"], train["confusion_matrix"], train["per_class"])
                correct, val_acc, _ = audit_predictions(z, r["validation_predictions"], manifest, "validation", cfg["dataset_dir"], r["validation_metrics"], r["validation_confusion_matrix"], r["validation_per_class"])
                checked_records += 908
                custom = r["model_class"] == "CustomConvLSTM"
                layers = [v[0] for v in cfg["convlstm_layers"]] if custom else []
                eff = r["efficiency"]
                row = dict(stage=stage, experiment_number=job["experiment_number"], model=r["name"], architecture="-".join(map(str, layers)) if custom else r["name"], depth=len(layers), frames=cfg["sequence_length"], fps=cfg["target_fps"], size=cfg["height"], seed=42, learning_rate=cfg["learning_rate"], weight_decay=0, batch_size=cfg["batch_size"], parameters=r["num_params"], weight_mib=r["weight_tensor_bytes"] / 2**20, checkpoint_mib=r["checkpoint_bytes"] / 2**20, train_accuracy=train_acc, train_loss=train["metrics"]["loss"], validation_correct=correct, validation_accuracy=val_acc, validation_loss=r["validation_metrics"]["loss"], selected_epoch=sel["selected_epoch"], actual_epochs=r["early_stopping"]["actual_epochs"], training_seconds=r["train_time_s"], latency_ms=eff["latency_ms_per_batch"], inference_peak_cuda_mib=eff["peak_cuda_memory_bytes"] / 2**20)
                rows.append(row)
                raw_results[(stage, r["name"])] = r
        assert sorted(r["experiment_number"] for r in rows) == list(range(1, 144))
        base = [r for r in rows if r["stage"] == "search"]
        checks = [r for r in rows if r["stage"] == "resolution_check"]
        comparison = [r for r in rows if r["stage"] == "comparison"]
        console = (HERE / "evidence/search_console.txt").read_text(encoding="utf-8")
        printed = re.findall(r"input_(\d+)f_32px_(\d+)fps (custom_[\d_]+) complete .*?val_acc=([\d.]+) val_loss=([\d.]+).*?params=(\d+)", console)
        assert len(printed) == 128
        assert len({(name, frames, fps) for frames, fps, name, *_ in printed}) == 128
        for frames, fps, name, accuracy, loss, params in printed:
            r = next(r for r in base if r["model"] == name and (r["frames"], r["fps"]) == (int(frames), int(fps)))
            close(float(accuracy), r["validation_accuracy"])
            # Console uses six significant digits, unlike the full-precision JSON.
            assert float(loss) == float(f"{r['validation_loss']:.6g}")
            assert int(params) == r["parameters"]
        grouped = defaultdict(list)
        for r in base:
            grouped[r["model"]].append(r)
        assert len(grouped) == 32
        ranking = []
        for name, items in grouped.items():
            assert sorted((r["frames"], r["fps"]) for r in items) == CONDITIONS
            ranking.append(dict(model=name, architecture=items[0]["architecture"], depth=items[0]["depth"], parameters=items[0]["parameters"], mean_validation_accuracy=float(sum(Fraction(r["validation_correct"], 160) for r in items) / 4), minimum_validation_accuracy=min(r["validation_accuracy"] for r in items), mean_validation_loss=mean(r["validation_loss"] for r in items)))
        ranking.sort(key=lambda r: (-r["mean_validation_accuracy"], -r["minimum_validation_accuracy"], r["mean_validation_loss"], r["parameters"], r["model"]))
        assert [r["model"] for r in ranking] == [r["name"] for r in selection["ranking"]]
        for r, saved in zip(ranking, selection["ranking"]):
            for field in ["mean_validation_accuracy", "minimum_validation_accuracy", "mean_validation_loss", "parameters"]:
                close(r[field], saved[field])
        assert [r["model"] for r in ranking[:5]] == selection["finalist_names"]
        for i, row in enumerate(ranking, 1):
            row["rank"] = i
        finalists = {r["model"] for r in ranking[:5]}
        inputs = []
        for frames, fps in CONDITIONS:
            subset = [r for r in base if r["model"] in finalists and (r["frames"], r["fps"]) == (frames, fps)]
            inputs.append((mean(r["validation_accuracy"] for r in subset), mean(r["validation_loss"] for r in subset), frames, fps))
        chosen = sorted(inputs, key=lambda r: (-r[0], r[1], r[2], -r[3]))[0]
        assert chosen[2:] == (8, 16)
        assert all((r["frames"], r["fps"]) == (8, 16) for r in checks + comparison)
        for i, ranked in enumerate(ranking[:5], 1):
            assert next(r for r in comparison if r["model"] == f"custom_top{i}")["architecture"] == ranked["architecture"]
        chosen_custom = sorted(comparison[:5], key=lambda r: (-r["validation_accuracy"], r["validation_loss"], r["parameters"], r["model"]))[0]
        check_order = sorted(checks, key=lambda r: (-r["validation_accuracy"], r["validation_loss"], r["parameters"], r["model"]))
        assert [r["model"] for r in check_order] == final["resolution_checks"]["validation_64px_order"]
        assert chosen_custom["model"] == freeze["best_custom"] == "custom_top1"
        test_records, per_class, matrices = {}, [], {}
        frozen = {r["model"]: r for r in freeze["models"]}
        assert set(frozen) == {r["model"] for r in comparison} == {t["model"] for t in final["tests"]}
        for test in final["tests"]:
            name, report = test["model"], test["report"]
            r = raw_results[("comparison", name)]
            row = next(row for row in comparison if row["model"] == name)
            assert report["reporting_protocol"] == "pooled_per_class_v2"
            assert report["checkpoint"]["path"] == frozen[name]["checkpoint"] == r["selected_checkpoint"]
            assert r["training_evaluation"]["checkpoint_sha256"] == frozen[name]["checkpoint_sha256"]
            assert report["checkpoint"]["selected_epoch"] == row["selected_epoch"]
            assert report["checkpoint"]["split_manifest_hash"] == split_sha
            assert report["num_params"] == row["parameters"]
            correct, acc, records = audit_predictions(z, report["predictions"], manifest, "test", report["experiment_config"]["dataset_dir"], report["metrics"], report["artifacts"]["confusion_matrix"], report["per_class"])
            checked_records += 161
            test_records[name] = records
            lo, hi = wilson(correct, 161)
            row.update(test_correct=correct, test_samples=161, test_accuracy=acc, test_precision=acc, test_recall=acc, test_f1=acc, test_loss=report["metrics"]["loss"], nominal_wilson95_low=lo, nominal_wilson95_high=hi)
            per_class.extend(dict(model=name, **entry) for entry in report["per_class"])
            matrices[name] = {k: report["artifacts"]["confusion_matrix"][k] for k in ["class_names", "matrix", "sample_count"]}
        paired = []
        winner_records = test_records[freeze["best_custom"]]
        for name, records in test_records.items():
            if name == freeze["best_custom"]:
                continue
            a = sum(winner_records[p]["correct"] and not r["correct"] for p, r in records.items())
            b = sum(not winner_records[p]["correct"] and r["correct"] for p, r in records.items())
            n = a + b
            p_value = min(1., 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2**n) if n else 1.
            paired.append(dict(reference=freeze["best_custom"], other=name, reference_only_correct=a, other_only_correct=b, exact_mcnemar_p_unadjusted=p_value))
        conditions = []
        for frames, fps in CONDITIONS:
            items = [r for r in base if (r["frames"], r["fps"]) == (frames, fps)]
            conditions.append(dict(frames=frames, fps=fps, mean_train_accuracy=mean(r["train_accuracy"] for r in items), mean_validation_accuracy=mean(r["validation_accuracy"] for r in items), at_epoch_cap=sum(r["actual_epochs"] == 128 for r in items)))
        hours = {stage: sum(r["training_seconds"] for r in rows if r["stage"] == stage) / 3600 for stage in ["search", "resolution_check", "comparison"]}
        stats = dict(conditions=conditions, top5=ranking[:5], selected_input_top5_mean=chosen[0], training_hours=hours, winner=chosen_custom, winner_errors={p: r for p, r in winner_records.items() if not r["correct"]}, resolution_checks=checks, paired_tests=paired)
        final_console = (HERE / "evidence/comparison_console.txt").read_text(encoding="utf-8")
        assert re.findall(r"SHA256 ([a-f0-9]{64})", final_console)[-1] == EXPECTED_SHA
        printed_tests = re.findall(r"^(\w+): test accuracy=([\d.]+), loss=([\d.]+), parameters=([\d,]+)", final_console, re.M)
        assert len(printed_tests) == 10
        for name, accuracy, loss, params in printed_tests:
            r = next(r for r in comparison if r["model"] == name)
            assert abs(float(accuracy) - r["test_accuracy"]) <= .000051
            assert abs(float(loss) - r["test_loss"]) <= .000051
            assert int(params.replace(",", "")) == r["parameters"]
        verification = dict(status="passed", archive_sha256=EXPECTED_SHA, archive_bytes=archive.stat().st_size, archive_members=len(names), inventory_entries_verified=len(inventory), split_sha256=split_sha, training_jobs=143, prediction_partitions=296, prediction_records_checked=checked_records, search_ranking="all 32 ranks match saved exact-count selection", console_rows_verified={"search": 128, "final": 10}, experiment_numbers="1 through 143, continuous", frozen_best_custom=freeze["best_custom"], git_revision=revision, search_code_identity=search["code_identity"], final_summary_code_identity=final["code_identity"], limits=["Checks prove internal archive consistency, not independent rerunning of models.", "No raw videos, checkpoint tensors, source-audit file, or full epoch histories in this compact ZIP.", "Checkpoint SHA identities agree across metadata; absent tensor bytes cannot be rehashed.", "Drive copy receipt matches this ZIP; live Drive availability was not independently queried.", "Final summary code_identity is null; both run.json files record the same revision and search retains source hashes."])
        assert checked_records == 131454
    for name, data in [("search_results.csv", base), ("architecture_ranking.csv", ranking), ("resolution_checks.csv", checks), ("comparison_results.csv", comparison), ("comparison_per_class.csv", per_class), ("paired_test_checks.csv", paired)]:
        save_csv(name, data)
    save_json("test_confusion_matrices.json", matrices)
    save_json("statistics.json", stats)
    save_json("verification.json", verification)
    return base, ranking, comparison, stats


def save_figure(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=180)
    path = HERE / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    # Normalize whitespace in generated SVGs only.
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def accuracy_axis_forward(values):
    """Use October 15's disclosed 20:1 compression below 90% accuracy."""
    import numpy as np
    values = np.asarray(values)
    return np.where(values <= 90, values * .05, 4.5 + values - 90)


def accuracy_axis_inverse(values):
    import numpy as np
    values = np.asarray(values)
    return np.where(values <= 4.5, values / .05, 90 + values - 4.5)


def panel_examples(rows):
    middle = median(r["validation_accuracy"] for r in rows)
    return {
        "Best": min(rows, key=lambda r: (-r["validation_accuracy"], r["parameters"], r["experiment_number"])),
        "Middle": min(rows, key=lambda r: (round(abs(r["validation_accuracy"] - middle), 12), r["parameters"], r["experiment_number"])),
        "Worst": min(rows, key=lambda r: (r["validation_accuracy"], r["parameters"], r["experiment_number"])),
    }


def search_figure(base, compressed):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    # Same layer and role colours/markers as the October 15 report.
    colors = {1: "#0072B2", 2: "#009E73", 3: "#D55E00", 4: "#CC79A7", 5: "#6A5ACD"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}
    role_colors = {"Best": "#006D2C", "Middle": "#2166AC", "Worst": "#B2182B"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    plotted, panels = [], []
    for ax, (frames, fps) in zip(axes.flat, CONDITIONS):
        subset = [r for r in base if (r["frames"], r["fps"]) == (frames, fps)]
        assert len(subset) == len({r["model"] for r in subset}) == 32
        for r in subset:
            point = ax.scatter(r["parameters"], 100 * r["validation_accuracy"], color=colors[r["depth"]], marker=markers[r["depth"]], s=65, edgecolors="white", linewidths=1.1, alpha=.8, zorder=3 + r["depth"])
            point.set_gid(f"search-run-{r['experiment_number']:03d}")
            plotted.append(r["experiment_number"])
        ax.set_title(f"{frames} frames at {fps} FPS", fontsize=12)
        ax.set_xscale("log")
        ax.set_xlim(min(r["parameters"] for r in base) * .72, max(r["parameters"] for r in base) * 1.52)
        ax.set_xticks([50000, 100000, 500000, 1000000, 3000000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1000:g}k"))
        if compressed:
            ax.set_yscale("function", functions=(accuracy_axis_forward, accuracy_axis_inverse))
            ax.set_ylim(0, 100.5)
            ax.set_yticks([0, 20, 40, 60, 80, 90, 92, 94, 96, 98, 100])
            ax.axhline(90, color=".5", linestyle="--", linewidth=.8)
            ax.text(.02, 90.3, "compressed below 90%", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8, color=".35")
        else:
            ax.set_ylim(0, 104)
        ax.grid(alpha=.2)
        examples = panel_examples(subset)
        handles = []
        for role, r in examples.items():
            ring = ax.scatter(r["parameters"], 100 * r["validation_accuracy"], s=195, facecolors="none", edgecolors=role_colors[role], linewidths=1.7, zorder=12)
            ring.set_gid(f"example-{role.lower()}-{r['experiment_number']:03d}")
            handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=role_colors[role], markeredgewidth=1.7, markersize=9, label=f"{role}: {r['architecture']} ({100 * r['validation_accuracy']:.2f}%)"))
        if compressed:
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(.02, .35), fontsize=9, framealpha=.94, edgecolor=".85", title="Highlighted models (val accuracy)", title_fontsize=9)
        else:
            ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=.94, title="Highlighted models (val accuracy)", title_fontsize=8)
        panels.append(dict(frames=frames, fps=fps, size=32, count=len(subset), highlights={role: {k: r[k] for k in ["experiment_number", "model", "architecture", "parameters", "validation_accuracy"]} for role, r in examples.items()}))
    for ax in axes[-1]:
        ax.set_xlabel("Trainable parameters (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation accuracy (%)")
    handles = [Line2D([], [], color=colors[d], marker=markers[d], linestyle="", label=f"{d} layer{'s' if d > 1 else ''}") for d in range(1, 6)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .012))
    fig.suptitle("ConvLSTM Architecture Search", y=.98)
    fig.tight_layout(rect=(0, .06, 1, .95))
    assert sorted(plotted) == list(range(1, 129))
    save_figure(fig, "search_capacity" if compressed else "search_capacity_linear")
    plt.close(fig)
    return dict(plotted_experiments=plotted, panels=panels, y_axis={"scale": "piecewise_linear" if compressed else "linear", "threshold": 90 if compressed else None, "lower_compression": .05 if compressed else 1})


def confusion_figure(comparison):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    winner = json.loads((HERE / "verification.json").read_text(encoding="utf-8"))["frozen_best_custom"]
    r = next(r for r in comparison if r["model"] == winner)
    cm = json.loads((HERE / "test_confusion_matrices.json").read_text(encoding="utf-8"))[winner]
    matrix, classes = np.asarray(cm["matrix"]), cm["class_names"]
    assert int(matrix.sum()) == cm["sample_count"] == r["test_samples"] == 161
    assert int(matrix.trace()) == r["test_correct"] == 160
    fig, ax = plt.subplots(figsize=(10, 8.6))
    plot = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.max()))
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center", fontsize=10, color="white" if count > matrix.max() / 2 else ("#B2182B" if count and i != j else (".15" if count else ".45")))
        if count and i != j:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#B2182B", linewidth=2))
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title="ConvLSTM Test Confusion Matrix")
    fig.colorbar(plot, ax=ax, fraction=.046, pad=.04, label="Number of test clips", ticks=range(0, int(matrix.max()) + 1, 4))
    fig.text(.5, .02, "Raw counts (not percentages). Only error: Hijack → Terrorism, 1 clip.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, 1))
    save_figure(fig, "best_model_test_confusion")
    plt.close(fig)
    return dict(model=winner, architecture=r["architecture"], partition="test", sample_count=int(matrix.sum()), correct=int(matrix.trace()), class_names=classes, matrix=cm["matrix"])


def test_table_rows(comparison):
    """A test-only convenience export; retain the complete audit CSV unchanged."""
    fields = ["model", "architecture", "test_accuracy", "test_precision", "test_recall", "test_f1", "parameters", "test_loss", "latency_ms", "test_correct", "test_samples"]
    return [{k: r[k] for k in fields} for r in comparison]


def figures(base, ranking, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "aad-2026-10-16"})
    compressed = search_figure(base, compressed=True)
    linear = search_figure(base, compressed=False)
    assert compressed["panels"] == linear["panels"]
    confusion = confusion_figure(comparison)
    save_csv("comparison_test.csv", test_table_rows(comparison))
    save_json("presentation_check.json", dict(status="passed", search=compressed, linear_search=linear, best_model_test_confusion=confusion, comparison_rows=len(comparison), test_table_columns=list(test_table_rows(comparison)[0])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for width in [32, 48, 64, 96]:
        items = [next(r for r in ranking if r["architecture"] == "-".join([str(width)] * d)) for d in range(1, 6)]
        axes[0].plot(range(1, 6), [r["mean_validation_accuracy"] * 100 for r in items], marker="o", label=f"Width {width}")
    for frames, fps in CONDITIONS:
        widths = [16, 32, 48, 64, 96, 128]
        items = [next(r for r in base if r["architecture"] == "-".join([str(w)] * 3) and (r["frames"], r["fps"]) == (frames, fps)) for w in widths]
        axes[1].plot(widths, [r["validation_accuracy"] * 100 for r in items], marker="o", label=f"{frames} frames / {fps} FPS")
    axes[0].set(title="Recurrent depth", xlabel="ConvLSTM layers", ylabel="Mean validation accuracy (%)", xticks=range(1, 6))
    axes[1].set(title="Layer width", xlabel="Filters in each layer", ylabel="Validation accuracy (%)", xticks=[16, 32, 48, 64, 96, 128])
    for ax in axes:
        ax.set_ylim(0, 104)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("ConvLSTM Depth and Width", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save_figure(fig, "capacity_controls")
    plt.close(fig)


if __name__ == "__main__":
    base, ranking, comparison, stats = analyze()
    figures(base, ranking, comparison)
    print("Verified 128 search jobs, 5 resolution checks and 10 final tests; figures rebuilt.")
    print("\nTest-only table: model | accuracy % | precision % | recall % | F1 % | M parameters | loss | ms")
    for r in comparison:
        scores = " | ".join(f"{100 * r[k]:.3f}" for k in ["test_accuracy", "test_precision", "test_recall", "test_f1"])
        print(f"| {r['architecture']} | {scores} | {r['parameters']/1e6:.3f} | {r['test_loss']:.4f} | {r['latency_ms']:.2f} |")
