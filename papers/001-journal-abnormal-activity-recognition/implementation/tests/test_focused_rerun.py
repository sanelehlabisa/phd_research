"""077/076: balanced search, honest shortlist and compact automatic delivery."""

import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from src import focused_protocol as focused, capacity_search as search, study_config
from src import study_matrix as matrix
from src.experiment_config import ExperimentConfig
from src.utils import write_json
from notebooks.utils import aad_study, study_archive
from test_capacity_search import ROOT, protocol, fake_train
from test_focused_workflow import review_for

PROFILE = ROOT / "configs/experiments/aad_focused_search_colab.json"


def records():
    cfg = study_config.load_study(PROFILE)[1]
    rows = []
    for i, spec in enumerate(focused.candidates()):
        for size, fps in focused.CONDITIONS:
            correct = 80 + i
            rows.append(
                {
                    "result": {
                        "name": spec["name"],
                        "seed": 42,
                        "partition": "validation",
                        "test_access": "locked",
                        "experiment_config": {
                            **cfg.to_dict(),
                            "height": size,
                            "width": size,
                            "target_fps": fps,
                        },
                        "num_params": 1000 + i,
                        "validation_metrics": {"accuracy": correct / 100, "loss": 0.2},
                        "validation_confusion_matrix": {
                            "matrix": [[correct, 100 - correct], [0, 0]]
                        },
                    }
                }
            )
    return rows


def test_exact_focused_plan_and_rejected_expansion(capsys):
    study, cfg, manifest = study_config.load_study(PROFILE)
    rows = [r for _, block in focused.blocks(cfg) for r in block]
    assert len(manifest.candidates) == 16 and len(rows) == 64
    assert len({search.job_key(r) for r in rows}) == 64
    assert set(tuple(s) for s in focused.STACKS) >= {(64,) * d for d in range(1, 6)}
    assert {r["config"]["target_fps"] for r in rows} == {8, 16}
    assert {r["config"]["height"] for r in rows} == {32, 64}
    assert all(
        r["config"]["sequence_length"] == 8
        and r["config"]["weight_decay"] == 0
        and r["matched_checkpoint_metrics"]
        for r in rows
    )
    study_config.print_study(PROFILE, study, cfg)
    assert "Experiment 64/64" in capsys.readouterr().out
    for field, value in (
        ("target_fps", [4, 8, 16]),
        ("seeds", [42, 2026]),
        ("max_runs", 65),
    ):
        with pytest.raises(ValueError, match="077 requires"):
            focused.load_focused({**study, field: value})
    changed = copy.deepcopy(study)
    changed["training"]["epochs"] = 128
    with pytest.raises(ValueError, match="077 requires"):
        focused.load_focused(changed)


def test_equal_condition_ranking_reserved_anchor_and_input_freeze():
    rows = records()
    decision = focused.decisions(rows)
    assert decision["anchor_reserved_slot"] and decision["anchor_rank"] == 16
    assert decision["finalist_names"] == decision["numerical_top5"][:4] + [
        focused.ANCHOR
    ]
    assert (
        decision["final_input"]["height"] == 32
        and decision["final_input"]["target_fps"] == 16
    )
    assert decision["shared_recipe"]["weight_decay"] == 0
    # A perfect isolated condition cannot beat consistently good four-condition coverage.
    for item in rows:
        r = item["result"]
        if r["name"] == focused.ANCHOR:
            correct = 100 if r["experiment_config"]["height"] == 32 else 50
            r["validation_confusion_matrix"]["matrix"] = [
                [correct, 100 - correct],
                [0, 0],
            ]
            r["validation_metrics"]["accuracy"] = correct / 100
    assert focused.decisions(rows)["anchor_rank"] == 16
    with pytest.raises(ValueError, match="coverage"):
        focused.decisions(rows[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        focused.decisions(rows + rows[:1])
    rows[0]["result"]["partition"] = "test"
    with pytest.raises(ValueError, match="validation"):
        focused.decisions(rows)


def test_preflight_common_batch_all_families_and_no_fit():
    cfg = study_config.load_study(PROFILE)[1]
    probes = []

    def probe(spec):
        probes.append(spec)
        return {"memory_safe": spec["batch_size"] <= 4, "training_step_seconds": 0.1}

    report = focused.preflight(cfg, final=True, probe=probe)
    assert report["selected_batch_size"] == 4
    passed = [r for r in probes if r["batch_size"] == 4]
    assert {r["model"] for r in passed} == {
        "custom_bound",
        "custom_depth5",
        *focused.BASELINES,
    }
    assert all(r["frames"] == 8 and r["size"] == 64 for r in passed)

    def oom(spec):
        raise torch.cuda.OutOfMemoryError("synthetic")

    assert focused.preflight(cfg, probe=oom)["selected_batch_size"] is None


def test_ranking_minimum_condition_tie_and_anchor_can_earn_top_five():
    rows = records()
    names = [c["name"] for c in focused.candidates()][:2]
    scores = {names[0]: [80, 100, 80, 100], names[1]: [90] * 4}
    for i, item in enumerate(rows):
        r = item["result"]
        count = scores[r["name"]][i % 4] if r["name"] in scores else 20
        r["validation_confusion_matrix"]["matrix"] = [[count, 100 - count], [0, 0]]
        r["validation_metrics"]["accuracy"] = count / 100
    decision = focused.decisions(rows)
    assert [r["name"] for r in decision["ranking"][:2]] == names[::-1]
    assert not decision["anchor_reserved_slot"] and decision["anchor_rank"] == 2
    assert decision["finalist_names"] == decision["numerical_top5"]


def test_eight_frame_adapted_parameter_count_and_distinct_native():
    from src.experiments import build_registered_model

    with torch.device("meta"):
        adapted = build_registered_model("paper_convlstm_adapted", 11, (3, 64, 64), 8)
        native = build_registered_model("paper_convlstm_published", 11, (3, 50, 50), 50)
    assert sum(p.numel() for p in adapted.parameters()) == 134415195
    assert sum(p.numel() for p in native.parameters()) == 512197467


def test_compact_bundle_omits_weights_and_retains_restart_metadata(tmp_path):
    group = tmp_path / "runs/studies/synthetic"
    leaf = tmp_path / "runs/experiments/synthetic"
    leaf.mkdir(parents=True)
    (leaf / "best.pth").write_bytes(b"checkpoint")
    write_json(leaf / "history.json", {"epochs": []})
    write_json(group / "progress.json", {"jobs": [{"run_dir": str(leaf)}]})
    write_json(
        group / "jobs/1/receipt.json",
        {"status": "complete", "files": matrix.inventory(leaf)},
    )
    archive = study_archive.compact_results(group, tmp_path / "runs")
    contents = study_archive.verify_archive(archive)
    assert "bundle_policy.json" in contents
    assert any(n.endswith("history.json") for n in contents)
    assert any(n.endswith("receipt.json") for n in contents)
    assert not any(n.endswith(".pth") for n in contents)
    assert (leaf / "best.pth").read_bytes() == b"checkpoint"
    full = study_archive.checkpoint_archive(tmp_path, group)
    assert any(n.endswith(".pth") for n in study_archive.verify_archive(full))
    (leaf / "history.json").write_text("modified")
    with pytest.raises(RuntimeError, match="changed"):
        study_archive.compact_results(group, tmp_path / "runs")


def test_new_notebook_export_parity_and_download_marker_scope(tmp_path, monkeypatch):
    nb = matrix.read_json(ROOT / "notebooks/07_aad_focused_workflow.ipynb")
    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    export = (ROOT / "notebooks/scripts/07_aad_focused_workflow.py").read_text(
        encoding="utf-8"
    )
    assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(export))
    assert "run_focused_aad_study(root)" in code and 'pull", "--ff-only"' in code
    assert all(c.get("outputs", []) == [] for c in nb["cells"])
    with pytest.raises(ValueError, match="outside"):
        aad_study._download_compact_marker(tmp_path, str(tmp_path / "outside.zip"))

    def fail(path):
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr(study_archive, "retry_download", fail)
    # Delivery failure is visible, not a reason to restart training.
    aad_study._download_compact_marker(
        tmp_path, str(tmp_path / "runs/focused_archives/results.zip")
    )


def test_automatic_64_search_ten_comparison_test_archive_and_reuse(
    protocol, monkeypatch
):
    p = protocol
    p.tested = []
    values = matrix.read_json(PROFILE)
    values["dataset"].update(
        path=str(p.data), split_manifest=p.values["dataset"]["split_manifest"]
    )
    values["runs_dir"] = str(p.root / "runs")
    for _, rel in aad_study.PROFILES:
        write_json(p.root / rel, matrix.read_json(ROOT / rel))
    write_json(p.root / "configs/experiments/aad_focused_search_colab.json", values)
    final = matrix.read_json(
        ROOT / "configs/experiments/aad_shortlist_comparison_colab.json"
    )
    final["runs_dir"] = str(p.root / "runs")
    write_json(
        p.root / "configs/experiments/aad_shortlist_comparison_colab.json", final
    )
    review = write_json(
        p.root / "review.json", review_for(values["dataset"]["split_manifest"])
    )
    audit = search.audit_split
    monkeypatch.setattr(
        search,
        "audit_split",
        lambda source, split, review_path=None: audit(source, split, review),
    )
    monkeypatch.setattr(aad_study, "resolve_dataset", lambda *args: str(p.data))
    downloads = []
    monkeypatch.setattr(
        aad_study, "_download_archive", lambda path: downloads.append(path)
    )
    base_train = fake_train(p)

    def train(args):
        folder = base_train(args)
        r = matrix.read_json(folder / "summary.json")["all"][0]
        if r["experiment_config"]["epochs"] == 512:
            r["early_stopping"]["actual_epochs"] = 128
            r["checkpoint_selection"]["selected_epoch"] = 128
            r["training_evaluation"]["selected_epoch"] = 128
            side = Path(r["selected_checkpoint"]).with_suffix(".json")
            write_json(side, {**matrix.read_json(side), "selected_epoch": 128})
        (folder / "curve.png").write_bytes(b"synthetic plot")
        r.update(
            weight_tensor_bytes=400,
            checkpoint_bytes=25,
            validation_metric_pass_seconds=0.1,
            training_curves=str(folder / "curve.png"),
        )
        write_json(folder / "summary.json", {"all": [r]})
        return folder

    def evaluate(path):
        cfg = matrix.read_json(path)
        frozen = matrix.read_json(Path(path).parent.parent / "validation_frozen.json")
        assert len(frozen["models"]) == 10
        assert sum(c["epochs"] == 512 for _, c in p.trained) == 10
        assert cfg["prediction_samples_per_category"] == (
            5 if Path(path).stem == frozen["best_custom"] else 0
        )
        split = matrix.read_json(cfg["split_manifest"])
        rows = [
            dict(
                source=str((p.data / r["path"]).resolve()),
                partition="test",
                target=r["class_index"],
                predicted=r["class_index"],
                loss=0.0,
            )
            for r in split["samples"]
            if r["split"] == "test"
        ]
        folder = p.root / "runs/evaluate" / Path(path).stem
        preds = write_json(folder / "predictions.json", rows)
        report = dict(
            partition="test",
            split={"manifest_hash": matrix.file_hash(cfg["split_manifest"])},
            checkpoint={"path": cfg["checkpoint_path"]},
            experiment_config={k: v for k, v in cfg.items() if k != "checkpoint_path"},
            predictions=str(preds),
            metrics=dict(loss=0.0, accuracy=1.0, precision=1.0, recall=1.0, f1=1.0),
            metric_pass_seconds=0.1,
            prediction_examples={
                "records": [],
                "saved_counts": {"correct": 0, "incorrect": 0},
            },
        )
        write_json(folder / "metrics/final.json", report)
        write_json(folder / "run.json", {"status": "complete"})
        p.tested.append(Path(path).stem)
        return folder

    def command(args, root, log_path=None):
        path = Path(args[args.index("--config") + 1])
        loaded = study_config.load_study(path)
        if "--list-plan" not in args:
            return study_config.execute_study(path, *loaded, train, evaluate)

    monkeypatch.setattr(aad_study, "_run_command", command)
    archive = aad_study.run_focused_aad_study(p.root)
    assert len(p.trained) == 74 and len(p.tested) == 10
    entries = study_archive.verify_archive(archive)
    assert any(n.endswith("comparison.csv") for n in entries)
    assert not any(n.endswith(".pth") for n in entries)
    group = next((p.root / "runs/studies").glob("*top3_comparison"))
    report = matrix.read_json(group / "comparison.json")
    assert len(report["table"]) == 10
    assert [r["experiment_number"] for r in report["table"]] == list(range(65, 75))
    assert all(r["frames"] == 8 and r["weight_decay"] == 0 for r in report["table"])
    assert all(
        "test_per_class" in r and "test_micro_f1" in r and "test_macro_f1" not in r
        for r in report["table"]
    )
    selected_path = Path(matrix.read_json(group / "study.json")["selected_config"])
    selected = matrix.load_selection(selected_path)
    assert len(selected["top5"]) == 5 and len(selected["ranking"]) == 16
    assert selected["final_input"]["height"] == report["table"][0]["height"]
    # Compatible rerun uses identical IDs, checkpoints and completed tests.
    assert aad_study.run_focused_aad_study(p.root) == archive
    assert len(p.trained) == 74 and len(p.tested) == 10
    resolved = matrix.read_json(group / "study.json")
    resolved["training"]["learning_rate"] = 0.003
    with pytest.raises(ValueError, match="frozen"):
        study_config.load_study(write_json(p.root / "bad_comparison.json", resolved))
    write_json(group / "test_receipts/custom_top1.json", {"status": "started"})
    with pytest.raises(RuntimeError):
        aad_study.run_focused_aad_study(p.root)
    assert len(p.trained) == 74 and len(p.tested) == 10
