"""Synthetic verification only: no real research data or test results."""

import ast
import copy
import json
from pathlib import Path

import pytest
import torch
from sklearn.metrics import precision_recall_fscore_support

from notebooks.utils import aad_final as final
from src import dataset, dataset_source, evaluate, experiments
from src.utils import write_json, write_video_torchvision

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/aad_staged_experiments.json"


def test_notebook_has_matching_plain_python_export():
    notebook = json.loads(
        (ROOT / "notebooks/05_aad_train_and_evaluate.ipynb").read_text()
    )
    code = "\n\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    script = (ROOT / "notebooks/05_aad_train_and_evaluate.py").read_text()
    assert script.strip() == code.strip()
    ast.parse(script)
    assert 'pull", "--ff-only"' in script and "reset" not in script
    assert 'str(IMPLEMENTATION_ROOT / "requirements.txt")' in script
    assert "bootstrap.returncode == 75" in script
    assert "validate_runtime(require_cuda=True)" in script
    assert "train_and_freeze" in script and "evaluate_frozen" in script
    assert not any(c.get("outputs") for c in notebook["cells"])


@pytest.mark.parametrize(
    "name", ["r3d_18", "mc3_18", "r2plus1d_18", "swin3d_t", "swin3d_s"]
)
def test_evaluator_reconstructs_every_practical_baseline(name):
    # Meta tensors exercise exact architecture/state keys without large allocations.
    with torch.device("meta"):
        original = experiments.build_registered_model(name, 11, (3, 32, 32), 8)
    checkpoint = {
        "model_config": {
            "registry_name": name,
            "num_classes": 11,
            "input_dimensions": {"height": 32, "width": 32, "sequence_length": 8},
        },
        "model_state_dict": original.state_dict(),
    }
    restored = evaluate.model_from_checkpoint(checkpoint)
    assert restored.state_dict().keys() == original.state_dict().keys()
    assert sum(p.numel() for p in restored.parameters()) == sum(
        p.numel() for p in original.parameters()
    )
    restored.eval()
    with torch.device("meta"), torch.inference_mode():
        assert restored(torch.zeros(1, 8, 3, 32, 32)).shape == (1, 11)


def test_unknown_or_incompatible_checkpoint_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        evaluate.model_from_checkpoint(
            {"model_config": {"registry_name": "paper_convlstm_published"}}
        )
    with pytest.raises(ValueError, match="legacy"):
        evaluate.model_from_checkpoint({})


def test_incomplete_study_is_rejected_before_training(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No training/evaluation before evidence exists")

    monkeypatch.setattr(experiments, "main", forbidden)
    monkeypatch.setattr(evaluate, "main", forbidden)
    with pytest.raises(ValueError, match="STUDY_RUN_DIR"):
        final.train_and_freeze("")
    write_json(tmp_path / "run.json", {"status": "running"})
    with pytest.raises(ValueError, match="incomplete"):
        final.train_and_freeze(tmp_path)


def test_synthetic_confirmation_freeze_evaluation_and_rerun(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = tmp_path / "aad"
    for name in dataset_source.AAD_CLASS_NAMES:
        for number in range(10):
            write_video_torchvision(
                torch.rand(3, 3, 8, 8), root / name / f"{number}.mp4"
            )
    split = tmp_path / "split.json"
    source = dataset.AHARDataset(root, 2, (8, 8))
    _, _, test, _ = dataset.load_split_subsets(source, split)
    locked = {source.samples[i][0].resolve() for i in test.indices}
    original_load = dataset.AHARDataset._load_video
    accesses = []
    allow_test = False

    def guarded(self, path):
        if Path(path).resolve() in locked:
            assert allow_test, "Decoded test before checkpoint freeze"
            accesses.append(str(path))
        return original_load(self, path)

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", guarded)
    values = json.loads(CONFIG.read_text())
    values["dataset"].update(path=str(root), split_manifest=str(split))
    values["runs_dir"] = str(tmp_path / "runs")
    values["models"] = ["tiny"]
    values["custom_candidates"] = [
        {
            "name": "tiny",
            "research_question": "Synthetic workflow test",
            "convlstm_layers": [[2, [3, 3]]],
            "hidden_classifier_width": None,
        }
    ]
    values["training"].update(
        epochs=1,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        prediction_samples_per_category=1,
    )
    values["reference_input"] = {
        "frame_size": 8,
        "sequence_length": 2,
        "weight_decay": 0.0,
    }
    values["factors"] = {
        "frame_sizes": [8],
        "sequence_lengths": [2, 3],
        "weight_decays": [0.0],
    }
    path = write_json(tmp_path / "study.json", values)
    study_dir = experiments.main(["--study-config", str(path)])
    plan = final.selection_plan(study_dir, epochs=1)
    assert plan["selection_partition"] == "validation"
    assert plan["seeds"] == [42, 2026]
    assert plan["configuration"]["sequence_length"] in [2, 3]
    assert not accesses

    # Missing jobs, test-based selection and modified study cannot be used.
    summary_path = study_dir / "summary.json"
    summary = final.read_json(summary_path)
    for kind in ["missing", "partition", "config"]:
        bad = copy.deepcopy(summary)
        if kind == "missing":
            bad["jobs"].pop()
        elif kind == "partition":
            bad["selection_partition"] = "test"
        else:
            bad["jobs"][-1]["result"]["experiment_config"]["weight_decay"] = 0.2
        write_json(summary_path, bad)
        with pytest.raises(ValueError):
            final.selection_plan(study_dir, epochs=1)
        write_json(summary_path, summary)
    stored_study = final.read_json(study_dir / "study.json")
    bad = copy.deepcopy(stored_study)
    bad["training"]["epochs"] = 2
    write_json(study_dir / "study.json", bad)
    with pytest.raises(ValueError, match="hash"):
        final.selection_plan(study_dir, epochs=1)
    write_json(study_dir / "study.json", stored_study)

    final_dir = final.train_and_freeze(study_dir, epochs=1)
    frozen = final.read_json(final_dir / "frozen.json")
    assert not accesses
    assert frozen["test_access"] == "locked"
    assert frozen["validation_std"]["accuracy"] >= 0
    records = [
        final.read_json(final_dir / f"seed-{seed}.json")["result"]
        for seed in (42, 2026)
    ]
    best = min(
        records,
        key=lambda r: (
            r["validation_metrics"]["loss"],
            -r["validation_metrics"]["accuracy"],
            r["seed"],
        ),
    )
    assert frozen["checkpoint"] == best["selected_checkpoint"]
    assert "learning_rate" in final.read_json(best["history"])["epochs"][0]
    final.show_results(final_dir)
    assert (final_dir / "curves-42.png").is_file()

    def forbidden(*args, **kwargs):
        pytest.fail("Completed confirmation must not retrain")

    monkeypatch.setattr(experiments, "main", forbidden)
    assert final.train_and_freeze(study_dir, epochs=1) == final_dir
    with pytest.raises(ValueError, match="differs"):
        final.train_and_freeze(study_dir, epochs=2)

    # A modified freeze/split must fail before the evaluator touches test.
    broken = {**frozen, "configuration": {**frozen["configuration"], "height": 96}}
    write_json(final_dir / "frozen.json", broken)
    with pytest.raises(ValueError, match="validation-selected"):
        final.evaluate_frozen(final_dir)
    write_json(final_dir / "frozen.json", frozen)
    split_bytes = split.read_text()
    split.write_text(split_bytes + "\n")  # Fixture-only mutation, restored below.
    with pytest.raises(ValueError, match="split changed"):
        final.evaluate_frozen(final_dir)
    split.write_text(split_bytes)

    # An earlier partial test is never automatically attempted a second time.
    attempt_path = final_dir / "test-attempt.json"
    write_json(
        attempt_path,
        {
            "status": "failed",
            "frozen_sha256": final.file_hash(final_dir / "frozen.json"),
        },
    )
    with pytest.raises(ValueError, match="already attempted"):
        final.evaluate_frozen(final_dir)
    attempt_path.unlink()  # Remove only this synthetic fixture marker before first actual smoke.

    allow_test = True
    report_path = final.evaluate_frozen(final_dir)
    assert accesses
    report = final.read_json(report_path)
    assert report["partition"] == "test"
    assert report["checkpoint"]["path"] == str(Path(frozen["checkpoint"]).resolve())
    predictions = final.read_json(report["predictions"])
    assert len(predictions) == len(test)
    assert {Path(p["source"]).resolve() for p in predictions} == locked
    assert all(p["partition"] == "test" for p in predictions)
    assert report["metrics"]["loss"] == pytest.approx(
        sum(p["loss"] for p in predictions) / len(predictions), abs=1e-6
    )
    assert report["metrics"]["accuracy"] == pytest.approx(
        sum(p["correct"] for p in predictions) / len(predictions), abs=1e-6
    )
    p, r, f, s = precision_recall_fscore_support(
        [row["target"] for row in predictions],
        [row["predicted"] for row in predictions],
        labels=list(range(11)),
        zero_division=0,
    )
    for i, row in enumerate(report["per_class"]):
        assert row["precision"] == pytest.approx(p[i])
        assert row["recall"] == pytest.approx(r[i])
        assert row["f1"] == pytest.approx(f[i])
        assert row["support"] == s[i]
    assert report["num_params"] == plan["num_params"]
    assert report["metric_pass_seconds"] > 0
    assert report["prediction_examples"]["records"]
    for record in report["prediction_examples"]["records"]:
        assert Path(record["path"]).is_file()
    before = len(accesses)
    monkeypatch.setattr(evaluate, "main", forbidden)
    assert final.evaluate_frozen(final_dir) == report_path
    final.show_results(final_dir, report_path)
    assert len(accesses) == before
    assert final.train_and_freeze(study_dir, epochs=1) == final_dir

    # Report changes invalidate reuse.
    changed = copy.deepcopy(report)
    changed["metrics"]["accuracy"] = 1.0
    write_json(report_path, changed)
    with pytest.raises(ValueError, match="report changed"):
        final.evaluate_frozen(final_dir)
    write_json(report_path, report)
    checkpoint = Path(frozen["checkpoint"])
    with checkpoint.open("ab") as stream:
        stream.write(b"fixture-tampering")
    with pytest.raises(ValueError, match="checkpoint changed"):
        final.evaluate_frozen(final_dir)
