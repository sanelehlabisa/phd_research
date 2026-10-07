"""Larger single-model profile and safe time-limited training; no GPU/downloads."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from notebooks.utils import config as settings
from notebooks.utils import workflows
from src import train as training
from src.model import CustomConvLSTM, count_trainable_parameters
from test_notebook_workflows import prepared


def test_training_profile_is_explicit_and_independent(tmp_path, monkeypatch):
    assert settings.TRAIN_LAYERS == ((32, (3, 3)), (64, (3, 3)), (64, (3, 3)))
    assert (settings.TRAIN_SEQUENCE_LENGTH, settings.TRAIN_TARGET_FPS) == (32, 8)
    assert (settings.TRAIN_FRAME_SIZE, settings.TRAIN_BATCH_SIZE) == (96, 8)
    assert (settings.TRAIN_EPOCHS, settings.TRAIN_HOURS, settings.SEED) == (200, 8, 42)
    assert (settings.SEQUENCE_LENGTH, settings.TARGET_FPS, settings.FRAME_SIZE) == (
        16,
        16,
        32,
    )
    assert settings.DEFAULT_LAYERS == ((8, (3, 3)), (16, (3, 3)))
    assert (settings.SUITE_SCREEN_EPOCHS, settings.SUITE_FRAME_SIZE) == (24, 64)
    assert settings.SUITE_FINAL_EPOCHS == 64
    data = dict(
        root=tmp_path,
        specification=SimpleNamespace(key="kinetics600-subset"),
        manifest_path=tmp_path / "manifest.json",
        dataset=SimpleNamespace(
            dataset_dir=tmp_path / "videos",
            frame_size=(96, 96),
            sequence_length=32,
            target_fps=8,
        ),
    )
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return data

    monkeypatch.setattr(workflows, "prepare_data", prepare)
    assert workflows.prepare_training_data(tmp_path, "local-videos") is data
    assert calls == [
        (
            (tmp_path, "local-videos"),
            dict(frame_size=96, sequence_length=32, target_fps=8),
        )
    ]
    monkeypatch.setattr(time, "time", lambda: 1000)

    def run(prepared, config, label, callback, **kwargs):
        assert prepared is data and label == "single-model"
        assert config.convlstm_layers == settings.TRAIN_LAYERS
        assert config.dataset_name == "kinetics600-subset"
        assert (config.epochs, config.height, config.width, config.batch_size) == (
            200,
            96,
            96,
            8,
        )
        assert config.learning_rate == 0.001 and config.weight_decay == 0.0001
        assert config.scheduler == "reduce_on_plateau" and not config.augment
        assert kwargs == dict(deadline=29800, return_on_timeout=True)
        return "trained"

    monkeypatch.setattr(workflows, "train_notebook_model", run)
    assert workflows.train_single(data) == "trained"
    for hours in (0, -1, float("nan"), float("inf")):
        monkeypatch.setattr(settings, "TRAIN_HOURS", hours)
        with pytest.raises(ValueError, match="finite and positive"):
            workflows.train_single(data)


def test_larger_architecture_forward_and_backward():
    torch.set_num_threads(1)
    model = CustomConvLSTM(5, layers=list(settings.TRAIN_LAYERS), dropout=0.1)
    preview = CustomConvLSTM(5, layers=list(settings.DEFAULT_LAYERS), dropout=0.1)
    assert count_trainable_parameters(model) > count_trainable_parameters(preview)
    # Same architecture/head, tiny inputs: do not allocate A100-scale activations.
    logits = model(torch.randn(2, 2, 3, 8, 8))
    assert logits.shape == (2, 5)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )


@pytest.mark.parametrize(
    "stage", ["training", "validation", "second_epoch", "clean_training"]
)
def test_time_limit_preserves_only_validated_evidence(prepared, monkeypatch, stage):
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(settings, "TRAIN_EPOCHS", 3 if stage == "second_epoch" else 1)
    deadline = now[0] + settings.TRAIN_HOURS * 3600
    calls = {"training": 0, "validation": 0}

    def wrap(function, kind):
        def execute(*args):
            calls[kind] += 1
            original_progress = args[-1]

            def progress(*values):
                if (
                    stage == kind
                    or (
                        stage == "second_epoch"
                        and kind == "training"
                        and calls[kind] == 2
                    )
                    or (
                        stage == "clean_training"
                        and kind == "validation"
                        and calls[kind] == 2
                    )
                ):
                    now[0] = deadline + 1
                original_progress(*values)

            return function(*args[:-1], progress)

        return execute

    monkeypatch.setattr(
        training,
        "train_classifier_epoch",
        wrap(training.train_classifier_epoch, "training"),
    )
    monkeypatch.setattr(
        training,
        "evaluate_classifier",
        wrap(training.evaluate_classifier, "validation"),
    )
    result = workflows.train_single(prepared)
    completed = int(stage in ("second_epoch", "clean_training"))
    assert result["status"] == "partial" and result["actual_epochs"] == completed
    assert result["stop_reason"] == "time_budget_exhausted"
    assert (
        result["clean_training_metrics"] is None and result["test_access"] == "locked"
    )
    directory = Path(result["run_dir"])
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["status"] == "partial" and manifest["results"] == result
    assert manifest["deadline_unix"] == deadline
    assert manifest["trainable_parameters"] == result["num_params"]
    history = json.loads((directory / "history.json").read_text())
    assert len(history) == completed
    assert not prepared["dataset"].training_windows
    assert prepared["dataset"].allowed_indices.isdisjoint(prepared["test"].indices)
    if completed:
        _, checkpoint = workflows.load_checkpoint(
            prepared, result["selected_checkpoint"]
        )
        assert checkpoint["training_status"] == "partial"
        assert checkpoint["early_stopping"]["actual_epochs"] == 1
        assert checkpoint["early_stopping"]["stop_reason"] == "time_budget_exhausted"
        assert checkpoint["selection_value"] == history[0]["validation"]["loss"]
        records = workflows.show_validation_predictions(prepared, result)
        assert len(records) == min(5, len(prepared["validation"]))
        assert all(row["partition"] == "validation" for row in records)
        assert (directory / "learning_curves.png").is_file()
    else:
        assert (
            result["selected_checkpoint"] is None
            and result["validation_metrics"] is None
        )
        assert not list(directory.rglob("*.pth"))
        assert workflows.show_validation_predictions(prepared, result) == []


def test_unrelated_timeout_is_not_treated_as_budget(prepared, monkeypatch):
    def fail(*args):
        raise TimeoutError("decoder timed out")

    monkeypatch.setattr(training, "train_classifier_epoch", fail)
    with pytest.raises(TimeoutError, match="decoder timed out"):
        workflows.train_single(prepared)
    manifest = next((prepared["root"] / "runs/train").glob("*/run.json"))
    assert json.loads(manifest.read_text())["status"] == "failed"


def test_suite_callers_still_raise_on_time_limit(prepared, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])

    def fail(*args):
        now[0] = 2000
        raise TimeoutError("Diagnostic compute budget exhausted")

    monkeypatch.setattr(training, "train_classifier_epoch", fail)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        training.train_notebook_model(
            prepared, workflows.configuration(prepared, 1), "suite", deadline=1001
        )


def test_training_rejects_preview_input(prepared, monkeypatch):
    monkeypatch.setattr(settings, "TRAIN_FRAME_SIZE", 96)
    with pytest.raises(ValueError, match="prepare_training_data"):
        workflows.train_single(prepared)


def test_predictions_require_successful_training():
    with pytest.raises(ValueError, match="Train a model"):
        workflows.show_validation_predictions({}, None)


def test_decoder_timeout_is_not_hidden_after_training_deadline(prepared, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])

    def fail(*args):
        now[0] += settings.TRAIN_HOURS * 3600 + 1
        raise TimeoutError("decoder timed out, not the training budget")

    monkeypatch.setattr(training, "train_classifier_epoch", fail)
    with pytest.raises(TimeoutError, match="decoder timed out"):
        workflows.train_single(prepared)
    manifest = next((prepared["root"] / "runs/train").glob("*/run.json"))
    assert json.loads(manifest.read_text())["status"] == "failed"


def test_full_larger_input_shape_without_allocating_gpu_memory():
    with torch.device("meta"), torch.inference_mode():
        model = CustomConvLSTM(
            5, layers=list(settings.TRAIN_LAYERS), dropout=0.1
        ).eval()
        output = model(torch.zeros(8, 32, 3, 96, 96))
    assert output.shape == (8, 5)


def test_fresh_notebook_export_and_current_helper_imports():
    import ast

    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/03_train_model.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code, filename=str(path))
    compile(tree, "fresh_03_export.py", "exec")
    modules = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(name and name.startswith("src.notebook_") for name in modules)
    for name in modules:
        if name and name.startswith("notebooks.utils."):
            assert root.joinpath(*name.split(".")).with_suffix(".py").is_file()
    assert "prepared = prepare_training_data(IMPLEMENTATION_ROOT)" in code
    assert "training_result = None" in code
    assert "['git', 'pull', '--ff-only', 'origin', 'master']" in code
    assert "reset" not in code and "checkout --" not in code
