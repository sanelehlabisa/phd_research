"""Budgeted suite contracts; synthetic clips only, no real study or downloads."""

import copy
import json
import time
from pathlib import Path

import pytest
import torch

from test_notebook_workflows import prepared
from src import notebook_config as settings
from src import notebook_diagnostics as diagnostics
from src import notebook_models as models
from src import notebook_suite as suite
from src.model import count_trainable_parameters
from src.notebook_data import NotebookVideoDataset
from src.notebook_data import prepare_data
from src.utils import write_video_torchvision


def test_declared_models_and_native_counts():
    root = Path(__file__).resolve().parents[1]
    specs = suite.candidate_specs(root)
    assert len(specs) == 14
    assert set(models.BASELINES) <= set(specs)
    with torch.device("meta"):
        for name, spec in specs.items():
            model = models.build_model(spec, 5, (3, 64, 64), 32)
            assert count_trainable_parameters(model) > 0
        paper = models.build_model(
            dict(name="paper_convlstm_published"), 11, (3, 50, 50), 50
        )
        assert count_trainable_parameters(paper) == 512197467
        with pytest.raises(ValueError, match="native"):
            models.build_model(
                dict(name="paper_convlstm_published"), 5, (3, 32, 32), 16
            )


@pytest.mark.parametrize("name", models.BASELINES)
def test_video_factory_forward_and_checkpoint(name):
    torch.set_num_threads(1)
    model = models.build_model(dict(name=name), 5, (3, 16, 16), 4).eval()
    inputs = torch.rand(1, 4, 3, 16, 16)
    with torch.no_grad():
        expected = model(inputs)
    restored = models.from_checkpoint(
        dict(model_config=model.configuration(), model_state_dict=model.state_dict())
    ).eval()
    with torch.no_grad():
        assert torch.equal(restored(inputs), expected)
    assert expected.shape == (1, 5)


def test_gentle_schedule_has_floor():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=settings.LR_FACTOR,
        patience=settings.LR_PATIENCE,
        min_lr=settings.MIN_LR,
    )
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step(1.0)
    assert rates[:7] == [0.001] * 7
    assert rates[7] == 0.0005
    assert min(rates) == 1e-5
    assert rates[23] >= 0.000125


def test_temporal_windows_seed_cache_and_test_lock(prepared):
    dataset = prepared["dataset"]
    index = prepared["train"].indices[0]
    dataset[index]
    center = dict(dataset.window_records[index])
    assert center["start_seconds"] > 0
    dataset.set_epoch(1, 42, training=True)
    first = dataset[index]
    random_window = dict(dataset.window_records[index])
    assert random_window["sampling"] == "random"
    assert torch.equal(first[0], dataset[index][0])
    dataset.set_epoch(2, 42, training=True)
    assert dataset.window(index)["start_seconds"] != random_window["start_seconds"]
    dataset.set_epoch(1, 42, training=True)
    assert dataset.window(index) == random_window
    dataset.training_windows = False
    dataset[index]
    assert dataset.window_records[index] == center
    with pytest.raises(PermissionError):
        dataset.window(prepared["test"].indices[0])


def test_center_window_reaches_later_action_and_low_fps_padding(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SELECTED_DIAGNOSTIC_DATASET", "vdd")
    root = tmp_path / "clips"
    for name in ("violent", "non-violent"):
        for number in range(6):
            frames = torch.zeros(32, 3, 8, 8)
            frames[8:24] = 1.0  # the first second contains no action
            write_video_torchvision(frames, root / name / f"{number}.mp4", fps=4)
    prepared = prepare_data(tmp_path, root, 8, sequence_length=8, target_fps=8)
    dataset = prepared["dataset"]
    index = prepared["train"].indices[0]
    frames, _ = dataset[index]
    assert frames.shape == (8, 3, 8, 8)
    assert float(frames.mean()) > 0.9
    assert dataset.window_records[index]["start_seconds"] == pytest.approx(3.5)
    long = prepare_data(tmp_path, root, 8, sequence_length=40, target_fps=4)
    frames, _ = long["dataset"][index]
    assert frames.shape[0] == 40
    assert torch.equal(frames[-1], frames[-2])
    assert long["manifest_path"] == prepared["manifest_path"]


def test_ablation_plan_changes_one_factor(prepared):
    config = suite.configuration(prepared, settings.SCREEN_EPOCHS).to_dict()
    spec = dict(name="custom", layers=config["convlstm_layers"], dropout=0.1)
    custom = dict(config=config, model_spec=spec)
    plan = dict(
        data_identity=dict(target_fps=8),
        coverage_fps=[4, 8, 16],
        dropout_trials=[0, 0.5],
        weight_decay_trials=[0, 0.001],
    )
    for variant in suite.ablation_trials(plan, custom):
        changed = [key for key in config if config[key] != variant["config"][key]]
        if variant["name"].startswith("coverage"):
            assert not changed and variant["model"] == spec and variant["fps"] != 8
        elif variant["name"].startswith("dropout"):
            assert not changed and variant["fps"] == 8
            assert {k for k in spec if spec[k] != variant["model"][k]} == {"dropout"}
        else:
            assert (
                changed == ["weight_decay"]
                and variant["model"] == spec
                and variant["fps"] == 8
            )


def test_audit_never_reads_test_and_tiny_diagnostics(prepared, monkeypatch):
    monkeypatch.setattr(settings, "TINY_STEPS", 16)
    monkeypatch.setattr(settings, "TINY_PER_CLASS", 1)
    dataset = prepared["dataset"]
    rows = diagnostics.audit_sources(
        prepared, prepared["root"] / "audit.json", time.time() + 60
    )
    assert len(rows) == len(prepared["train"]) + len(prepared["validation"])
    assert {r["partition"] for r in rows} == {"train", "validation"}
    assert all(r["timestamp_fps"] == pytest.approx(24) for r in rows)
    target = prepared["root"] / "tiny.json"
    try:
        diagnostics.tiny_learnability(prepared, target, time.time() + 60)
    except ValueError as error:
        assert "did not reach" in str(error)
    result = json.loads(target.read_text())
    assert set(result["indices"]) <= set(prepared["train"].indices)
    assert result["history"][-1]["gradient_norm"] > 0
    assert result["history"][-1]["parameter_update_norm"] > 0
    assert dataset.allowed_indices.isdisjoint(prepared["test"].indices)


def _small_suite(monkeypatch):
    monkeypatch.setattr(
        suite,
        "candidate_specs",
        lambda root: {
            "custom": dict(name="custom", layers=[[2, [3, 3]]], dropout=0.1),
            "r3d_18": dict(name="r3d_18"),
        },
    )
    monkeypatch.setattr(suite, "BASELINES", ("r3d_18",))
    monkeypatch.setattr(settings, "TEMPORAL_FPS", (4, 8, 16))
    monkeypatch.setattr(suite, "tiny_learnability", lambda *args: {"passed": True})
    monkeypatch.setattr(suite, "display", lambda *args: None)
    monkeypatch.setattr(suite, "LiveCurves", lambda: None)
    factory = models.build_model
    monkeypatch.setattr(
        models,
        "build_model",
        lambda spec, n, shape, frames: factory(
            dict(name="custom", layers=[[2, [3, 3]]], dropout=spec.get("dropout", 0.1)),
            n,
            shape,
            frames,
        ),
    )


def test_full_suite_freeze_and_saved_display_recovery(prepared, monkeypatch):
    _small_suite(monkeypatch)
    directory = suite.run_screen(prepared)
    assert (directory / "screen.csv").exists()
    assert (
        json.loads((directory / "native_paper.json").read_text())["status"]
        == "resource_limited"
    )
    fine = suite.train_winner(prepared, directory)
    frozen = json.loads((directory / "frozen.json").read_text())
    assert len(frozen["results"]) == 4
    assert {r["config"]["seed"] for r in frozen["results"]} == {42, 2026}
    suite._checked_confirmation(fine, directory)
    altered = copy.deepcopy(frozen)
    altered["results"].pop()
    (directory / "frozen.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Both seeds"):
        suite.final_evaluate(fine, directory)
    (directory / "frozen.json").write_text(json.dumps(frozen))
    with monkeypatch.context() as patch:
        patch.setattr(
            suite,
            "show_predictions",
            lambda *args: (_ for _ in ()).throw(TypeError("frontend failed")),
        )
        with pytest.raises(TypeError, match="frontend"):
            suite.final_evaluate(fine, directory)
    assert (directory / "final_test.json").exists()
    assert fine["dataset"].allowed_indices.isdisjoint(fine["test"].indices)
    monkeypatch.setattr(
        NotebookVideoDataset, "__getitem__", lambda *args: pytest.fail("test reopened")
    )
    monkeypatch.setattr(suite, "show_predictions", lambda *args: None)
    report = suite.final_evaluate(fine, directory)
    assert len(report["rows"]) == 4
    assert "previously_inspected" in report["evidence_role"]
    assert len(report["examples"]) == min(5, len(fine["test"]))
    with pytest.raises(ValueError, match="Already frozen"):
        suite.train_winner(prepared, directory)


def test_budget_and_tiny_failure_prevent_screen(prepared, monkeypatch):
    _small_suite(monkeypatch)
    monkeypatch.setattr(settings, "SUITE_HOURS", 0)
    monkeypatch.setattr(
        suite, "_trial", lambda *args: pytest.fail("training after deadline")
    )
    with pytest.raises(TimeoutError):
        suite.run_screen(prepared)
    runs = list((prepared["root"] / "runs/experiments").glob("*/run.json"))
    assert json.loads(runs[-1].read_text())["status"] == "partial"
    monkeypatch.setattr(settings, "SUITE_HOURS", 8)
    monkeypatch.setattr(suite, "audit_sources", lambda *args: [])
    monkeypatch.setattr(
        suite,
        "tiny_learnability",
        lambda *args: (_ for _ in ()).throw(ValueError("tiny failure")),
    )
    with pytest.raises(ValueError, match="tiny failure"):
        suite.run_screen(prepared)
