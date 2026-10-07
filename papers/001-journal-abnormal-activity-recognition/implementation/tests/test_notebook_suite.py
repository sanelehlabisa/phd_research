"""Budgeted suite contracts; synthetic clips only, no real study or downloads."""

import copy
import json
import sys
import time
import types
import zipfile
from pathlib import Path

import pytest
import torch

from test_notebook_workflows import prepared
from notebooks.utils import config as settings
from notebooks.utils import diagnostics
from notebooks.utils import models
from notebooks.utils import suite
from src.model import count_trainable_parameters
from notebooks.utils.data import NotebookVideoDataset
from notebooks.utils.data import prepare_data
from src.utils import write_video_torchvision


def test_declared_models_and_native_counts():
    root = Path(__file__).resolve().parents[1]
    specs = suite.candidate_specs(root)
    assert len(specs) == 14
    kinetics_manifest = json.loads(
        (root / "configs/experiments/kinetics_diagnostic_candidates.json").read_text()
    )
    aad_manifest = json.loads(
        (root / "configs/experiments/aad_architecture_candidates.json").read_text()
    )
    assert len(kinetics_manifest["candidates"]) == 11
    assert len(aad_manifest["candidates"]) == 3
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
        spatial_sizes=[config["height"], 12, 16],
    )
    for variant in suite.ablation_trials(plan, custom):
        changed = [key for key in config if config[key] != variant["config"][key]]
        if variant["name"].startswith("coverage"):
            assert not changed and variant["model"] == spec and variant["fps"] != 8
        elif variant["name"].startswith("dropout"):
            assert not changed and variant["fps"] == 8
            assert {k for k in spec if spec[k] != variant["model"][k]} == {"dropout"}
        elif variant["name"].startswith("spatial"):
            assert set(changed) == {"height", "width"}
            assert variant["config"]["height"] == variant["config"]["width"]
            assert variant["model"] == spec and variant["fps"] == 8
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
    monkeypatch.setattr(settings, "SUITE_SCREEN_EPOCHS", 1)
    monkeypatch.setattr(settings, "SUITE_FINAL_EPOCHS", 2)
    monkeypatch.setattr(settings, "SUITE_FINAL_FRAME_SIZE", 12)
    monkeypatch.setattr(settings, "SUITE_SPATIAL_SIZES", (8, 12, 16))
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
    assert "not_independent_paper_evidence" in report["evidence_role"]
    assert len(report["examples"]) == min(5, len(fine["test"]))
    with pytest.raises(ValueError, match="Already frozen"):
        suite.train_winner(prepared, directory)
    archive_path = suite.create_suite_artifact_archive(prepared, directory)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    for run_dir in prepared["suite_train_run_dirs"]:
        prefix = f"{Path(run_dir).relative_to(prepared['root'])}/"
        assert f"{prefix}config.json" in names
        assert f"{prefix}history.json" in names
        assert any(name.startswith(prefix + "checkpoints/") for name in names)


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
    archive_path = suite.create_suite_artifact_archive(prepared)
    with zipfile.ZipFile(archive_path) as archive:
        assert any(name.endswith("/plan.json") for name in archive.namelist())
        assert any(name.endswith("/run.json") for name in archive.namelist())
    monkeypatch.setattr(settings, "SUITE_HOURS", 8)
    monkeypatch.setattr(suite, "audit_sources", lambda *args: [])
    monkeypatch.setattr(
        suite,
        "tiny_learnability",
        lambda *args: (_ for _ in ()).throw(ValueError("tiny failure")),
    )
    with pytest.raises(ValueError, match="tiny failure"):
        suite.run_screen(prepared)


def test_suite_artifact_archive_contains_only_current_suite_and_models(tmp_path):
    root = tmp_path / "implementation"
    suite_dir = root / "runs/experiments/current-suite"
    train_dir = root / "runs/train/current-model"
    old_train_dir = root / "runs/train/older-model"
    dataset_dir = root / "datasets/source-videos"
    for directory in (suite_dir, train_dir, old_train_dir, dataset_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (suite_dir / "plan.json").write_text("{}", encoding="utf-8")
    (suite_dir / "summary.csv").write_text("model,accuracy\n", encoding="utf-8")
    (train_dir / "config.json").write_text("{}", encoding="utf-8")
    (train_dir / "history.json").write_text("[]", encoding="utf-8")
    (train_dir / "loss_curve.png").write_bytes(b"curve")
    (train_dir / "best_model.pth").write_bytes(b"weights")
    (old_train_dir / "old.json").write_text("{}", encoding="utf-8")
    (dataset_dir / "source.mp4").write_bytes(b"dataset")
    prepared = {
        "root": root,
        "suite_run_dir": suite_dir,
        "suite_train_run_dirs": [train_dir],
    }

    archive_path = suite.create_suite_artifact_archive(prepared)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "runs/experiments/current-suite/plan.json" in names
    assert "runs/train/current-model/config.json" in names
    assert "runs/train/current-model/history.json" in names
    assert "runs/train/current-model/loss_curve.png" in names
    assert "runs/train/current-model/best_model.pth" in names
    assert not any("older-model" in name for name in names)
    assert not any("source-videos" in name for name in names)
    assert suite_dir.joinpath("plan.json").is_file()
    assert train_dir.joinpath("best_model.pth").is_file()


def test_suite_data_views_share_registered_model_runs(prepared):
    prepared["suite_run_dir"] = prepared["root"] / "runs/experiments/current-suite"
    prepared["suite_train_run_dirs"] = [Path("runs/train/current-model")]
    config = suite.configuration(prepared, settings.SUITE_SCREEN_EPOCHS)

    view = suite._data(prepared, config, prepared["dataset"].target_fps)

    assert view["suite_run_dir"] == prepared["suite_run_dir"]
    assert view["suite_train_run_dirs"] is prepared["suite_train_run_dirs"]


def test_suite_artifact_download_uses_colab_browser(tmp_path, monkeypatch):
    root = tmp_path / "implementation"
    suite_dir = root / "runs/experiments/current-suite"
    suite_dir.mkdir(parents=True)
    (suite_dir / "plan.json").write_text("{}", encoding="utf-8")
    downloaded = []
    google_module = types.ModuleType("google")
    colab_module = types.ModuleType("google.colab")
    files_module = types.ModuleType("google.colab.files")
    files_module.download = downloaded.append
    colab_module.files = files_module
    google_module.colab = colab_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    monkeypatch.setitem(sys.modules, "google.colab.files", files_module)

    archive_path = suite.download_suite_artifacts(
        {"root": root, "suite_run_dir": suite_dir}, suite_dir
    )

    assert downloaded == [str(archive_path)]
    assert archive_path.is_file()


def test_experiment_notebook_cells_download_even_after_budget_timeout():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/04_run_experiments.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for index in (4, 6, 8):
        source = "".join(cells[index]["source"])
        compile(source, f"notebook cell {index}", "exec")
        assert "except TimeoutError" in source
    assert "download_suite_artifacts(prepared, screen_dir)" in "".join(
        cells[8]["source"]
    )
