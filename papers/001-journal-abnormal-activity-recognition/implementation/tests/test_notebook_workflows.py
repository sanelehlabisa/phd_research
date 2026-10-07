"""Exercise all four workflows on real tiny videos, without Kaggle or CUDA."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import av
import pytest
import torch

from notebook_files import NOTEBOOKS
from notebooks.utils import config as settings
from notebooks.utils import display as visuals
from notebooks.utils import workflows
from notebooks.utils.data import NotebookVideoDataset, prepare_data
from src.model import CustomConvLSTM
from src.utils import seed_everything, write_video_torchvision
from src.vdd_diagnostic import resolve_vdd_root


@pytest.fixture(params=["vdd", "kinetics-subset"])
def prepared(tmp_path, monkeypatch, request):
    torch.set_num_threads(1)
    monkeypatch.setattr(settings, "SELECTED_DIAGNOSTIC_DATASET", request.param)
    monkeypatch.setattr(settings, "SEQUENCE_LENGTH", 2)
    monkeypatch.setattr(settings, "TARGET_FPS", 8)
    monkeypatch.setattr(settings, "FRAME_SIZE", 8)
    monkeypatch.setattr(settings, "BATCH_SIZE", 8)
    monkeypatch.setattr(settings, "TRAIN_EPOCHS", 1)
    monkeypatch.setattr(settings, "TRAIN_SEQUENCE_LENGTH", 2)
    monkeypatch.setattr(settings, "TRAIN_TARGET_FPS", 8)
    monkeypatch.setattr(settings, "TRAIN_FRAME_SIZE", 8)
    monkeypatch.setattr(settings, "TRAIN_BATCH_SIZE", 8)
    monkeypatch.setattr(settings, "TRAIN_LAYERS", ((2, (3, 3)),))
    monkeypatch.setattr(settings, "SCREEN_EPOCHS", 1)
    monkeypatch.setattr(settings, "FINAL_EPOCHS", 2)
    monkeypatch.setattr(settings, "SCREEN_FRAME_SIZES", [8, 12])
    monkeypatch.setattr(settings, "SCREEN_SEQUENCE_LENGTHS", [2, 3])
    monkeypatch.setattr(settings, "SCREEN_BATCH_SIZE", 8)
    monkeypatch.setattr(settings, "SCREEN_WEIGHT_DECAYS", [0.0001])
    monkeypatch.setattr(settings, "SCREEN_AUGMENT_OPTIONS", [False, True])
    monkeypatch.setattr(settings, "DEFAULT_LAYERS", ((2, (3, 3)),))
    monkeypatch.setattr(settings, "SCREEN_MODEL_NAMES", ["small", "wide"])
    monkeypatch.setattr(
        settings,
        "SCREEN_CANDIDATES",
        {
            "small": ((2, (3, 3)),),
            "wide": ((3, (3, 3)),),
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(visuals, "display", lambda *a, **kw: None)
    monkeypatch.setattr(workflows, "display", lambda *a, **kw: None)
    dataset_root = tmp_path / request.param
    classes = (
        ("non-violent", "violent")
        if request.param == "vdd"
        else (
            "headbutting",
            "hugging",
            "punching_person__boxing_",
            "shaking_hands",
            "slapping",
        )
    )
    for label, name in enumerate(classes):
        for index in range(12):
            clip = torch.zeros(12, 3, 16, 16)
            clip[:, label % 3, :, :8] = 0.8
            clip[:, label % 3, :, 8:] = 0.2 + index / 100
            write_video_torchvision(clip, dataset_root / name / f"{index}.mp4", fps=24)
    # Preparation must not open even a header of the locked test clips.
    with monkeypatch.context() as patch:
        patch.setattr(
            av, "open", lambda *a, **kw: pytest.fail("preparation opened video")
        )
        result = prepare_data(tmp_path, dataset_root)
    return result


def test_previews_model_and_single_training(prepared, monkeypatch):
    dataset = prepared["dataset"]
    with pytest.raises(PermissionError):
        dataset[prepared["test"].indices[0]]
    preview = visuals.show_dataset(prepared)
    assert preview["native_fps"] == 24
    assert preview["target_fps"] == 8
    assert preview["partition"] == "train"
    cards = []
    current_video = visuals.Video

    def legacy_video(data=None, filename=None, **kwargs):
        # Reproduce Colab's older IPython filename-only constructor failure.
        os.path.exists(data)
        return current_video(data, filename=filename, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(visuals, "display", cards.append)
        patch.setattr(visuals, "Video", legacy_video)
        visuals.video_card(preview["sampled"], "true <label>", correct=True)
    assert "#26734d" in cards[0].data and "<video" in cards[0].data
    assert "&lt;label&gt;" in cards[0].data
    with av.open(preview["sampled"]) as video:
        assert float(video.streams.video[0].average_rate) == 8
        assert len(list(video.decode(video=0))) == 2
    with av.open(preview["native"]) as video:
        assert float(video.streams.video[0].average_rate) == 24
    inspected = workflows.inspect_model(prepared)
    assert len(inspected) == 1 and inspected[0]["partition"] == "train"
    assert sum(inspected[0]["probabilities"]) == pytest.approx(1)
    trained = workflows.train_single(prepared)
    assert trained["status"] == "complete" and trained["actual_epochs"] == 1
    assert trained["test_access"] == "locked"
    checkpoint = torch.load(
        trained["selected_checkpoint"], map_location="cpu", weights_only=True
    )
    seed_everything(settings.SEED)
    initial = CustomConvLSTM(
        num_classes=dataset.num_classes, layers=list(settings.DEFAULT_LAYERS)
    )
    assert len(inspected[0]["probabilities"]) == dataset.num_classes
    assert (
        checkpoint["model_state_dict"]["classifier.weight"].shape[0]
        == dataset.num_classes
    )
    assert any(
        not torch.equal(value, checkpoint["model_state_dict"][name])
        for name, value in initial.state_dict().items()
    )
    examples = workflows.show_validation_predictions(prepared, trained)
    assert {r["partition"] for r in examples} == {"validation"}
    assert Path(trained["run_dir"], "learning_curves.png").is_file()
    assert dataset.allowed_indices.isdisjoint(prepared["test"].indices)
    assert dataset.cache_bytes <= dataset.cache_limit_bytes
    if prepared["specification"].key == "kinetics-subset":
        # Even the same files/classes under a different configured version must
        # not load an old checkpoint. Class-subset changes must also be rejected.
        with monkeypatch.context() as patch:
            patch.setattr(
                NotebookVideoDataset,
                "__getitem__",
                lambda *a: pytest.fail("incompatible checkpoint opened a clip"),
            )
            specification = prepared["specification"]
            patch.setitem(
                settings.DIAGNOSTIC_DATASETS,
                "kinetics-subset",
                replace(
                    specification, kaggle_handle=specification.kaggle_handle[:-1] + "2"
                ),
            )
            newer = prepare_data(prepared["root"], dataset.dataset_dir)
            with pytest.raises(ValueError, match="incompatible"):
                workflows.load_checkpoint(newer, trained["selected_checkpoint"])
            patch.setitem(
                settings.DIAGNOSTIC_DATASETS, "kinetics-subset", specification
            )
            patch.setattr(
                settings, "CLASSES_OF_INTEREST", tuple(dataset.class_names[1:])
            )
            reduced = prepare_data(prepared["root"], dataset.dataset_dir)
            with pytest.raises(ValueError):
                workflows.load_checkpoint(reduced, trained["selected_checkpoint"])


def test_complete_screen_fine_training_and_test_gate(prepared, monkeypatch):
    workflows.show_experiment_plan(prepared)
    screen = workflows.run_screen(prepared)
    screen_path = screen / "screen.json"
    original_screen = screen_path.read_text()
    declared = json.loads((screen / "plan.json").read_text())
    assert len(declared["candidates"]) == 16
    assert declared["candidate_data_identities"]["small_s12_t2_wd0p0001_aug1"][
        "frame_size"
    ] == [12, 12]
    assert (
        declared["candidate_data_identities"]["small_s8_t3_wd0p0001_aug0"][
            "sequence_length"
        ]
        == 3
    )
    assert {
        identity["split_manifest_hash"]
        for identity in declared["candidate_data_identities"].values()
    } == {prepared["split"]["manifest_hash"]}
    incomplete = json.loads(original_screen)
    incomplete["ranked"].pop()
    screen_path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="Every declared candidate"):
        workflows.checked_winner(screen)
    screen_path.write_text(original_screen)
    changed_shape = json.loads(original_screen)
    changed_shape["ranked"][0]["data_identity"]["sequence_length"] += 1
    screen_path.write_text(json.dumps(changed_shape))
    with pytest.raises(ValueError, match="provenance"):
        workflows.checked_winner(screen)
    screen_path.write_text(original_screen)
    reversed_ranking = json.loads(original_screen)
    reversed_ranking["ranked"].reverse()
    screen_path.write_text(json.dumps(reversed_ranking))
    with pytest.raises(ValueError, match="validation ranking"):
        workflows.checked_winner(screen)
    screen_path.write_text(original_screen)
    with pytest.raises(FileNotFoundError):
        workflows.final_evaluate(prepared, screen)
    fine = workflows.train_winner(prepared, screen)
    winner, _ = workflows.checked_winner(screen)
    assert fine["dataset"].frame_size == (
        winner["config"]["height"],
        winner["config"]["width"],
    )
    assert fine["dataset"].sequence_length == winner["config"]["sequence_length"]
    assert fine["dataset"].allowed_indices.isdisjoint(fine["test"].indices)
    # Wrong preprocessing is rejected without opening a test clip.
    with monkeypatch.context() as patch:
        patch.setattr(prepared["dataset"], "frame_size", (7, 7))
        with pytest.raises(ValueError, match="incompatible"):
            workflows.final_evaluate(prepared, screen)

    def fail_export(*args, **kwargs):
        raise RuntimeError("prediction export failed")

    with monkeypatch.context() as patch:
        patch.setattr(workflows, "prediction_examples", fail_export)
        with pytest.raises(RuntimeError, match="prediction export failed"):
            workflows.final_evaluate(fine, screen)
    assert (screen / "final_test_metrics.json").is_file()
    assert not (screen / "final_test.json").exists()
    assert fine["dataset"].allowed_indices.isdisjoint(fine["test"].indices)

    def fail_display(*args, **kwargs):
        raise TypeError("frontend display failed")

    with monkeypatch.context() as patch:
        patch.setattr(
            workflows,
            "evaluate_classifier",
            lambda *a, **kw: pytest.fail("metrics recomputed"),
        )
        patch.setattr(workflows, "show_predictions", fail_display)
        with pytest.raises(TypeError, match="frontend display failed"):
            workflows.final_evaluate(fine, screen)
    assert (screen / "final_test.json").is_file()
    assert fine["dataset"].allowed_indices.isdisjoint(fine["test"].indices)
    monkeypatch.setattr(
        NotebookVideoDataset, "__getitem__", lambda *a: pytest.fail("test reopened")
    )
    report = workflows.final_evaluate(fine, screen)
    assert report["partition"] == "test"
    assert {r["partition"] for r in report["examples"]} == {"test"}
    assert 0 <= report["metrics"]["accuracy"] <= 1
    assert fine["dataset"].allowed_indices.isdisjoint(fine["test"].indices)
    monkeypatch.setattr(
        NotebookVideoDataset, "__getitem__", lambda *a: pytest.fail("test reopened")
    )
    assert workflows.final_evaluate(fine, screen) == report
    with pytest.raises(ValueError, match="already frozen"):
        workflows.train_winner(prepared, screen)
    frozen = json.loads((screen / "frozen.json").read_text())
    frozen["result"]["config"]["height"] += 4
    (screen / "frozen.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="compatible"):
        workflows.final_evaluate(fine, screen)


def test_grid_includes_every_model_and_factor_combination(tmp_path):
    from itertools import product
    from types import SimpleNamespace

    prepared = {
        "root": tmp_path,
        "dataset": SimpleNamespace(
            dataset_dir=tmp_path, sequence_length=16, frame_size=(32, 32)
        ),
        "manifest_path": tmp_path / "split.json",
    }
    configs = workflows.screen_configurations(prepared)
    assert len(settings.SCREEN_CANDIDATES) == 16
    assert len(configs) == 12
    assert {config.epochs for config in configs.values()} == {4}
    assert {config.batch_size for config in configs.values()} == {4}
    assert {config.height for config in configs.values()} == {64, 128}
    assert {config.sequence_length for config in configs.values()} == {16}
    assert {config.weight_decay for config in configs.values()} == {0.0, 0.0001}
    actual = {
        (
            config.convlstm_layers,
            config.height,
            config.sequence_length,
            config.weight_decay,
            config.augment,
        )
        for config in configs.values()
    }
    expected = set(
        product(
            [settings.SCREEN_CANDIDATES[name] for name in settings.SCREEN_MODEL_NAMES],
            settings.SCREEN_FRAME_SIZES,
            settings.SCREEN_SEQUENCE_LENGTHS,
            settings.SCREEN_WEIGHT_DECAYS,
            settings.SCREEN_AUGMENT_OPTIONS,
        )
    )
    assert actual == expected
    assert len(actual) == len(configs)
    assert configs == workflows.screen_configurations(prepared)


@pytest.mark.parametrize("values", [[], [8, 8]])
def test_grid_rejects_empty_or_duplicate_sizes(tmp_path, monkeypatch, values):
    from types import SimpleNamespace

    prepared = {
        "root": tmp_path,
        "dataset": SimpleNamespace(
            dataset_dir=tmp_path, sequence_length=16, frame_size=(32, 32)
        ),
        "manifest_path": tmp_path / "split.json",
    }
    monkeypatch.setattr(settings, "SCREEN_FRAME_SIZES", values)
    with pytest.raises(ValueError, match="nonempty list of distinct"):
        workflows.screen_configurations(prepared)


def test_confirmation_preserves_selected_inputs_and_regularisation():
    from src.experiment_config import ExperimentConfig

    winner = {
        "config": ExperimentConfig(
            height=96, width=96, sequence_length=64, augment=True, weight_decay=0.001
        ).to_dict()
    }
    selected = workflows.final_training_configuration(
        winner, {"final_input_policy": "selected", "final_epochs": 128}
    )
    assert (selected.height, selected.width, selected.sequence_length) == (96, 96, 64)
    assert (selected.augment, selected.weight_decay, selected.epochs) == (
        True,
        0.001,
        128,
    )
    legacy = workflows.final_training_configuration(
        winner,
        {"final_frame_size": 128, "final_sequence_length": 32, "final_epochs": 128},
    )
    assert (legacy.height, legacy.width, legacy.sequence_length) == (128, 128, 32)


def test_no_corrupt_fallback_and_bounded_root_lookup(prepared, monkeypatch):
    dataset = prepared["dataset"]
    index = prepared["train"].indices[0]
    dataset.samples[index][0].write_bytes(b"broken video")
    with pytest.raises(ValueError, match="no sample was substituted"):
        dataset[index]
    monkeypatch.setattr(
        Path, "rglob", lambda *a: pytest.fail("recursive full-tree scan")
    )
    assert resolve_vdd_root(dataset.dataset_dir) == dataset.dataset_dir


def test_live_curves_update_the_existing_display(tmp_path, monkeypatch):
    class Handle:
        updates = 0

        def update(self, figure):
            self.updates += 1

    handle = Handle()
    monkeypatch.setattr(visuals, "display", lambda *a, **kw: handle)
    curves = visuals.LiveCurves()
    history = [
        dict(
            epoch=1,
            train=dict(loss=1.0, accuracy=0.4),
            validation=dict(loss=1.1, accuracy=0.3),
        )
    ]
    curves(history, tmp_path)
    curves(history + [dict(history[0], epoch=2)], tmp_path)
    assert handle.updates == 1
    assert (tmp_path / "learning_curves.png").is_file()


def test_video_card_validates_path(tmp_path):
    with pytest.raises(ValueError, match="path is missing"):
        visuals.video_card(None, "Missing")
    with pytest.raises(FileNotFoundError, match="Video not found"):
        visuals.video_card(tmp_path / "missing.mp4", "Missing")


def test_notebook_contracts_and_reference_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert len(NOTEBOOKS) == 4
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        )
        compile(source, str(path), "exec")
        if path.name.startswith("04_"):
            assert "prepare_experiment_data(IMPLEMENTATION_ROOT)" in source
            assert "from notebooks.utils.suite import prepare_experiment_data" in source
            assert "from notebooks.utils.suite import (" in source
            assert "show_experiment_plan" in source
        elif path.name.startswith("03_"):
            assert "prepare_training_data(IMPLEMENTATION_ROOT)" in source
            assert (
                "from notebooks.utils.workflows import prepare_training_data" in source
            )
        else:
            assert "prepare_data(IMPLEMENTATION_ROOT)" in source
        assert "RUN_" not in source and "controlled_stage_command" not in source
        assert "requirements.txt" in source
    reference = root / "notebooks" / "aad_experiment_workflow.ipynb"
    assert (
        hashlib.sha256(reference.read_bytes()).hexdigest()
        == "cbe430531729e6d0444c783cea799b99467928ab5964a526adb58dad986b078b"
    )
