"""Exercise all four workflows on real tiny videos, without Kaggle or CUDA."""

import hashlib
import json
from pathlib import Path

import av
import pytest
import torch

from src import notebook_config as settings
from src import notebook_display as visuals
from src import notebook_workflows as workflows
from src.notebook_data import NotebookVideoDataset, prepare_data
from src.model import CustomConvLSTM
from src.utils import seed_everything, write_video_torchvision
from src.vdd_diagnostic import resolve_vdd_root


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(settings, "SEQUENCE_LENGTH", 2)
    monkeypatch.setattr(settings, "TARGET_FPS", 8)
    monkeypatch.setattr(settings, "FRAME_SIZE", 8)
    monkeypatch.setattr(settings, "FINAL_FRAME_SIZE", 12)
    monkeypatch.setattr(settings, "BATCH_SIZE", 8)
    monkeypatch.setattr(settings, "TRAIN_EPOCHS", 1)
    monkeypatch.setattr(settings, "SCREEN_EPOCHS", 1)
    monkeypatch.setattr(settings, "FINAL_EPOCHS", 2)
    monkeypatch.setattr(settings, "DEFAULT_LAYERS", ((2, (3, 3)),))
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
    dataset_root = tmp_path / "vdd"
    for label, name in enumerate(("non-violent", "violent")):
        for index in range(12):
            clip = torch.zeros(12, 3, 16, 16)
            clip[:, label, :, :8] = 0.8
            clip[:, label, :, 8:] = 0.2 + index / 100
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
    with monkeypatch.context() as patch:
        patch.setattr(visuals, "display", cards.append)
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
    checkpoint = torch.load(
        trained["selected_checkpoint"], map_location="cpu", weights_only=True
    )
    seed_everything(settings.SEED)
    initial = CustomConvLSTM(num_classes=2, layers=list(settings.DEFAULT_LAYERS))
    assert any(
        not torch.equal(value, checkpoint["model_state_dict"][name])
        for name, value in initial.state_dict().items()
    )
    examples = workflows.show_validation_predictions(prepared, trained)
    assert {r["partition"] for r in examples} == {"validation"}
    assert Path(trained["run_dir"], "learning_curves.png").is_file()
    assert dataset.allowed_indices.isdisjoint(prepared["test"].indices)
    assert dataset.cache_bytes <= dataset.cache_limit_bytes


def test_complete_screen_fine_training_and_test_gate(prepared, monkeypatch):
    workflows.show_experiment_plan(prepared)
    screen = workflows.run_screen(prepared)
    screen_path = screen / "screen.json"
    original_screen = screen_path.read_text()
    incomplete = json.loads(original_screen)
    incomplete["ranked"].pop()
    screen_path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="Every declared candidate"):
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
    assert fine["dataset"].frame_size == (12, 12)
    assert fine["dataset"].allowed_indices.isdisjoint(fine["test"].indices)
    # Wrong preprocessing is rejected without opening a test clip.
    with pytest.raises(ValueError, match="incompatible"):
        workflows.final_evaluate(prepared, screen)
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
    frozen["result"]["config"]["height"] = 8
    (screen / "frozen.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="compatible"):
        workflows.final_evaluate(fine, screen)


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


def test_notebook_contracts_and_reference_unchanged():
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("0*.ipynb"))
    assert len(notebooks) == 4
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        )
        compile(source, str(path), "exec")
        assert "prepare_data(IMPLEMENTATION_ROOT)" in source
        assert "RUN_" not in source and "controlled_stage_command" not in source
        assert "requirements.txt" in source
    reference = root / "notebooks" / "aad_experiment_workflow.ipynb"
    assert (
        hashlib.sha256(reference.read_bytes()).hexdigest()
        == "cbe430531729e6d0444c783cea799b99467928ab5964a526adb58dad986b078b"
    )
