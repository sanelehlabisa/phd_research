"""Merged Kinetics source and expanded-suite contracts; no real downloads/GPU."""

from dataclasses import replace
from pathlib import Path

import av
import pytest

from test_kinetics600_subset import add_activity_clips, write_trusted_source
from notebooks.utils import kinetics600_subset as kinetics
from notebooks.utils import config as settings
from notebooks.utils import suite
from notebooks.utils.data import data_identity, prepare_data
from notebooks.utils.workflows import prepare_training_data


@pytest.mark.parametrize("selected,expected", [(64, 96), (96, 96), (128, 128)])
def test_confirmation_never_shrinks_selected_resolution(selected, expected):
    config = suite.ExperimentConfig(height=selected, width=selected, epochs=24)
    final = suite.confirmation_configuration(
        dict(final_size=96, final_epochs=64), dict(config=config.to_dict())
    )
    assert final.height == final.width == expected
    assert final.epochs == final.early_stopping_patience == 64
    assert final.weight_decay == config.weight_decay
    assert final.sequence_length == config.sequence_length


def test_count_gate_rejects_2000_and_accepts_2001_without_decoding(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(av, "open", lambda *a, **k: pytest.fail("decoded video"))
    for label in kinetics.KINETICS600_CLASSES:
        directory = tmp_path / label
        directory.mkdir()
        for index in range(400):
            (directory / f"{index:011d}.mp4").write_bytes(f"{label}-{index}".encode())
    with pytest.raises(
        ValueError, match="2000 usable unique clips after exact duplicate removal"
    ):
        kinetics.inventory_clips(tmp_path)
    (tmp_path / kinetics.KINETICS600_CLASSES[0] / "00000000400.mp4").write_bytes(
        b"extra"
    )
    assert len(kinetics.inventory_clips(tmp_path)) == 2001


def test_suite_views_keep_kinetics_source_grouping_and_test_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(kinetics, "MIN_CLIPS", 50)
    monkeypatch.setattr(settings, "SELECTED_DIAGNOSTIC_DATASET", "kinetics600-subset")
    monkeypatch.setattr(
        av, "open", lambda *a, **k: pytest.fail("decoded held-out video")
    )
    dataset_root = tmp_path / "kinetics-600-train"
    files = add_activity_clips(dataset_root)
    write_trusted_source(dataset_root, files)
    preview = prepare_data(tmp_path, dataset_root)
    training = prepare_training_data(tmp_path, dataset_root)
    assert training["dataset"].sequence_length == 32
    assert training["dataset"].frame_size == (96, 96)
    assert training["dataset"].target_fps == 8
    assert training["manifest_path"] == preview["manifest_path"]
    assert training["source"] == preview["source"]
    assert training["dataset"].class_names == preview["dataset"].class_names
    for partition in ("train", "validation", "test"):
        assert training[partition].indices == preview[partition].indices
    assert training["dataset"].allowed_indices.isdisjoint(training["test"].indices)
    prepared = suite.prepare_experiment_data(tmp_path, dataset_root)
    assert preview["dataset"].sequence_length == 16
    assert preview["dataset"].frame_size == (32, 32)
    assert prepared["dataset"].sequence_length == 32
    assert prepared["dataset"].frame_size == (64, 64)
    assert prepared["dataset"].target_fps == 8
    assert preview["manifest_path"] == prepared["manifest_path"]
    config = suite.configuration(prepared, settings.SUITE_SCREEN_EPOCHS)
    for size in settings.SUITE_SPATIAL_SIZES:
        for fps in settings.TEMPORAL_FPS:
            view = suite._data(prepared, replace(config, height=size, width=size), fps)
            assert view["split"]["assignment_unit"] == "source_video"
            for partition in ("train", "validation", "test"):
                assert view[partition].indices == prepared[partition].indices
            assert view["dataset"].allowed_indices.isdisjoint(view["test"].indices)
            expected = dict(
                data_identity(prepared), frame_size=[size, size], target_fps=fps
            )
            assert data_identity(view) == expected

    # Use the real candidate catalogue without reading any dataset video.
    prepared["root"] = Path(__file__).resolve().parents[1]
    plan = suite.suite_plan(prepared)
    assert plan["run_counts"] == dict(
        screen=14,
        one_factor=8,
        native_attempt=1,
        confirmation=8,
        total=31,
        tiny_checks=1,
    )
    assert plan["screen_config"]["epochs"] == 24 and plan["final_epochs"] == 64
    assert plan["hours"] == 8 and plan["spatial_sizes"] == [64, 96, 128]
    assert plan["weight_decay_trials"] == [0.0, 0.001]
    assert plan["screen_config"]["weight_decay"] == 0.0001
    monkeypatch.setattr(suite, "display", lambda *a: None)
    assert suite.show_experiment_plan(prepared) == plan

    changed = dict(prepared, split=dict(prepared["split"], manifest_hash="changed"))
    monkeypatch.setattr(suite, "prepare_data", lambda *a, **k: changed)
    with pytest.raises(ValueError, match="Input view changed"):
        suite._data(prepared, config, 8)
