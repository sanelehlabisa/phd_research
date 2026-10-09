"""Ticket 074: recipe transfer, durable numbering and processed-cache reuse."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import (
    comparison_recipe as recipe,
    dataset,
    study_cache,
    study_matrix as matrix,
)
from src.experiment_config import ExperimentConfig
from src.utils import write_json
from test_capacity_search import protocol, fake_train  # noqa: F401


def test_search_proposal_uses_exact_complete_resolution_evidence(tmp_path):
    from src.capacity_search import job_key

    state = {"stages": [], "jobs": []}
    for wd, correct in ((0, 1), (0.0001, 2), (0.001, 1)):
        rows = []
        for size in (32, 48, 64):
            row = {
                "model": "custom_4",
                "minimum_epochs": 64,
                "config": {"weight_decay": wd, "height": size},
            }
            rows.append(row)
            result = {
                "validation_confusion_matrix": {
                    "matrix": [[correct, 3 - correct], [0, 0]]
                },
                "validation_metrics": {"accuracy": correct / 3, "loss": 0.2},
            }
            state["jobs"].append({"job_id": job_key(row), "result": result})
        state["stages"].append(
            {"name": "flat" if wd == 0 else "weight_decay", "rows": rows}
        )
    write_json(tmp_path / "progress.json", state)
    write_json(tmp_path / "decisions.json", {"learning_rates": {"custom": 0.003}})
    selected = {
        "source_group": str(tmp_path),
        "top3": [{"candidate": {"name": "custom_4"}}],
    }
    assert recipe.search_proposal(selected) == {
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
    }
    state["stages"][-1]["rows"].pop()
    write_json(tmp_path / "progress.json", state)
    with pytest.raises(ValueError, match="complete top-one"):
        recipe.search_proposal(selected)


@pytest.fixture
def transfer(protocol, monkeypatch):
    p = protocol
    monkeypatch.setattr(
        recipe,
        "search_proposal",
        lambda selected: {"learning_rate": 0.003, "weight_decay": 0.001},
    )
    config = ExperimentConfig(
        dataset_dir=str(p.data),
        dataset_name="aad",
        split_manifest=p.values["dataset"]["split_manifest"],
        runs_dir=str(p.root / "runs"),
        height=50,
        width=50,
        sequence_length=50,
        batch_size=1,
        epochs=512,
        cache_dataset=True,
        learning_rate=0.001,
        weight_decay=0.0001,
    )
    selected = {
        "protocol": "capacity_top3_v2",
        "source_group": str(p.root),
        "top3": [
            {
                "candidate": {
                    "name": "custom_4",
                    "convlstm_layers": [[4, [3, 3]]],
                    "hidden_classifier_width": None,
                }
            }
        ],
    }
    study = {"selected_config": str(write_json(p.root / "selection.json", selected))}
    split = {
        "manifest_path": config.split_manifest,
        "manifest_hash": matrix.file_hash(config.split_manifest),
    }
    return SimpleNamespace(
        p=p,
        config=config,
        selected=selected,
        study=study,
        split=split,
        group=p.root / "runs/studies/comparison",
    )


def scored_train(t, prefer_lr=True):
    train = fake_train(t.p)

    def run(args):
        folder = train(args)
        result = matrix.read_json(folder / "summary.json")["all"][0]
        cfg = result["experiment_config"]
        # Loss-only ordering makes exact-count ties explicit and reproducible.
        loss = (0.2 if prefer_lr else 0.8) if cfg["learning_rate"] == 0.003 else 0.5
        if cfg["weight_decay"] == 0.001:
            loss -= 0.1
        records = matrix.read_json(result["validation_predictions"])
        for r in records:
            r["loss"] = loss
        write_json(result["validation_predictions"], records)
        result["validation_metrics"]["loss"] = loss
        write_json(folder / "summary.json", {"all": [result]})
        return folder

    return run


def select(t, callback):
    return recipe.select_recipe(
        t.study, t.config, t.selected, t.group, t.split, callback, 170
    )


@pytest.mark.parametrize("prefer_lr", [True, False])
def test_native_one_factor_transfer_freeze_and_resume(transfer, prefer_lr):
    t = transfer
    config, report = select(t, scored_train(t, prefer_lr))
    assert config.learning_rate == (0.003 if prefer_lr else 0.001)
    assert config.weight_decay == 0.001 and config.epochs == 512
    assert config.sequence_length == config.height == config.width == 50
    assert [j["experiment_number"] for j in report["jobs"]] == [171, 172, 173]
    values = [cfg for _, cfg in t.p.trained]
    assert all(cfg["epochs"] == 128 and cfg["cache_dataset"] for cfg in values)
    assert values[0]["weight_decay"] == values[1]["weight_decay"] == 0.0001
    assert values[2]["learning_rate"] == config.learning_rate
    assert select(t, lambda _: pytest.fail("complete recipe must not retrain")) == (
        config,
        report,
    )
    assert len(t.p.trained) == 3


def test_identical_settings_reuse_one_experiment(transfer, monkeypatch):
    t = transfer
    monkeypatch.setattr(
        recipe,
        "search_proposal",
        lambda _: {"learning_rate": 0.001, "weight_decay": 0.0001},
    )
    _, report = select(t, scored_train(t))
    assert len(t.p.trained) == len(report["jobs"]) == 1
    assert all(
        s["incumbent"] == s["challenger"] == s["selected"] == 171
        for s in report["steps"]
    )


def test_recipe_tampering_and_missing_receipt_never_retrain(transfer):
    t = transfer
    _, report = select(t, scored_train(t))
    path = t.group / "recipe_selection.json"
    write_json(path, {**report, "selected": {"learning_rate": 99, "weight_decay": 99}})
    with pytest.raises(ValueError, match="recipe changed"):
        select(t, lambda _: pytest.fail("must not retrain"))
    write_json(path, report)
    receipt = Path(report["jobs"][0]["receipt"])
    receipt.unlink()
    with pytest.raises(ValueError, match="evidence is missing"):
        select(t, lambda _: pytest.fail("must not retrain"))


def test_interrupted_recipe_is_not_repeated(transfer):
    t = transfer

    def fail(args):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        select(t, fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        select(t, lambda _: pytest.fail("failed job must not retry"))


def test_disk_cache_revisit_new_process_metadata_invalidation_and_test_lock(
    protocol, monkeypatch
):
    p = protocol
    source = dataset.AHARDataset(p.data, 2, (8, 8))
    train, val, test, split = dataset.load_split_subsets(
        source, p.values["dataset"]["split_manifest"]
    )
    allowed = {source.samples[i][0] for i in train.indices + val.indices}
    seen = []

    def decode(self, path):
        assert path in allowed
        seen.append(path)
        return torch.ones(self.sequence_length, 3, *self.frame_size)

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", decode)
    study_cache._CACHE.clear()
    cache = p.root / "runs/cache/training"

    def get(src=source):
        return study_cache.cached_training_dataset(
            src, train.indices, val.indices, split["manifest_hash"], cache
        )

    first, initial = get()
    assert len(seen) == len(allowed) and initial["test_clips"] == 0
    second, ram = get()
    assert second is first and ram["reuse_source"] == "ram"
    other = dataset.AHARDataset(p.data, 2, (10, 10))
    get(other)
    count = len(seen)
    restored, disk = get()
    assert disk["reused"] and disk["reuse_source"] == "disk" and len(seen) == count
    with pytest.raises(KeyError):
        restored[test.indices[0]]
    study_cache._CACHE.clear()  # Simulate a new command/process.
    assert get()[1]["reuse_source"] == "disk" and len(seen) == count
    changed = source.samples[train.indices[0]][0]
    changed.write_bytes(b"changed source metadata")
    _, invalidated = get()
    assert invalidated["key"] != initial["key"] and len(seen) > count
    study_cache._CACHE.clear()
    Path(invalidated["disk_path"]).write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="corrupt"):
        get()


def test_disk_budget_falls_back_without_deleting_artifacts(protocol, monkeypatch):
    p = protocol
    source = dataset.AHARDataset(p.data, 2, (8, 8))
    monkeypatch.setattr(
        dataset.AHARDataset, "_load_video", lambda self, path: torch.zeros(2, 3, 8, 8)
    )
    monkeypatch.setattr(study_cache, "MAX_DISK_BYTES", 1)
    study_cache._CACHE.clear()
    cache = p.root / "runs/cache/training"
    _, report = study_cache.cached_training_dataset(source, [0], [1], "split", cache)
    assert "disk_skip_reason" in report and not list(cache.glob("*.pt"))
    assert report["bytes"] <= study_cache.MAX_CACHE_BYTES


def test_bad_training_clip_never_falls_through_to_another_partition(
    protocol, monkeypatch
):
    p = protocol
    source = dataset.AHARDataset(p.data, 2, (8, 8))
    seen = []

    def decode(self, path):
        seen.append(path)
        raise ValueError("bad source clip")

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", decode)
    study_cache._CACHE.clear()
    with pytest.raises(ValueError, match="bad source"):
        study_cache.cached_training_dataset(source, [0], [1], "split")
    assert seen == [source.samples[0][0]]
