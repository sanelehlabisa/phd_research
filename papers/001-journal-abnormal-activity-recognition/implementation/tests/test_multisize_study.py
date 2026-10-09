"""Ticket 068: matrix completeness, provenance, restart/test locks and ZIP delivery."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from src import (
    dataset,
    evaluate,
    experiments,
    study_cache,
    study_config,
    study_matrix as matrix,
)
from src.utils import write_json, write_video_torchvision
from notebooks.utils import aad_study, study_archive

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "configs/experiments/aad_custom_search_colab.json"
COMPARISON = ROOT / "configs/experiments/aad_model_comparison_colab.json"


@pytest.fixture
def protocol(tmp_path_factory):
    root = tmp_path_factory.mktemp("m068")
    data = root / "aad"
    for label in ("a", "b"):
        (data / label).mkdir(parents=True)
        for index in range(10):
            (data / label / f"{index}.mp4").write_bytes(b"metadata-only fixture")
    values = json.loads(SEARCH.read_text())
    values["dataset"].update(path=str(data), split_manifest=str(root / "split.json"))
    values["runs_dir"] = str(root / "runs")
    path = write_json(root / "search.json", values)
    return SimpleNamespace(
        root=root, data=data, values=values, path=path, trained=[], tested=[]
    )


def scores(accuracy):
    return {
        "loss": 1 - accuracy,
        "accuracy": accuracy,
        "precision": accuracy,
        "recall": accuracy,
        "f1": accuracy,
    }


def fake_train(p):
    def run(args):
        config = matrix.read_json(args[args.index("--config") + 1])
        name = args[args.index("--model") + 1]
        folder = p.root / "runs/experiments" / f"leaf{len(p.trained):02d}"
        folder.mkdir(parents=True)
        checkpoint = folder / "best_model.pth"
        checkpoint.write_bytes(b"synthetic checkpoint")
        split = {
            "manifest_path": config["split_manifest"],
            "manifest_hash": matrix.file_hash(config["split_manifest"]),
        }
        accuracy = 0.8 if name == "custom_top2" else 0.6
        if name == "swin3d_s":
            accuracy = 0.99  # Best overall must NOT become the custom example model.
        selection = {"selected_epoch": 1, "selection_partition": "validation"}
        sidecar = {
            "checkpoint_role": "validation_selected_lowest_loss",
            "selection_partition": "validation",
            "experiment_config": config,
            "split_manifest_hash": split["manifest_hash"],
            "selected_epoch": 1,
            "seed": 42,
            "model_registry_entry": {"name": name},
        }
        write_json(checkpoint.with_suffix(".json"), sidecar)
        history = write_json(folder / "history.json", {"epochs": [{"epoch": 1}]})
        curves = folder / "curves.png"
        curves.write_bytes(b"synthetic curve")
        predictions = write_json(folder / "validation_predictions.json", [])
        result = {
            "name": name,
            "seed": 42,
            "partition": "validation",
            "test_access": "locked",
            "dataset_dir": config["dataset_dir"],
            "experiment_config": config,
            "split": split,
            "num_params": sum(c[0] for c in config["convlstm_layers"]),
            "family": "ConvLSTM" if name.startswith("custom") else "baseline",
            "validation_metrics": scores(accuracy),
            "selected_checkpoint": str(checkpoint),
            "checkpoint_selection": selection,
            "early_stopping": {"actual_epochs": 1},
            "history": str(history),
            "training_curves": str(curves),
            "train_time_s": 1,
            "weight_tensor_bytes": 32,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "validation_metric_pass_seconds": 0.1,
            "validation_predictions": str(predictions),
        }
        write_json(folder / "summary.json", {"all": [result]})
        write_json(folder / "run.json", {"status": "complete"})
        p.trained.append(name)
        return folder

    return run


def run_search(p):
    return study_config.execute_study(
        p.path, *study_config.load_study(p.path), fake_train(p)
    )


def comparison_path(p, search):
    values = matrix.read_json(COMPARISON)
    selected = matrix.load_selection(search / "selected_config.json")
    values["selected_config"] = str(search / "selected_config.json")
    values["dataset"].update(
        path=str(p.data), split_manifest=selected["split"]["manifest_path"]
    )
    values["runs_dir"] = str(p.root / "runs")
    return write_json(p.root / "comparison.json", values)


def fake_test(p):
    def run(path):
        config = matrix.read_json(path)
        freeze = Path(path).parent.parent / "validation_frozen.json"
        frozen = matrix.read_json(freeze)
        assert len(frozen["models"]) == 8 and frozen["best_custom"] == "custom_top2"
        assert len(p.trained) == 44  # All 36 search + 8 comparison BEFORE any test.
        folder = p.root / "runs/evaluate" / Path(path).stem
        folder.mkdir(parents=True)
        source = dataset.AHARDataset(
            config["dataset_dir"],
            config["sequence_length"],
            (config["width"], config["height"]),
        )
        _, _, test, split = dataset.load_split_subsets(source, config["split_manifest"])
        records = [
            {
                "source": str(source.samples[i][0]),
                "partition": "test",
                "target": source.samples[i][1],
                "predicted": source.samples[i][1],
                "correct": True,
                "loss": 0.0,
                "confidence": 1.0,
            }
            for i in test.indices
        ]
        predictions = write_json(folder / "predictions.json", records)
        report = {
            "partition": "test",
            "split": split,
            "checkpoint": {"path": config["checkpoint_path"]},
            "experiment_config": {
                k: v for k, v in config.items() if k != "checkpoint_path"
            },
            "predictions": str(predictions),
            "metrics": scores(1),
            "metric_pass_seconds": 0.1,
            "prediction_examples": {
                "records": [],
                "saved_counts": {"correct": 0, "incorrect": 0},
            },
        }
        write_json(folder / "metrics/final.json", report)
        write_json(folder / "run.json", {"status": "complete"})
        p.tested.append(Path(path).stem)
        return folder

    return run


def test_exact_matrix_handoff_frozen_comparison_and_safe_rerun(protocol):
    p = protocol
    search = run_search(p)
    bundle = matrix.load_selection(search / "selected_config.json")
    assert (
        len(p.trained) == 36
        and len(bundle["ranking"]) == 12
        and len(bundle["top3"]) == 3
    )
    assert all(len(r["per_resolution"]) == 3 for r in bundle["ranking"])
    assert all(len(c["sources"]) == 3 for c in bundle["top3"])
    path = comparison_path(p, search)
    study, config, manifest = study_config.load_study(path)
    rows = study_config.study_rows(study, config)
    assert len(rows) == 8 and config.epochs == 256
    assert all(
        (
            r["config"]["sequence_length"],
            r["config"]["height"],
            r["config"]["batch_size"],
            r["minimum_epochs"],
        )
        == (50, 50, 1, 32)
        for r in rows
    )
    group = study_config.execute_study(
        path, study, config, manifest, fake_train(p), fake_test(p)
    )
    report = matrix.read_json(group / "comparison.json")
    assert len(report["table"]) == len(p.tested) == 8
    assert report["validation_frozen"]["best_custom"] == "custom_top2"
    for receipt in (group / "test_configs").glob("*.json"):
        assert matrix.read_json(receipt)["prediction_samples_per_category"] == (
            3 if receipt.stem == "custom_top2" else 0
        )
    assert all(
        "test_micro_f1" in r and "validation_micro_precision" in r
        for r in report["table"]
    )
    assert (group / "comparison.csv").is_file()
    # The same request verifies receipts and returns; no extra train or test calls.
    assert run_search(p) == search
    assert (
        study_config.execute_study(
            path, study, config, manifest, fake_train(p), fake_test(p)
        )
        == group
    )
    assert len(p.trained) == 44 and len(p.tested) == 8
    for _, profile in aad_study.PROFILES:
        write_json(p.root / profile, matrix.read_json(ROOT / profile))
    archive_dir = p.root / "runs/notebook_studies/export"
    progress = write_json(archive_dir / "progress.json", {"stages": []})
    archive = study_archive.make_archive(
        p.root, archive_dir / "artifacts.zip", progress, {group}, aad_study.PROFILES
    )
    entries = study_archive.verify_archive(archive)
    assert sum(name.endswith("/best_model.pth") for name in entries) == 44
    assert sum(name.endswith("/metrics/final.json") for name in entries) == 8
    assert sum(name.startswith("runs/study_receipts/") for name in entries) == 2
    assert any(name.endswith("/selected_config.json") for name in entries)
    assert any(name.endswith("/comparison.csv") for name in entries)
    assert not any(name.startswith("aad/") for name in entries)


def test_equal_resolution_weighting_missing_duplicates_protocol_and_ties(protocol):
    group = run_search(protocol)
    records = [j["result"] for j in matrix.read_json(group / "progress.json")["jobs"]]
    names = protocol.values["models"]
    for r in records:
        size = r["experiment_config"]["height"]
        r["validation_metrics"] = scores(
            0.99 if r["name"] == names[0] and size == 32 else 0.1
        )
        if r["name"] == names[1]:
            r["validation_metrics"] = scores(0.6)
    ranking = matrix.rank_resolutions(records, names, [32, 48, 64])
    assert ranking[0]["name"] == names[1]
    assert (
        next(r for r in ranking if r["name"] == names[0])["worst_validation_accuracy"]
        == 0.1
    )
    with pytest.raises(ValueError, match="every architecture"):
        matrix.rank_resolutions(records[:-1], names, [32, 48, 64])
    with pytest.raises(ValueError, match="every architecture"):
        matrix.rank_resolutions([*records[:-1], records[0]], names, [32, 48, 64])
    records[0]["experiment_config"]["batch_size"] = 99
    with pytest.raises(ValueError, match="fixed training"):
        matrix.rank_resolutions(records, names, [32, 48, 64])


def test_selection_rejects_tampering_and_partial_search(protocol):
    group = run_search(protocol)
    path = group / "selected_config.json"
    bundle = matrix.read_json(path)
    bad = copy.deepcopy(bundle)
    bad["top3"][0]["candidate"]["hidden_classifier_width"] = 12
    write_json(path, bad)
    with pytest.raises(ValueError, match="differs"):
        matrix.load_selection(path)
    write_json(path, bundle)
    checkpoint = Path(
        matrix.read_json(group / "progress.json")["jobs"][0]["result"][
            "selected_checkpoint"
        ]
    )
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="missing or modified"):
        matrix.load_selection(path)


def test_failed_training_is_saved_and_never_silently_repeated(protocol):
    p = protocol

    def fail(args):
        if len(p.trained) == 2:
            raise RuntimeError("synthetic training failure")
        return fake_train(p)(args)

    with pytest.raises(RuntimeError, match="synthetic"):
        study_config.execute_study(p.path, *study_config.load_study(p.path), fail)
    group = next((p.root / "runs/studies").iterdir())
    assert matrix.read_json(group / "progress.json")["status"] == "incomplete"
    assert len(matrix.read_json(group / "progress.json")["jobs"]) == 2
    with pytest.raises(RuntimeError, match="never silently retrain"):
        run_search(p)
    assert len(p.trained) == 2 and not (group / "selected_config.json").exists()


def test_failed_test_is_not_repeated_and_freeze_is_preserved(protocol):
    p = protocol
    group = run_search(p)
    path = comparison_path(p, group)

    def fail_test(path):
        assert (Path(path).parent.parent / "validation_frozen.json").is_file()
        raise RuntimeError("test interrupted")

    with pytest.raises(RuntimeError, match="test interrupted"):
        study_config.execute_study(
            path, *study_config.load_study(path), fake_train(p), fail_test
        )
    with pytest.raises(RuntimeError, match="already attempted"):
        study_config.execute_study(
            path, *study_config.load_study(path), fake_train(p), fake_test(p)
        )
    assert len(p.trained) == 44 and not p.tested
    comparison = next(
        d
        for d in (p.root / "runs/studies").iterdir()
        if (d / "validation_frozen.json").exists()
    )
    assert (
        matrix.read_json(comparison / "run.json")["test_access"]
        == "started_after_validation_freeze"
    )


def test_comparison_rejects_unfrozen_input_and_wrong_architecture(protocol):
    group = run_search(protocol)
    path = comparison_path(protocol, group)
    values = matrix.read_json(path)
    values["training"]["batch_size"] = 2
    write_json(path, values)
    with pytest.raises(ValueError, match="batch 1"):
        study_config.load_study(path)
    values["training"]["batch_size"] = 1
    values["custom_candidates"] = [{"name": "fake"}]
    write_json(path, values)
    with pytest.raises(ValueError, match="candidate"):
        study_config.load_study(path)


def test_cache_keys_bounds_and_test_isolation(tmp_path, monkeypatch):
    for cls in ("a", "b"):
        (tmp_path / cls).mkdir()
        for i in range(10):
            (tmp_path / cls / f"{i}.mp4").write_bytes(b"video metadata")
    source = dataset.AHARDataset(tmp_path, 2, (8, 8))
    tr, val, test, split = dataset.load_split_subsets(source, tmp_path / "split.json")
    seen = []

    def decode(self, path):
        assert (
            self.samples.index((path, 0 if path.parent.name == "a" else 1))
            not in test.indices
        )
        seen.append(str(path))
        return torch.zeros(self.sequence_length, 3, *self.frame_size)

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", decode)
    study_cache._CACHE.clear()
    one, first = study_cache.cached_training_dataset(
        source, tr.indices, val.indices, split["manifest_hash"]
    )
    two, second = study_cache.cached_training_dataset(
        source, tr.indices, val.indices, split["manifest_hash"]
    )
    assert one is two and second["reused"] and first["test_clips"] == 0
    assert len(seen) == len(tr) + len(val)
    changed = dataset.AHARDataset(tmp_path, 2, (10, 10))
    three, third = study_cache.cached_training_dataset(
        changed, tr.indices, val.indices, split["manifest_hash"]
    )
    assert three is not one and third["key"] != first["key"]
    assert len(study_cache._CACHE) == 1
    monkeypatch.setattr(study_cache, "MAX_CACHE_BYTES", 1)
    lazy, report = study_cache.cached_training_dataset(
        source, tr.indices, val.indices, split["manifest_hash"]
    )
    assert lazy is source and report["bytes"] == 0 and not study_cache._CACHE


def archive_fixture(tmp_path):
    for _, config in aad_study.PROFILES:
        write_json(tmp_path / config, {})
    study = tmp_path / "runs/notebook_studies/current"
    progress = write_json(study / "progress.json", {"stages": []})
    leaf = tmp_path / "runs/experiments/leaf"
    write_json(leaf / "metrics.json", {"status": "partial"})
    (leaf / "best_model.pth").write_bytes(b"checkpoint")
    (leaf / "cache").mkdir()
    (leaf / "cache/raw.bin").write_bytes(b"exclude")
    return study, progress, leaf


def test_zip_inventory_references_crc_and_independent_retry(tmp_path, monkeypatch):
    study, progress, leaf = archive_fixture(tmp_path)
    group = tmp_path / "runs/studies/group"
    write_json(group / "progress.json", {"jobs": [{"run_dir": str(leaf)}]})
    previous = tmp_path / "runs/notebook_studies/search"
    write_json(previous / "resolved_search.json", {})
    write_json(previous / "batch_benchmark.json", {"selected_batch_size": 16})
    index = write_json(
        tmp_path / "runs/study_receipts/key.json", {"run_dir": str(group)}
    )
    write_json(
        group / "run.json",
        {
            "arguments": {"study_file": str(previous / "resolved_search.json")},
            "resume_index": str(index),
        },
    )
    path = study_archive.make_archive(
        tmp_path, study / "artifacts.zip", progress, {group}, aad_study.PROFILES
    )
    entries = study_archive.verify_archive(path)
    assert "runs/experiments/leaf/best_model.pth" in entries
    assert "runs/notebook_studies/search/batch_benchmark.json" in entries
    assert "runs/study_receipts/key.json" in entries
    assert not any("raw.bin" in name for name in entries)
    calls = []
    monkeypatch.setattr(study_archive, "download_archive", lambda p: calls.append(p))
    monkeypatch.setattr(experiments, "main", lambda *a: pytest.fail("retry trained"))
    monkeypatch.setattr(evaluate, "main", lambda *a: pytest.fail("retry tested"))
    study_archive.retry_download(path)
    assert calls == [path]
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("unexpected.txt", "corrupt inventory")
    with pytest.raises(ValueError, match="inventory"):
        study_archive.retry_download(path)


def test_zip_space_and_write_failures_preserve_evidence(tmp_path, monkeypatch):
    study, progress, leaf = archive_fixture(tmp_path)
    monkeypatch.setattr(
        study_archive.shutil, "disk_usage", lambda p: SimpleNamespace(free=0)
    )
    with pytest.raises(OSError, match="Not enough disk"):
        study_archive.make_archive(
            tmp_path, study / "artifacts.zip", progress, {leaf}, aad_study.PROFILES
        )
    assert (leaf / "best_model.pth").is_file()
    monkeypatch.setattr(
        study_archive.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**10)
    )
    monkeypatch.setattr(
        study_archive,
        "verify_archive",
        lambda p: (_ for _ in ()).throw(ValueError("bad ZIP")),
    )
    with pytest.raises(RuntimeError, match="retry packaging only"):
        study_archive.make_archive(
            tmp_path, study / "artifacts.zip", progress, {leaf}, aad_study.PROFILES
        )
    assert (leaf / "best_model.pth").is_file() and not (
        study / "artifacts.zip"
    ).exists()


def test_download_failure_visible_without_claiming_success(
    tmp_path, monkeypatch, capsys
):
    import sys

    files = SimpleNamespace(
        download=lambda p: (_ for _ in ()).throw(OSError("browser unavailable"))
    )
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(files=files))
    with pytest.raises(RuntimeError, match="Download request failed"):
        study_archive.download_archive(tmp_path / "artifacts.zip")
    files.download = lambda p: None
    assert study_archive.download_archive(tmp_path / "artifacts.zip") == "requested"
    assert "not confirmed" in capsys.readouterr().out


def test_published_model_checkpoint_restores_native_topology_on_meta():
    with torch.device("meta"):
        model = experiments.build_registered_model(
            "paper_convlstm_published", 11, (3, 50, 50), 50
        )
    checkpoint = {
        "model_config": {
            "registry_name": "paper_convlstm_published",
            "num_classes": 11,
            "input_dimensions": {"height": 50, "width": 50, "sequence_length": 50},
        },
        "model_state_dict": model.state_dict(),
    }
    restored = evaluate.model_from_checkpoint(checkpoint)
    assert sum(p.numel() for p in restored.parameters()) == 512197467


def test_cpu_end_to_end_real_search_training_evaluation_and_playable_examples(
    protocol, monkeypatch
):
    """Tiny real ConvLSTM smoke; baseline doubles avoid allocating A100 models on CPU."""
    from src.model import CustomConvLSTM

    p = protocol
    torch.set_num_threads(1)
    for file in p.data.glob("*/*.mp4"):
        level = 0.1 if file.parent.name == "a" else 0.9
        write_video_torchvision(torch.full((3, 3, 12, 12), level), file, 8)
    values = p.values
    values["custom_candidates"] = [
        {
            "name": f"tiny_{i}",
            "research_question": f"Smoke width {i}",
            "convlstm_layers": [[i, [3, 3]]],
            "hidden_classifier_width": None,
        }
        for i in (1, 2, 3)
    ]
    values["models"] = [c["name"] for c in values["custom_candidates"]]
    values["reference_input"].update(frame_size=8, sequence_length=2)
    values["factors"].update(frame_sizes=[8, 10, 12], sequence_lengths=[2])
    values["training"].update(
        epochs=1,
        early_stopping_patience=1,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        prediction_samples_per_category=0,
    )
    values["max_runs"] = 9
    write_json(p.path, values)
    source = dataset.AHARDataset(p.data, 2, (8, 8))
    _, _, test, _ = dataset.load_split_subsets(source, p.root / "split.json")
    test_paths = {source.samples[i][0] for i in test.indices}
    original_decode = dataset.AHARDataset._load_video
    allowed = {"test": False}
    seen_test = []

    def guarded_decode(self, path):
        if path in test_paths:
            assert allowed["test"], "test decoded before validation freeze"
            seen_test.append(path)
        return original_decode(self, path)

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", guarded_decode)
    search = experiments.main(["--config", str(p.path)])
    bundle = matrix.load_selection(search / "selected_config.json")
    assert len(bundle["ranking"]) == 3 and not seen_test
    path = comparison_path(p, search)
    comparison = matrix.read_json(path)
    comparison["training"].update(
        epochs=1, num_workers=0, pin_memory=False, cache_dataset=True
    )
    comparison["minimum_epochs"] = 1
    write_json(path, comparison)

    def tiny_baseline(
        name, num_classes, input_shape, frames, config=None, manifest=None
    ):
        layers = (
            list(config.convlstm_layers)
            if name in matrix.CUSTOM_SLOTS
            else [(1, (3, 3))]
        )
        return CustomConvLSTM(num_classes, layers=layers)

    monkeypatch.setattr(experiments, "build_registered_model", tiny_baseline)
    calls = []

    def gated_evaluate(config_path):
        group = Path(config_path).parent.parent
        freeze = matrix.read_json(group / "validation_frozen.json")
        assert len(freeze["models"]) == 8
        assert len(matrix.read_json(group / "progress.json")["jobs"]) == 8
        assert all(Path(m["checkpoint"]).is_file() for m in freeze["models"])
        allowed["test"] = True
        try:
            result = evaluate.main(["--config", str(config_path)])
            calls.append(result)
            return result
        finally:
            allowed["test"] = False

    group = study_config.execute_study(
        path, *study_config.load_study(path), experiments.main, gated_evaluate
    )
    report = matrix.read_json(group / "comparison.json")
    assert len(calls) == 8 and set(seen_test) == test_paths
    for row in report["table"]:
        records = matrix.read_json(row["test_predictions"])
        accuracy = sum(r["correct"] for r in records) / len(records)
        assert row["test_accuracy"] == pytest.approx(accuracy, abs=1e-6)
        assert row["test_micro_f1"] == pytest.approx(accuracy, abs=1e-6)
        assert Path(row["curves"]).is_file() and Path(row["history"]).is_file()
        assert row["checkpoint_bytes"] > row["weight_tensor_bytes"] > 0
        validation = matrix.read_json(row["validation_predictions"])
        assert {r["partition"] for r in validation} == {"validation"}
    examples = report["best_custom_examples"]
    assert len(examples["records"]) > 0
    for record in examples["records"]:
        assert Path(record["source"]) in test_paths and record["partition"] == "test"
        assert Path(record["path"]).is_file()
        frames, fps = dataset.read_video_torchvision(record["path"])
        assert frames.shape[0] == 50 and fps > 0
    # Actual checkpoints/records/curves are reused, with zero extra evaluator calls.
    study_config.execute_study(
        path, *study_config.load_study(path), experiments.main, gated_evaluate
    )
    assert len(calls) == 8


def test_changed_resume_identity_preserves_completed_evidence(protocol, monkeypatch):
    group = run_search(protocol)
    old = (group / "progress.json").read_bytes()
    original = matrix.file_hash
    monkeypatch.setattr(
        matrix,
        "file_hash",
        lambda p: "changed-code" if Path(p).name == "requirements.txt" else original(p),
    )
    with pytest.raises(ValueError, match="identity changed"):
        run_search(protocol)
    assert len(protocol.trained) == 36
    assert (group / "progress.json").read_bytes() == old
    assert matrix.read_json(group / "run.json")["status"] == "complete"


def test_zip_rejects_missing_completed_checkpoint(tmp_path):
    study, progress, leaf = archive_fixture(tmp_path)
    checkpoint = leaf / "best_model.pth"
    expected = matrix.file_hash(checkpoint)
    group = tmp_path / "runs/studies/group"
    write_json(
        group / "jobs/01/receipt.json",
        {
            "status": "complete",
            "run_dir": str(leaf),
            "files": {str(checkpoint): expected},
        },
    )
    checkpoint.unlink()
    with pytest.raises(ValueError, match="Completed evidence is missing"):
        study_archive.make_archive(
            tmp_path, study / "artifacts.zip", progress, {group}, aad_study.PROFILES
        )


def test_notebook_download_stage_never_bootstraps_or_trains(monkeypatch):
    import ast

    notebook = matrix.read_json(ROOT / "notebooks/aad_experiment_workflow.ipynb")
    code = "".join(
        next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    )
    tree = ast.parse(code)
    tree.body.pop()  # Do not execute the user-selected entry point until configured.
    namespace = {}
    exec(compile(tree, "<notebook>", "exec"), namespace)
    namespace.update(WORKFLOW_STAGE="download", ARTIFACT_ZIP_PATH="saved.zip")
    calls = []
    monkeypatch.setattr(
        study_archive, "retry_download", lambda path: calls.append(path) or "requested"
    )
    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda *a, **k: pytest.fail("download bootstrapped"),
    )
    assert namespace["run_workflow"]() == "requested"
    assert calls == ["saved.zip"]
