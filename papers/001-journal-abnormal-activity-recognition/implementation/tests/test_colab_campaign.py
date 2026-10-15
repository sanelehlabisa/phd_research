"""089: CPU/synthetic evidence only; no Colab, Drive credentials or research run."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from notebooks.utils import aad_study, diagnostics, study_archive as archive
from src import campaign_checks as checks, capacity_search as search
from src import study_config, study_matrix as matrix, study_resources, dataset
from src.utils import write_json
from test_capacity_search import ROOT, protocol
from test_comparison_only import pooled_train, pooled_test, comparison_fixture
from test_frame_depth_search import drive_standin
from test_focused_workflow import review_for


def passed_trace():
    return dict(
        learning=[dict(accuracy=0.5, loss=1.0), dict(accuracy=1.0, loss=0.1)],
        gradient_updates=[
            dict(
                all_gradients_present_and_finite=True,
                parameters={"weight": dict(update_norm=0.1)},
            )
        ]
        * 128,
        optimizer_covers_all_parameters=True,
    )


def test_notebook_export_parity():
    path = ROOT / "notebooks/aad_frame_depth_workflow"
    export = path.with_suffix(".py").read_text(encoding="utf-8")
    notebook = matrix.read_json(path.with_suffix(".ipynb"))
    code = "".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert code.strip() == export.strip()
    ast.parse(code)
    assert "aad_fast_frame_depth_search_colab.json" in code
    assert "run_frame_depth_aad_study(root)" in code
    assert all(not c.get("outputs") for c in notebook["cells"])


def test_fast_profile_is_separate_and_preflight_uses_32px():
    fast_path = ROOT / "configs/experiments/aad_fast_frame_depth_search_colab.json"
    original_path = ROOT / "configs/experiments/aad_frame_depth_search_colab.json"
    fast, cfg, manifest = study_config.load_study(fast_path)
    old, old_cfg, _ = study_config.load_study(original_path)
    assert cfg.epochs == 128 and cfg.height == 32 and fast["max_runs"] == 128
    assert old_cfg.epochs == 256 and old_cfg.height == 64 and old["schema_version"] == 6
    seen = []

    def probe(spec):
        seen.append(spec)
        return dict(memory_safe=True)

    search._frame_preflight(fast, cfg, manifest, probe=probe)
    assert len(seen) == 32 and {(r["frames"], r["size"]) for r in seen} == {(16, 32)}


def test_failed_or_changed_transformer_gate_cannot_authorize_training(
    protocol, monkeypatch
):
    p = protocol
    _, cfg, _ = study_config.load_study(p.path)
    selection = write_json(p.root / "selection.json", {"version": 1})
    study = dict(approved_recipe=checks.APPROVED_RECIPE, selected_config=str(selection))
    selected = dict(split=dict(manifest_hash=matrix.file_hash(cfg.split_manifest)))
    group = p.root / "gate"
    state = dict(jobs=[])
    calls = []

    def failed(config, path):
        calls.append(path)
        write_json(path, dict(status="failed", models={}))
        raise RuntimeError("synthetic failed learning gate")

    monkeypatch.setattr(checks, "run_transformer_checks", failed)
    with pytest.raises(RuntimeError, match="failed learning"):
        checks.ensure_transformer_checks(group, study, cfg, state, selected)
    with pytest.raises(ValueError, match="no silent retry"):
        checks.ensure_transformer_checks(group, study, cfg, state, selected)
    assert len(calls) == 1 and not state["jobs"]
    good_group = p.root / "good_gate"

    def passed(config, path):
        return write_report(path)

    def write_report(path):
        report = dict(status="passed", models=dict(swin3d_t={}, swin3d_s={}))
        write_json(path, report)
        return report

    monkeypatch.setattr(checks, "run_transformer_checks", passed)
    checks.ensure_transformer_checks(good_group, study, cfg, state, selected)
    checks.ensure_transformer_checks(good_group, study, cfg, state, selected)
    write_json(selection, {"version": 2})
    with pytest.raises(ValueError, match="changed"):
        checks.ensure_transformer_checks(good_group, study, cfg, state, selected)


def test_incomplete_resolution_checks_block_final_training(tmp_path):
    selected = dict(top5=[dict(comparison_name=f"custom_top{i}") for i in range(1, 6)])
    with pytest.raises(ValueError, match="All five"):
        checks.freeze_resolution_checks(tmp_path, [], selected)


def test_visible_drive_test_and_refusal_before_training(tmp_path, monkeypatch):
    backup = drive_standin(tmp_path, monkeypatch)
    path = archive.connect_drive(backup)
    assert path.is_file() and "PhD results backup test OK" in path.read_text()
    assert archive.connect_drive(backup) != path  # Never overwrite previous tests.
    monkeypatch.setattr(archive.os.path, "ismount", lambda p: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        archive.preflight_backup(backup)


@pytest.mark.parametrize(
    "failure", ["loss", "accuracy", "gradient", "updates", "optimizer"]
)
def test_train_only_gate_rejects_bad_learning(failure):
    report = passed_trace()
    assert checks.check_passed(report)
    if failure == "loss":
        report["learning"][-1]["loss"] = 1.0
    elif failure == "accuracy":
        report["learning"][-1]["accuracy"] = 0.99
    elif failure == "gradient":
        report["gradient_updates"][0]["all_gradients_present_and_finite"] = False
    elif failure == "updates":
        report["gradient_updates"][0]["parameters"]["weight"]["update_norm"] = 0.0
    else:
        report["optimizer_covers_all_parameters"] = False
    assert not checks.check_passed(report)


def test_recipe_and_full_step_learning_rates(protocol, monkeypatch):
    path, selected = comparison_fixture(protocol, monkeypatch)
    selected["protocol"] = "fast_frame_depth_top5_v1"
    selected["final_input"].update(sequence_length=16, height=32, width=32)
    values = matrix.read_json(path)
    values.pop("comparison_only")
    values.pop("recovery_source")
    values["training"]["epochs"] = 256
    values["max_runs"] = 15
    values.update(
        comparison_protocol="frame_depth_final_v1",
        approved_recipe=checks.APPROVED_RECIPE,
    )
    values["reference_input"]["sequence_length"] = 16
    values["factors"]["sequence_lengths"] = [16]
    write_json(path, values)
    study, cfg, manifest = study_config.load_study(path)
    rows = study_config.study_rows(study, cfg)
    assert len(rows) == 15
    assert all(r["config"]["epochs"] == 128 for r in rows[:5])
    assert all(r["config"]["epochs"] == 256 for r in rows[5:])
    assert all(
        r["config"]["learning_rate"] == checks.model_learning_rate(r["model"])
        for r in rows
    )
    assert all(r["config"]["sequence_length"] == 16 for r in rows)
    values["approved_recipe"] = {**checks.APPROVED_RECIPE, "swin_learning_rate": 0.001}
    write_json(path, values)
    with pytest.raises(ValueError, match="owner-approved"):
        study_config.load_study(path)


def test_checks_use_only_train_and_exact_batch_and_save_failure(protocol, monkeypatch):
    p = protocol
    _, cfg, _ = study_config.load_study(p.path)
    from src.experiment_config import ExperimentConfig

    cfg = ExperimentConfig.from_mapping(
        {
            **cfg.to_dict(),
            "batch_size": 8,
            "sequence_length": 16,
            "height": 64,
            "width": 64,
        }
    )
    split = matrix.read_json(cfg.split_manifest)
    allowed = {
        str((p.data / r["path"]).resolve())
        for r in split["samples"]
        if r["split"] == "train"
    }
    decoded, models = [], []

    def decode(self, path):
        assert str(Path(path).resolve()) in allowed
        decoded.append(path)
        return torch.zeros(16, 3, 64, 64)

    def check(name, clips, targets, classes, **kwargs):
        assert clips.shape == (8, 16, 3, 64, 64)
        assert kwargs["batch_size"] == 8 and kwargs["learning_rate"] == 0.00001
        models.append(name)
        result = passed_trace()
        if name == "swin3d_s":
            result["learning"][-1]["accuracy"] = 0.5
        return result

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", decode)
    monkeypatch.setattr(diagnostics, "registered_learning_check", check)
    destination = p.root / "check.json"
    with pytest.raises(RuntimeError, match="swin3d_s failed"):
        checks.run_transformer_checks(cfg, destination)
    assert len(set(decoded)) == 4 and models == ["swin3d_t", "swin3d_s"]
    saved = matrix.read_json(destination)
    assert saved["status"] == "failed" and saved["test_clips_opened"] == 0


def test_full_run_all_search_checks_comparison_test_backup_resume(
    protocol, monkeypatch, capsys
):
    p = protocol
    backup = drive_standin(p.root, monkeypatch)
    for name in (
        "aad_fast_frame_depth_search_colab.json",
        "aad_frame_depth_comparison_colab.json",
    ):
        values = matrix.read_json(ROOT / "configs/experiments" / name)
        values["dataset"].update(
            path=str(p.data), split_manifest=str(p.root / "split.json")
        )
        values.update(runs_dir=str(p.root / "runs"), backup=backup)
        write_json(p.root / "configs/experiments" / name, values)
    review = write_json(p.root / "review.json", review_for(p.root / "split.json"))
    audit = search.audit_split
    monkeypatch.setattr(
        search,
        "audit_split",
        lambda source, split, review_path=None: audit(source, split, review),
    )
    monkeypatch.setattr(search, "_validate_frame_split", lambda *a: None)
    resource = dict(status="synthetic", selected_batch_size=2, results=[])
    monkeypatch.setattr(search, "_frame_preflight", lambda *a: resource)
    monkeypatch.setattr(
        study_resources, "comparison_preflight", lambda *a, **kw: resource
    )
    monkeypatch.setattr(
        dataset.AHARDataset,
        "_load_video",
        lambda self, path: torch.zeros(self.sequence_length, 3, *self.frame_size),
    )
    checked, tested = [], []

    def check(name, *a, **kw):
        assert len(p.trained) == 128 and not tested
        checked.append(name)
        return passed_trace()

    monkeypatch.setattr(diagnostics, "registered_learning_check", check)
    train = pooled_train(p)

    def evaluate(path):
        assert len(p.trained) == 143 and checked == ["swin3d_t", "swin3d_s"]
        return pooled_test(
            SimpleNamespace(root=p.root, data=p.data, trained=p.trained[133:]), tested
        )(path)

    def command(args, root, log_path=None):
        path = Path(args[args.index("--config") + 1])
        values = study_config.load_study(path)
        if "--list-plan" in args:
            return study_config.print_study(path, values[0], values[1])
        return study_config.execute_study(path, *values, train, evaluate)

    monkeypatch.setattr(aad_study, "_run_command", command)
    monkeypatch.setattr(
        aad_study,
        "_download_archive",
        lambda *a: pytest.fail("No browser downloads during Run All"),
    )
    result = aad_study.run_frame_depth_aad_study(p.root)
    assert len(p.trained) == 143 and len(tested) == 10
    members = archive.verify_archive(result)
    assert all(n.endswith(".json") for n in members)
    with zipfile.ZipFile(result) as z:
        summary = json.loads(z.read("essential_results.json"))
    assert [len(s["jobs"]) for s in summary["studies"]] == [128, 10]
    assert len(summary["studies"][1]["tests"]) == 10
    comparison = summary["studies"][1]
    assert len(comparison["transfer_jobs"]) == 5
    assert [j["experiment_number"] for j in comparison["transfer_jobs"]] == list(
        range(129, 134)
    )
    assert all(
        j["result"]["experiment_config"]["height"] == 64
        for j in comparison["transfer_jobs"]
    )
    assert all(
        j["result"]["experiment_config"]["height"] == 32
        for j in summary["studies"][0]["jobs"]
    )
    assert {j["result"]["selected_checkpoint"] for j in comparison["jobs"]}.isdisjoint(
        j["result"]["selected_checkpoint"] for j in comparison["transfer_jobs"]
    )
    assert "transformer_checks" in comparison["validation_frozen"]
    for job in comparison["jobs"]:
        r = job["result"]
        assert r["experiment_config"]["learning_rate"] == checks.model_learning_rate(
            r["name"]
        )
        assert r["num_params"] > 0 and r["validation_confusion_matrix"]
    assert [j["experiment_number"] for j in comparison["jobs"]] == list(range(134, 144))
    group = p.root / "runs/studies" / comparison["study_id"]
    delivered = matrix.read_json(group / "drive_backup.json")
    assert matrix.file_hash(result) == matrix.file_hash(delivered["destination"])
    assert result.stat().st_size < 128 * 1024**2
    assert aad_study.run_frame_depth_aad_study(p.root) == result
    assert len(p.trained) == 143 and len(tested) == 10 and len(checked) == 2
    assert "Experiment 143/143" in capsys.readouterr().out
