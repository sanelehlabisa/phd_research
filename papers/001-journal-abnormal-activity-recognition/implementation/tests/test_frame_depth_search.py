"""087: metadata/CPU fixtures only; no real search or Drive access."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src import capacity_config as plan, capacity_search as search
from src import focused_protocol as focused, study_config, study_matrix as matrix
from src import study_cache
from src.study_reporting import POOLED_REPORTING, result_row
from src.utils import write_json
from notebooks.utils import study_archive as archive
from test_capacity_search import ROOT, protocol
from test_comparison_only import pooled_train, pooled_test, comparison_fixture
from test_focused_workflow import review_for

PROFILE = ROOT / "configs/experiments/aad_frame_depth_search_colab.json"


def test_manifest_and_plan_preserve_old_profile(capsys):
    study, cfg, manifest = study_config.load_study(PROFILE)
    stacks = study["architectures"]
    assert len(manifest.candidates) == len(set(map(tuple, stacks))) == 32
    assert Counter(map(len, stacks)) == {1: 5, 2: 6, 3: 13, 4: 4, 5: 4}
    assert set(map(tuple, focused.STACKS)) <= set(map(tuple, stacks))
    assert all([w] * d in stacks for w in (32, 48, 64, 96) for d in range(1, 6))
    rows = [r for _, block in focused.blocks(cfg, study) for r in block]
    assert len(rows) == len({search.job_key(r) for r in rows}) == 128
    assert set(plan.stage_limits(study).values()) == {32}
    assert {
        (
            r["config"]["sequence_length"],
            r["config"]["height"],
            r["config"]["target_fps"],
        )
        for r in rows
    } == {(8, 64, 8), (8, 64, 16), (16, 64, 8), (16, 64, 16)}
    study_config.print_study(PROFILE, study, cfg)
    output = capsys.readouterr().out
    assert "Experiment 128/128" in output and "authorize/mount" in output
    old, old_cfg, old_manifest = study_config.load_study(
        ROOT / "configs/experiments/aad_focused_search_colab.json"
    )
    assert old["schema_version"] == 5 and len(old_manifest.candidates) == 16
    assert len([r for _, block in focused.blocks(old_cfg) for r in block]) == 64
    assert list(plan.stage_limits(old).values()) == [16] * 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("architectures", [[4]] * 32),
        ("num_frames", [8]),
        ("frame_sizes", [32, 64]),
        ("seeds", [42, 2026]),
        ("max_runs", 64),
        ("target_fps", [4, 8]),
    ],
)
def test_unapproved_protocol_changes_rejected(field, value):
    study = matrix.read_json(PROFILE)
    study[field] = value
    with pytest.raises(ValueError):
        plan.load_capacity(study)


def ranked_records():
    study, cfg, _ = study_config.load_study(PROFILE)
    rows = []
    for i, spec in enumerate(focused.candidates(study)):
        for n, s, f in focused.input_conditions(study):
            count = 60 + i
            result = dict(
                name=spec["name"],
                seed=42,
                partition="validation",
                test_access="locked",
                reporting_protocol=POOLED_REPORTING,
                num_params=1000 + i,
                experiment_config={
                    **cfg.to_dict(),
                    "sequence_length": n,
                    "height": s,
                    "width": s,
                    "target_fps": f,
                    "convlstm_layers": spec["convlstm_layers"],
                },
                validation_metrics={"accuracy": count / 100, "loss": 0.2},
                validation_confusion_matrix={"matrix": [[count, 100 - count], [0, 0]]},
            )
            rows.append({"status": "complete", "result": result})
    return study, rows


def test_four_condition_ranking_and_new_input_receipt():
    study, rows = ranked_records()
    decision = focused.decisions(rows, study)
    assert len(decision["ranking"]) == 32
    assert decision["numerical_top5"] == decision["finalist_names"]
    assert not decision["anchor_reserved_slot"]
    assert decision["final_input"]["sequence_length"] == 8
    assert decision["final_input"]["target_fps"] == 16
    assert len(decision["ranking"][0]["conditions"]) == 4
    # Equal-mean ties prefer a better worst condition, not a single lucky input.
    names = [r["result"]["name"] for r in rows[::4]][:2]
    for i, job in enumerate(rows):
        r = job["result"]
        count = (
            [80, 100, 80, 100][i % 4]
            if r["name"] == names[0]
            else 90 if r["name"] == names[1] else 20
        )
        r["validation_metrics"]["accuracy"] = count / 100
        r["validation_confusion_matrix"]["matrix"] = [[count, 100 - count], [0, 0]]
    assert [r["name"] for r in focused.decisions(rows, study)["ranking"][:2]] == names[
        ::-1
    ]
    for bad in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError):
            focused.decisions(bad, study)
    for field, value in (
        ("partition", "test"),
        ("reporting_protocol", "legacy_extended_v1"),
    ):
        bad = deepcopy(rows)
        bad[0]["result"][field] = value
        with pytest.raises(ValueError):
            focused.decisions(bad, study)
    rows[0]["status"] = "failed"
    with pytest.raises(ValueError, match="failed"):
        focused.decisions(rows, study)


def test_all_actual_models_preflight_at_max_frames():
    study, cfg, manifest = study_config.load_study(PROFILE)
    probes = []

    def probe(spec):
        probes.append(spec)
        return {"memory_safe": spec["batch_size"] <= 2, "training_step_seconds": 0.1}

    report = search._frame_preflight(study, cfg, manifest, probe=probe)
    assert report["selected_batch_size"] == 2
    assert len([r for r in probes if r["batch_size"] == 2]) == 32
    assert all(
        (r["frames"], r["size"], r["phase"]) == (16, 64, "search") for r in probes
    )
    assert (
        search._frame_preflight(
            study, cfg, manifest, probe=lambda s: {"memory_safe": False}
        )["selected_batch_size"]
        is None
    )
    with pytest.raises(RuntimeError, match="CPU fit is unverified"):
        search._frame_preflight(study, cfg, manifest, device=torch.device("cpu"))
    storage = search._frame_storage_estimates(study, 908)
    assert len(storage["parameter_counts"]) == 32
    assert (
        storage["train_validation_cache_bytes_by_condition"][2]
        == 2 * storage["train_validation_cache_bytes_by_condition"][0]
    )


def test_reviewed_split_gate():
    split = {"counts": {"train": 748, "validation": 160, "test": 161, "total": 1069}}
    search._validate_frame_split(split, 11)
    with pytest.raises(ValueError, match="reviewed AAD"):
        search._validate_frame_split(split, 2)
    split["counts"]["train"] = 747
    with pytest.raises(ValueError):
        search._validate_frame_split(split, 11)


@pytest.mark.parametrize("stack", [[128, 128, 128], [96, 96, 96, 96, 96]])
def test_wide_and_deep_stack_real_tiny_cpu_optimizer_step(stack):
    from src.model import CustomConvLSTM

    torch.set_num_threads(1)
    model = CustomConvLSTM(11, layers=[(w, (3, 3)) for w in stack])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.classifier.weight.detach().clone()
    logits = model(torch.randn(2, 2, 3, 4, 4))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    assert logits.shape == (2, 11) and torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.weight)
    # Small tensors prove execution, never 16-frame/64px A100 memory fit.


def test_cache_isolated_by_frames_and_fps_and_reused(protocol, monkeypatch):
    from src.dataset import AHARDataset, load_split_subsets

    study_cache._CACHE.clear()
    calls, keys = [], []

    def decode(self, path):
        calls.append(str(path))
        return torch.zeros(self.sequence_length, 3, *self.frame_size)

    monkeypatch.setattr(AHARDataset, "_load_video", decode)
    for n, f in ((8, 8), (8, 16), (16, 8), (16, 16)):
        source = AHARDataset(
            protocol.data, n, (64, 64), target_fps=f, sampling_version="timestamps_v1"
        )
        tr, va, te, split = load_split_subsets(
            source, Path(protocol.values["dataset"]["split_manifest"])
        )
        before = len(calls)
        cached, report = study_cache.cached_training_dataset(
            source,
            tr.indices,
            va.indices,
            split["manifest_hash"],
            protocol.root / "cache",
        )
        _, again = study_cache.cached_training_dataset(
            source,
            tr.indices,
            va.indices,
            split["manifest_hash"],
            protocol.root / "cache",
        )
        assert again["reused"] and len(calls) - before == len(tr) + len(va)
        assert cached[tr.indices[0]][0].shape == (n, 3, 64, 64)
        assert not set(calls) & {str(source.samples[i][0]) for i in te.indices}
        keys.append(report["key"])
        study_cache._CACHE.clear()
        _, disk = study_cache.cached_training_dataset(
            source,
            tr.indices,
            va.indices,
            split["manifest_hash"],
            protocol.root / "cache",
        )
        assert disk["reuse_source"] == "disk" and len(calls) - before == len(tr) + len(
            va
        )
    assert len(set(keys)) == 4
    study_cache._CACHE.clear()


def drive_standin(tmp_path, monkeypatch):
    mount = tmp_path / "drive"
    (mount / "MyDrive").mkdir(parents=True)
    monkeypatch.setattr(archive.os.path, "ismount", lambda p: Path(p) == mount)
    return {
        "mount_root": str(mount),
        "directory": str(mount / "MyDrive/phd_research/paper-001"),
    }


def test_backup_checks_mount_before_any_data_resolution(tmp_path, monkeypatch):
    study, cfg, manifest = study_config.load_study(PROFILE)
    monkeypatch.setattr(archive.os.path, "ismount", lambda p: False)
    from src import dataset_source

    monkeypatch.setattr(
        dataset_source,
        "resolve_dataset",
        lambda *a: pytest.fail("data resolution preceded Drive check"),
    )
    with pytest.raises(RuntimeError, match="not mounted/authorized"):
        search.execute_capacity(
            PROFILE, study, cfg, manifest, lambda args: pytest.fail("training")
        )


def test_results_backup_copy_hash_failure_keeps_previous_and_local(
    tmp_path, monkeypatch, capsys
):
    backup = drive_standin(tmp_path, monkeypatch)
    group = tmp_path / "runs/studies/campaign"
    leaf = tmp_path / "runs/experiments/leaf"
    leaf.mkdir(parents=True)
    (leaf / "best.pth").write_bytes(b"weights excluded")
    write_json(leaf / "metrics.json", {"accuracy": 0.5})
    write_json(group / "study.json", {"backup": backup})
    write_json(group / "progress.json", {"jobs": [{"run_dir": str(leaf)}]})
    local = archive.compact_results(group, tmp_path / "runs")
    delivery = matrix.read_json(group / "drive_backup.json")
    destination = Path(delivery["destination"])
    assert (
        matrix.file_hash(local) == matrix.file_hash(destination) == delivery["sha256"]
    )
    members = archive.verify_archive(destination)
    assert any(p.endswith("metrics.json") for p in members) and not any(
        p.endswith(".pth") for p in members
    )
    previous = destination.read_bytes()
    monkeypatch.setattr(
        archive.shutil, "copyfileobj", lambda source, out: out.write(b"truncated")
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        archive.backup_results(local, group, backup)
    assert destination.read_bytes() == previous and local.is_file()
    assert "DRIVE BACKUP FAILED" in capsys.readouterr().out
    assert not list(destination.parent.glob(".results-*.partial"))
    write_json(destination.parent / "backup_owner.json", {"study_id": "other"})
    with pytest.raises(RuntimeError, match="another/unverified campaign"):
        archive.backup_results(local, group, backup)
    assert destination.read_bytes() == previous


def test_all_128_points_and_controlled_groups(tmp_path):
    study, jobs = ranked_records()
    rows = [result_row(j["result"], raw=True) for j in jobs]
    focused.frame_depth_summaries(tmp_path, rows, study)
    points = matrix.read_json(tmp_path / "frame_depth_plot_points.json")
    assert points["measured_points"] == 128
    assert [len(p["points"]) for p in points["panels"]] == [32] * 4
    assert (tmp_path / "frame_depth_capacity.png").stat().st_size > 1000
    groups = matrix.read_json(tmp_path / "matched_capacity_groups.json")["groups"]
    assert all(
        len(g["rows"]) == 5 and not g["missing"]
        for g in groups
        if g["group"].startswith("depth_at_width")
    )
    assert any(g["missing"] for g in groups if g["group"].startswith("position_96"))
    assert any(r["model"] == "custom_48_48_48" for g in groups for r in g["rows"])


def test_synthetic_128_search_selection_archive_and_verified_resume(
    protocol, monkeypatch
):
    p = protocol
    values = matrix.read_json(PROFILE)
    values["dataset"].update(
        path=str(p.data), split_manifest=p.values["dataset"]["split_manifest"]
    )
    values["runs_dir"] = str(p.root / "runs")
    values["backup"] = drive_standin(p.root, monkeypatch)
    path = write_json(p.root / "frame_search.json", values)
    review = write_json(
        p.root / "review.json", review_for(values["dataset"]["split_manifest"])
    )
    audit = search.audit_split
    monkeypatch.setattr(
        search,
        "audit_split",
        lambda source, split, review_path=None: audit(source, split, review),
    )
    # Two-class miniature only; the real reviewed-count gate is tested separately.
    monkeypatch.setattr(search, "_validate_frame_split", lambda *a: None)
    monkeypatch.setattr(
        search,
        "_frame_preflight",
        lambda *a: {"status": "synthetic", "selected_batch_size": 2, "results": []},
    )
    train = pooled_train(p)

    def forbidden(*args):
        pytest.fail("Search must not decode test or evaluate a final model")

    from src.dataset import AHARDataset

    monkeypatch.setattr(AHARDataset, "_load_video", forbidden)
    group = study_config.execute_study(
        path, *study_config.load_study(path), train, forbidden
    )
    assert len(p.trained) == 128
    selected = matrix.load_selection(group / "selected_config.json")
    assert selected["protocol"] == "frame_depth_top5_v1"
    assert len(selected["ranking"]) == 32 and len(selected["top5"]) == 5
    assert all(len(c["sources"]) == 4 for c in selected["top5"])
    assert selected["finalist_names"] == selected["numerical_top5"]
    state = matrix.read_json(group / "progress.json")
    assert [j["experiment_number"] for j in state["jobs"]] == list(range(1, 129))
    local = p.root / "runs/focused_archives" / group.name / "results.zip"
    delivery = matrix.read_json(group / "drive_backup.json")
    assert matrix.file_hash(local) == matrix.file_hash(delivery["destination"])
    members = archive.verify_archive(local)
    assert any(n.endswith("selected_config.json") for n in members)
    assert any(n.endswith("train_predictions.json") for n in members)
    assert not any(n.endswith(".pth") for n in members)
    assert (
        study_config.execute_study(
            path, *study_config.load_study(path), forbidden, forbidden
        )
        == group
    )
    assert len(p.trained) == 128
    changed = deepcopy(selected)
    changed["final_input"]["sequence_length"] = 32
    write_json(group / "tampered_selection.json", changed)
    with pytest.raises(ValueError, match="differs"):
        matrix.load_selection(group / "tampered_selection.json")


def test_comparison_uses_same_backup_path_before_after_models_tests(
    protocol, monkeypatch
):
    p = protocol
    path, _ = comparison_fixture(p, monkeypatch)
    values = matrix.read_json(path)
    values["backup"] = drive_standin(p.root, monkeypatch)
    write_json(path, values)
    copies = []
    original = archive.backup_results

    def backup(*args):
        result = original(*args)
        copies.append(result)
        return result

    monkeypatch.setattr(archive, "backup_results", backup)
    tested = []
    group = study_config.execute_study(
        path, *study_config.load_study(path), pooled_train(p), pooled_test(p, tested)
    )
    assert len(p.trained) == len(tested) == 10
    assert len(copies) >= 21
    delivery = matrix.read_json(group / "drive_backup.json")
    assert matrix.file_hash(delivery["local_zip"]) == matrix.file_hash(
        delivery["destination"]
    )
    members = archive.verify_archive(delivery["destination"])
    assert any(n.endswith("validation_frozen.json") for n in members)
    assert any(n.endswith("comparison.csv") for n in members)
    assert not any(n.endswith(".pth") for n in members)


def test_handled_search_failure_keeps_completed_evidence_and_copies_zip(
    protocol, monkeypatch
):
    p = protocol
    values = matrix.read_json(PROFILE)
    values["dataset"].update(
        path=str(p.data), split_manifest=p.values["dataset"]["split_manifest"]
    )
    values["runs_dir"] = str(p.root / "runs")
    values["backup"] = drive_standin(p.root, monkeypatch)
    path = write_json(p.root / "frame_search.json", values)
    review = write_json(
        p.root / "review.json", review_for(values["dataset"]["split_manifest"])
    )
    audit = search.audit_split
    monkeypatch.setattr(
        search,
        "audit_split",
        lambda source, split, review_path=None: audit(source, split, review),
    )
    monkeypatch.setattr(search, "_validate_frame_split", lambda *a: None)
    monkeypatch.setattr(
        search,
        "_frame_preflight",
        lambda *a: {"status": "synthetic", "selected_batch_size": 2, "results": []},
    )
    train = pooled_train(p)

    def fail_second(args):
        if p.trained:
            raise RuntimeError("synthetic training interrupted")
        return train(args)

    with pytest.raises(RuntimeError, match="synthetic training interrupted"):
        study_config.execute_study(path, *study_config.load_study(path), fail_second)
    group = next((p.root / "runs/studies").iterdir())
    progress = matrix.read_json(group / "progress.json")
    assert progress["status"] == "incomplete"
    assert [j["status"] for j in progress["jobs"]] == ["complete", "failed"]
    delivery = matrix.read_json(group / "drive_backup.json")
    assert matrix.file_hash(delivery["local_zip"]) == matrix.file_hash(
        delivery["destination"]
    )
    assert not (group / "selected_config.json").exists()
    with pytest.raises(RuntimeError, match="interrupted/failed"):
        study_config.execute_study(path, *study_config.load_study(path), fail_second)
    assert len(p.trained) == 1
