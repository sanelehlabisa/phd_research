import json
from pathlib import Path
import zipfile

from notebooks.utils import aad_study


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/aad_experiment_workflow.ipynb"
EXPORT = ROOT / "notebooks/aad_experiment_workflow.py"


def test_notebook_and_python_export_have_the_same_runner_cell():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert len(code_cells) == 1
    notebook_source = "".join(code_cells[0]["source"])
    export_source = EXPORT.read_text(encoding="utf-8")
    assert export_source == "# In[ ]:\n" + notebook_source
    compile(notebook_source, str(NOTEBOOK), "exec")
    compile(export_source, str(EXPORT), "exec")
    assert "RUN_FULL_STUDY = True" in notebook_source
    assert "if bootstrap.returncode == 75" in notebook_source
    assert "raise SystemExit" not in notebook_source
    assert "Restart the Colab runtime, reconnect the GPU, then rerun this cell." in notebook_source


def test_smoke_path_lists_all_profiles_without_dataset_download_or_training(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        aad_study,
        "_run_command",
        lambda command, root: calls.append((command, root)),
    )

    result = aad_study.run_aad_study(ROOT, run_full_study=False)

    assert result is None
    assert len(calls) == 2
    assert all(call[0][-1] == "--list-plan" for call in calls)
    assert "no dataset download or training" in capsys.readouterr().out


def test_archive_contains_only_selected_runs_profiles_and_progress(tmp_path):
    root = tmp_path / "implementation"
    for _, config in aad_study.PROFILES:
        config_path = root / config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}\n", encoding="utf-8")

    run_directory = root / "runs/experiments/run-a"
    (run_directory / "checkpoints").mkdir(parents=True)
    (run_directory / "metrics").mkdir()
    (run_directory / "predictions/correct").mkdir(parents=True)
    (run_directory / "checkpoints/best_model.pth").write_bytes(b"checkpoint")
    (run_directory / "metrics/validation.json").write_text("{}\n", encoding="utf-8")
    (run_directory / "predictions/correct/example.mp4").write_bytes(b"video")
    (run_directory / "resolved_config.json").write_text("{}\n", encoding="utf-8")
    evaluation_directory = root / "runs/evaluate/run-a/metrics"
    evaluation_directory.mkdir(parents=True)
    (evaluation_directory / "final.json").write_text("{}\n", encoding="utf-8")
    selection_directory = root / "runs/studies/run-a"
    selection_directory.mkdir(parents=True)
    (selection_directory / "selected_config.json").write_text("{}\n", encoding="utf-8")
    progress = root / "runs/notebook_studies/current/progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text('{"stages": []}\n', encoding="utf-8")
    archive_path = progress.parent / "artifacts.zip"

    aad_study._make_archive(
        root, archive_path, progress,
        {run_directory.resolve(), (root / "runs/evaluate/run-a").resolve(), selection_directory.resolve()},
    )

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "configs/experiments/aad_custom_search_colab.json" in names
    assert "configs/experiments/aad_model_comparison_colab.json" in names
    assert "runs/experiments/run-a/checkpoints/best_model.pth" in names
    assert "runs/experiments/run-a/metrics/validation.json" in names
    assert "runs/experiments/run-a/predictions/correct/example.mp4" in names
    assert "runs/experiments/run-a/resolved_config.json" in names
    assert "runs/notebook_studies/current/progress.json" in names
    assert "runs/experiments/unrelated-old-run" not in names
    assert "runs/evaluate/run-a/metrics/final.json" in names
    assert "runs/studies/run-a/selected_config.json" in names


def test_study_records_each_stage_and_downloads_one_archive(tmp_path, monkeypatch):
    root = tmp_path / "implementation"
    for index, (_, config) in enumerate(aad_study.PROFILES):
        config_path = root / config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        values = {"dataset": {"name": "aad", "path": "datasets/aad"}}
        if index == 1:
            values["schema_version"] = 1
        if index == 2:
            values["models"] = ["custom_selected", "paper_convlstm_published", "r3d_18", "mc3_18", "swin3d_t", "swin3d_s"]
        config_path.write_text(json.dumps(values), encoding="utf-8")

    commands = []
    downloads = []

    def fake_run(command, implementation_root, log_path=None):
        commands.append(command)
        if command[-1] == "--list-plan":
            return
        config_path = Path(command[command.index("--config") + 1])
        if "resolved_custom_search" in config_path.name:
            run_directory = implementation_root / "runs/studies/search"
            run_directory.mkdir(parents=True)
            (run_directory / "selected_config.json").write_text(json.dumps({
                "candidate": {"name": "tiny", "convlstm_layers": [[8, [3, 3]], [4, [3, 3]]], "hidden_classifier_width": None}
            }), encoding="utf-8")
        else:
            run_directory = implementation_root / "runs/experiments/comparison"
            run_directory.mkdir(parents=True)
            (run_directory / "summary.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(aad_study, "_run_command", fake_run)
    monkeypatch.setattr(aad_study, "resolve_dataset", lambda *args: tmp_path / "aad")
    monkeypatch.setattr(aad_study, "_download_archive", downloads.append)

    archive = aad_study.run_aad_study(root)

    assert archive is not None and archive.is_file()
    assert len(commands) == 5
    assert downloads == [archive]
    with zipfile.ZipFile(archive) as packaged:
        progress = json.loads(
            packaged.read("runs/notebook_studies/" + archive.parent.name + "/progress.json")
        )
        names = set(packaged.namelist())
    assert [stage["status"] for stage in progress["stages"]] == [
        "complete",
        "complete",
    ]
    assert "runs/studies/search/selected_config.json" in names
    assert "runs/experiments/comparison/summary.json" in names
