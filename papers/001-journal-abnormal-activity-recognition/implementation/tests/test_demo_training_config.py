"""Keep local demo retraining tied to the frozen winner, without opening test."""

import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from src import dataset_source, experiments


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/aad_demo_selected_local.json"
ARCHIVE = ROOT / "reports/2026-10-16/evidence/results.zip"
MODEL = "custom_convlstm_configured"


@pytest.fixture(scope="module")
def archived_final():
    """Read immutable evidence; the compact archive contains no model weights."""
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == (
        "3042f4512825c93cfc831f0f591d0ecd37c51c85dcdfbd058290d5153cbefe6f"
    )
    with ZipFile(ARCHIVE) as archive:
        return json.loads(archive.read("essential_results.json"))["studies"][1]


def resolved():
    """Minimum epochs and distinct run provenance are explicit CLI operations."""
    raw = vars(
        experiments.parser.parse_args(
            [
                "--config",
                str(CONFIG),
                "--minimum-epochs",
                "128",
                "--run-label",
                "demo-selected-retrained",
                "--matched-training-evaluation",
                "--per-class-reporting",
            ]
        )
    )
    return experiments._resolved_arguments(raw)


def test_recipe_matches_validation_frozen_winner(archived_final):
    winner = archived_final["validation_frozen"]["best_custom"]
    assert winner == "custom_top1"
    result = next(
        job["result"]
        for job in archived_final["jobs"]
        if job["result"]["name"] == winner
    )
    args, config, candidates, print_only = resolved()
    assert candidates is None and not print_only
    assert args.models == [MODEL]
    assert args.minimum_epochs == 128
    assert args.run_label == "demo-selected-retrained"
    assert args.matched_training_evaluation and args.per_class_reporting
    values = config.to_dict()
    # Only path/provenance and Windows worker count differ from the Colab recipe.
    local_fields = {
        "dataset_name",
        "dataset_dir",
        "runs_dir",
        "split_manifest",
        "num_workers",
    }
    for key, value in result["experiment_config"].items():
        if key not in local_fields:
            assert values[key] == value, key
    assert config.num_workers == 0
    assert config.runs_dir == "runs/demo"
    assert "minimum_epochs" not in json.loads(CONFIG.read_text(encoding="utf-8"))
    registry = experiments.model_registry(config, selected_models=args.models)
    assert len(registry) == 1 and registry[0]["name"] == MODEL
    model = experiments.build_registered_model(
        MODEL, 11, (3, 64, 64), 8, experiment_config=config
    )
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == (
        result["num_params"]
    ) == 1_336_459


def test_original_split_assignments_and_class_order_are_preserved(archived_final):
    _, config, _, _ = resolved()
    split = json.loads((ROOT / config.split_manifest).read_text(encoding="utf-8"))
    # Git may use CRLF locally, so compare assignments, not the Colab byte hash.
    assert split == archived_final["split_manifest"]
    assert split["counts"] == {
        "train": 748,
        "validation": 160,
        "test": 161,
        "total": 1069,
    }
    assert split["class_names"] == list(dataset_source.AAD_CLASS_NAMES)


def test_missing_local_dataset_never_downloads(tmp_path, monkeypatch):
    _, config, _, _ = resolved()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("The local demo must not download data")

    monkeypatch.setattr(dataset_source.kagglehub, "dataset_download", forbidden)
    with pytest.raises(FileNotFoundError, match="not found"):
        dataset_source.resolve_dataset(
            config.dataset_name, config.dataset_dir, tmp_path
        )


@pytest.mark.parametrize("nested", [False, True])
def test_local_class_folder_layout_resolves_without_download(
    tmp_path, monkeypatch, nested
):
    _, config, _, _ = resolved()
    root = tmp_path / config.dataset_dir
    if nested:
        root /= "abnormal-activities-dataset"
    for name in dataset_source.AAD_CLASS_NAMES:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "example.mp4").touch()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("The local demo must not download data")

    monkeypatch.setattr(dataset_source.kagglehub, "dataset_download", forbidden)
    assert dataset_source.resolve_dataset(
        config.dataset_name, config.dataset_dir, tmp_path
    ) == root.resolve()


@pytest.mark.parametrize("operation", ["--print-config", "--list-models"])
def test_inspection_commands_never_load_data_or_train(operation, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Configuration inspection must not start a run")

    monkeypatch.setattr(experiments, "resolve_dataset", forbidden)
    monkeypatch.setattr(experiments, "RunContext", forbidden)
    monkeypatch.setattr(experiments, "build_registered_model", forbidden)
    experiments.main(
        ["--config", str(CONFIG), "--minimum-epochs", "128", operation]
    )
    output = capsys.readouterr().out
    if operation == "--print-config":
        assert json.loads(output)["convlstm_layers"] == [
            [64, [3, 3]],
            [64, [3, 3]],
            [128, [3, 3]],
        ]
    else:
        assert MODEL in output
        assert "swin3d_t" not in output and "r3d_18" not in output


def test_existing_runner_keeps_approved_plateau_schedule():
    """The leaf schema names the scheduler; its factor/patience live in runner."""
    tree = ast.parse(Path(experiments.__file__).read_text(encoding="utf-8"))
    schedules = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "ReduceLROnPlateau"
    ]
    assert len(schedules) == 1
    options = {
        option.arg: ast.literal_eval(option.value)
        for option in schedules[0].keywords
    }
    assert options == {"mode": "min", "factor": 0.9, "patience": 5}
