"""Reusable helpers for the small, independent Colab notebooks."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import kagglehub
import torch

from src.dataset import AHARDataset, create_split_manifest, load_split_subsets
from src.metrics import validate_selected_checkpoint
from src.model import (
    CustomConvLSTM,
    count_trainable_parameters,
    custom_model_from_checkpoint,
)
from .config import (
    CONTROLLED_AAD_DATASET_HANDLE,
    DiagnosticDataset,
    selected_diagnostic_dataset,
)
from src.utils import runtime_environment, seed_everything
from src.vdd_diagnostic import inspect_vdd, resolve_vdd_root


def implementation_root(start: str | Path | None = None) -> Path:
    """Resolve the Paper 001 implementation directory from common entry points."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [
        current,
        current / "papers/001-journal-abnormal-activity-recognition/implementation",
        Path(
            "/content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation"
        ),
    ]
    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "requirements.txt").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Paper 001 implementation directory was not found")


def validate_runtime(require_cuda: bool = True) -> dict[str, object]:
    """Return runtime provenance and optionally require an available CUDA device."""
    if require_cuda and not torch.cuda.is_available():
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            gpu_description = gpu.stdout.strip() or gpu.stderr.strip()
            visible_gpu = gpu.returncode == 0 and bool(gpu.stdout.strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            gpu_description = str(error)
            visible_gpu = False
        if visible_gpu:
            advice = (
                "An NVIDIA GPU is attached, but this Python kernel cannot use CUDA. "
                "Restart this notebook's kernel after package installation and rerun setup. "
                "If it persists, reconnect to a fresh Colab GPU runtime."
            )
        else:
            advice = (
                "This notebook's kernel cannot see an NVIDIA GPU. In VS Code, use "
                "Select Kernel to connect THIS notebook to the Colab A100 runtime; "
                "another notebook may use a different runtime. Then rerun setup."
            )
        raise RuntimeError(
            f"{advice}\nPython: {sys.executable}\nPyTorch: {torch.__version__} "
            f"(CUDA build: {torch.version.cuda})\nNVIDIA: {gpu_description}"
        )
    environment = runtime_environment()
    environment["cuda_device"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    return environment


def download_selected_dataset() -> tuple[DiagnosticDataset, Path]:
    """Download and resolve the single shared diagnostic dataset selection."""
    specification = selected_diagnostic_dataset()
    download_root = Path(kagglehub.dataset_download(specification.kaggle_handle))
    return specification, resolve_vdd_root(download_root)


def load_selected_dataset(
    sequence_length: int = 16,
    frame_size: int = 32,
) -> tuple[DiagnosticDataset, AHARDataset]:
    """Build the selected dataset using only its declared accepted classes."""
    specification, dataset_root = download_selected_dataset()
    dataset = AHARDataset(
        dataset_root,
        sequence_length=sequence_length,
        frame_size=(frame_size, frame_size),
        accepted_classes=list(specification.accepted_classes),
    )
    return specification, dataset


def diagnostic_manifest_path(root: Path, dataset_key: str) -> Path:
    """Return an ignored, dataset-specific split-manifest path."""
    return root / "runs" / "manifests" / f"{dataset_key}_diagnostic_seed42.json"


def prepare_selected_splits(
    root: Path,
    sequence_length: int = 16,
    frame_size: int = 32,
) -> dict[str, object]:
    """Inventory the selected data and prepare train/validation/test indices."""
    specification, dataset = load_selected_dataset(sequence_length, frame_size)
    manifest_path = diagnostic_manifest_path(root, specification.key)
    if not manifest_path.is_file():
        create_split_manifest(dataset, manifest_path, seed=42)
    train, validation, test, split = load_split_subsets(dataset, manifest_path, seed=42)
    inventory = inspect_vdd(dataset)
    return {
        "specification": specification,
        "dataset": dataset,
        "manifest_path": manifest_path,
        "train": train,
        "validation": validation,
        "test_count_locked": len(test),
        "split": split,
        "inventory": inventory,
    }


def inspect_random_model(
    root: Path,
    sequence_length: int = 16,
    frame_size: int = 32,
) -> dict[str, object]:
    """Run one labelled random-weight prediction batch from training only."""
    prepared = prepare_selected_splits(root, sequence_length, frame_size)
    dataset = prepared["dataset"]
    train = prepared["train"]
    seed_everything(42)
    model = CustomConvLSTM(
        num_classes=dataset.num_classes,
        layers=[(8, (3, 3)), (16, (3, 3))],
    )
    indices = deterministic_subset_indices(train, count=5, seed=42)
    clips = torch.stack([dataset[index][0] for index in indices])
    labels = torch.tensor([dataset.samples[index][1] for index in indices])
    with torch.inference_mode():
        logits = model(clips)
        probabilities = logits.softmax(dim=1)
    predicted = logits.argmax(dim=1)
    return {
        "evidence_role": "random_weight_pipeline_check_only",
        "test_access": "locked",
        "dataset": prepared["specification"].key,
        "classes": dataset.class_names,
        "input_shape": list(clips.shape),
        "logit_shape": list(logits.shape),
        "examples": [
            {
                "path": str(dataset.samples[index][0]),
                "expected": dataset.class_names[int(labels[position])],
                "predicted": dataset.class_names[int(predicted[position])],
                "confidence": float(probabilities[position, predicted[position]]),
                "correct": bool(predicted[position] == labels[position]),
            }
            for position, index in enumerate(indices)
        ],
        "trainable_parameters": count_trainable_parameters(model),
        "model": model.configuration(),
    }


def deterministic_subset_indices(
    subset: object,
    count: int = 5,
    seed: int = 42,
) -> list[int]:
    """Select exact underlying dataset indices reproducibly from one subset."""
    indices = list(subset.indices)
    if len(indices) < count:
        raise ValueError(f"partition has {len(indices)} samples; {count} are required")
    return sorted(random.Random(seed).sample(indices, count))


def selected_training_examples(root: Path, count: int = 5) -> list[dict[str, object]]:
    """Return reproducible training examples with tensors for notebook display."""
    prepared = prepare_selected_splits(root)
    dataset = prepared["dataset"]
    indices = deterministic_subset_indices(prepared["train"], count=count, seed=42)
    examples = []
    for index in indices:
        clip, label = dataset[index]
        examples.append(
            {
                "path": str(dataset.samples[index][0]),
                "label": dataset.class_names[label],
                "shape": list(clip.shape),
                "preview_frame": clip[len(clip) // 2],
            }
        )
    return examples


def latest_completed_diagnostic_run(root: Path) -> Path:
    """Find the newest completed selected-dataset diagnostic run."""
    candidates: list[Path] = []
    for manifest_path in (root / "runs" / "train").glob("*/run.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("evidence_role") == "pipeline_learnability_diagnostic"
            and manifest.get("results", {}).get("tiny_passed") is True
        ):
            candidates.append(manifest_path.parent)
    if not candidates:
        raise FileNotFoundError("no completed learnability diagnostic run was found")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def validation_prediction_examples(
    root: Path,
    run_dir: str | Path | None = None,
    count: int = 5,
) -> dict[str, object]:
    """Load the selected diagnostic checkpoint and predict validation examples."""
    prepared = prepare_selected_splits(root)
    dataset = prepared["dataset"]
    selected_run = (
        Path(run_dir) if run_dir is not None else latest_completed_diagnostic_run(root)
    )
    checkpoint_path = selected_run / "checkpoints" / "bounded_validation_selected.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selection = validate_selected_checkpoint(
        checkpoint,
        dataset.dataset_dir.resolve().name,
        str(prepared["split"]["manifest_hash"]),
    )
    model = custom_model_from_checkpoint(checkpoint)
    indices = deterministic_subset_indices(prepared["validation"], count=count, seed=43)
    clips = torch.stack([dataset[index][0] for index in indices])
    labels = torch.tensor([dataset.samples[index][1] for index in indices])
    with torch.inference_mode():
        probabilities = model(clips).softmax(dim=1)
    predicted = probabilities.argmax(dim=1)
    return {
        "partition": "validation",
        "evidence_role": "pipeline_learnability_diagnostic",
        "checkpoint": str(checkpoint_path),
        "selection": selection,
        "examples": [
            {
                "path": str(dataset.samples[index][0]),
                "expected": dataset.class_names[int(labels[position])],
                "predicted": dataset.class_names[int(predicted[position])],
                "confidence": float(probabilities[position, predicted[position]]),
                "correct": bool(predicted[position] == labels[position]),
            }
            for position, index in enumerate(indices)
        ],
    }


def diagnostic_training_command(root: Path) -> list[str]:
    """Build the selected-dataset training command without running it."""
    specification = selected_diagnostic_dataset()
    download_root = Path(kagglehub.dataset_download(specification.kaggle_handle))
    command = [
        sys.executable,
        "-m",
        "src.vdd_diagnostic",
        "--dataset-dir",
        str(download_root),
        "--runs-dir",
        str(root / "runs"),
        "--manifest",
        str(diagnostic_manifest_path(root, specification.key)),
    ]
    for class_name in specification.accepted_classes:
        command.extend(["--accepted-class", class_name])
    return command


def controlled_plan_list_command(root: Path) -> list[str]:
    """Build the safe, fixed-AAD controlled-plan listing command."""
    return [
        sys.executable,
        "-m",
        "src.experiments",
        "--plan-config",
        str(root / "configs" / "experiments" / "aad_controlled_experiment_plan.json"),
        "--list-plan",
    ]


def controlled_stage_command(
    root: Path,
    stage: str,
    candidate_expansion_decision: str,
    reference_candidate: str | None = None,
) -> list[str]:
    """Build one guarded controlled-AAD stage command."""
    if stage == "architecture-screen" and candidate_expansion_decision == "pending":
        raise ValueError(
            "Record ticket 021 as skip or expanded before architecture screening"
        )
    if candidate_expansion_decision not in {"skip", "expanded"}:
        raise ValueError("candidate expansion decision must be skip or expanded")
    aad_download = Path(kagglehub.dataset_download(CONTROLLED_AAD_DATASET_HANDLE))
    aad_root = resolve_vdd_root(aad_download)
    command = [
        sys.executable,
        "-m",
        "src.experiments",
        "--plan-config",
        str(root / "configs" / "experiments" / "aad_controlled_experiment_plan.json"),
        "--run-plan-stage",
        stage,
        "--plan-dataset-dir",
        str(aad_root),
        "--plan-runs-dir",
        str(root / "runs"),
    ]
    if reference_candidate is not None:
        command.extend(["--reference-candidate", reference_candidate])
    return command


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validation_screen_winner(screen_run_dir: str | Path) -> dict[str, object]:
    """Validate a completed screen and return its validation-ranked winner."""
    run_dir = Path(screen_run_dir).resolve()
    manifest = _json_object(run_dir / "run.json")
    summary = _json_object(run_dir / "summary.json")
    if manifest.get("status") != "complete":
        raise ValueError("architecture screen is incomplete")
    ranked = summary.get("ranked")
    all_results = summary.get("all")
    if not isinstance(ranked, list) or not ranked or not isinstance(all_results, list):
        raise ValueError("architecture screen has no complete validation ranking")
    if len(ranked) != len(all_results):
        raise ValueError("architecture screen ranking is incomplete")
    provenance = summary.get("candidate_manifest")
    content = provenance.get("content") if isinstance(provenance, dict) else None
    candidates = content.get("candidates") if isinstance(content, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("architecture screen lacks candidate-manifest provenance")
    expected_names = {
        str(candidate.get("name"))
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    completed_names = {
        str(result.get("name")) for result in all_results if isinstance(result, dict)
    }
    if not expected_names or completed_names != expected_names:
        raise ValueError(
            "architecture screen did not complete every declared candidate"
        )
    if any(
        not isinstance(result, dict) or result.get("partition") != "validation"
        for result in ranked
    ):
        raise ValueError("architecture winner must be selected from validation results")
    winner = ranked[0]
    if not isinstance(winner.get("selected_checkpoint"), str):
        raise ValueError("winning validation result has no selected checkpoint")
    return winner


def confirmation_stage_command(
    root: Path,
    screen_run_dir: str | Path,
    candidate_expansion_decision: str,
) -> list[str]:
    """Build longer confirmation training for the validation-screen winner."""
    winner = validation_screen_winner(screen_run_dir)
    return controlled_stage_command(
        root,
        "baseline-confirmation",
        candidate_expansion_decision,
        reference_candidate=str(winner["name"]),
    )


def frozen_confirmation_checkpoint(
    screen_run_dir: str | Path,
    confirmation_run_dirs: list[str | Path],
) -> Path:
    """Select a compatible confirmed winner using validation evidence only."""
    winner = validation_screen_winner(screen_run_dir)
    winner_name = str(winner["name"])
    matching: list[dict[str, object]] = []
    seeds: set[int] = set()
    for value in confirmation_run_dirs:
        run_dir = Path(value).resolve()
        manifest = _json_object(run_dir / "run.json")
        summary = _json_object(run_dir / "summary.json")
        if manifest.get("status") != "complete":
            raise ValueError(f"confirmation run is incomplete: {run_dir}")
        results = summary.get("all")
        if not isinstance(results, list):
            raise ValueError(f"confirmation summary is incomplete: {run_dir}")
        for result in results:
            if isinstance(result, dict) and result.get("name") == winner_name:
                if result.get("partition") != "validation":
                    raise ValueError("confirmation candidate was not validated")
                selection = result.get("checkpoint_selection")
                if not isinstance(selection, dict) or not isinstance(
                    selection.get("seed"), int
                ):
                    raise ValueError("confirmation checkpoint provenance is incomplete")
                seeds.add(int(selection["seed"]))
                matching.append(result)
    if len(seeds) < 2:
        raise ValueError("both declared confirmation seeds must complete before test")
    if not matching:
        raise ValueError("confirmation outputs do not contain the screen winner")
    selected = sorted(
        matching,
        key=lambda result: (
            -float(result["validation_metrics"]["accuracy"]),
            float(result["validation_metrics"]["loss"]),
            int(result["num_params"]),
        ),
    )[0]
    checkpoint_path = Path(str(selected["selected_checkpoint"])).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"confirmed checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def final_test_evaluation_command(
    root: Path,
    checkpoint_path: str | Path,
    acknowledgement: str,
) -> list[str]:
    """Build the one-time final AAD test command after explicit acknowledgement."""
    if acknowledgement != "OPEN FINAL AAD TEST":
        raise ValueError(
            "set acknowledgement to 'OPEN FINAL AAD TEST' after freezing the model"
        )
    checkpoint = Path(checkpoint_path).resolve()
    for manifest_path in (root / "runs" / "evaluate").glob("*/run.json"):
        manifest = _json_object(manifest_path)
        if (
            manifest.get("status") == "complete"
            and Path(str(manifest.get("input_checkpoint", ""))).resolve() == checkpoint
        ):
            raise ValueError(
                f"this checkpoint already has a completed final evaluation: {manifest_path.parent}"
            )
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    experiment = metadata.get("experiment_config")
    if not isinstance(experiment, dict):
        raise ValueError("confirmed checkpoint lacks resolved experiment configuration")
    dataset_root = resolve_vdd_root(
        Path(kagglehub.dataset_download(CONTROLLED_AAD_DATASET_HANDLE))
    )
    manifest_path = Path(str(experiment["split_manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    dataset = AHARDataset(
        dataset_root,
        sequence_length=int(experiment["sequence_length"]),
        frame_size=(int(experiment["width"]), int(experiment["height"])),
    )
    _, _, _, split = load_split_subsets(
        dataset,
        manifest_path,
        train_ratio=float(experiment["train_ratio"]),
        val_ratio=float(experiment["val_ratio"]),
        seed=int(experiment["split_seed"]),
    )
    validate_selected_checkpoint(
        metadata,
        dataset.dataset_dir.resolve().name,
        str(split["manifest_hash"]),
    )
    if custom_model_from_checkpoint(metadata).num_classes != dataset.num_classes:
        raise ValueError("confirmed checkpoint class count does not match AAD")
    command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--dataset_dir",
        str(dataset_root),
        "--checkpoint_path",
        str(checkpoint),
        "--runs_dir",
        str(root / "runs"),
        "--split_manifest",
        str(manifest_path),
        "--seed",
        str(metadata["seed"]),
        "--batch_size",
        str(experiment["batch_size"]),
        "--sequence_length",
        str(experiment["sequence_length"]),
        "--height",
        str(experiment["height"]),
        "--width",
        str(experiment["width"]),
        "--train_ratio",
        str(experiment["train_ratio"]),
        "--val_ratio",
        str(experiment["val_ratio"]),
        "--num_samples",
        "5",
    ]
    return command


def final_test_report(evaluation_run_dir: str | Path) -> dict[str, object]:
    """Read current per-category exports or historical five-example reports."""
    run_dir = Path(evaluation_run_dir).resolve()
    manifest = _json_object(run_dir / "run.json")
    report = _json_object(run_dir / "metrics" / "final.json")
    if manifest.get("status") != "complete" or report.get("partition") != "test":
        raise ValueError("final test evaluation is incomplete")
    clips = report.get("artifacts", {}).get("prediction_clips")
    examples = report.get("prediction_examples")
    if examples is None:
        if not isinstance(clips, list) or len(clips) != 5:
            raise ValueError(
                "legacy final test evaluation must contain exactly five predictions"
            )
    else:
        limit = report.get("prediction_samples_per_category")
        if (
            type(limit) is not int
            or limit < 0
            or not isinstance(clips, list)
            or not isinstance(examples, dict)
            or examples.get("partition") != "test"
            or examples.get("prediction_samples_per_category") != limit
            or examples.get("records") != clips
            or any(
                not isinstance(item, dict)
                or item.get("partition") != "test"
                or type(item.get("correct")) is not bool
                for item in clips
            )
        ):
            raise ValueError("invalid per-category final test prediction provenance")
        counts = {
            "correct": sum(item["correct"] for item in clips),
            "incorrect": sum(not item["correct"] for item in clips),
        }
        if counts != examples.get("saved_counts") or any(
            value > limit for value in counts.values()
        ):
            raise ValueError(
                "final test prediction counts exceed or differ from their report"
            )
    return {
        "evidence_role": "final_test_evaluation",
        "partition": "test",
        "checkpoint": report.get("checkpoint"),
        "metrics": report.get("metrics"),
        "examples": clips,
    }


def run_command(command: list[str], root: Path) -> None:
    """Stream one existing module command and fail on a non-zero exit."""
    subprocess.run(command, cwd=root, check=True)


def printable(value: object) -> str:
    """Format notebook summaries as stable readable JSON."""
    return json.dumps(value, indent=2, default=str, sort_keys=True)
