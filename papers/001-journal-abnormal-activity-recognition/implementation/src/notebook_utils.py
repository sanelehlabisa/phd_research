"""Reusable helpers for the small, independent Colab notebooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import kagglehub
import torch

from .dataset import AHARDataset, create_split_manifest, load_split_subsets
from .model import CustomConvLSTM, count_trainable_parameters
from .notebook_config import (
    CONTROLLED_AAD_DATASET_HANDLE,
    DiagnosticDataset,
    selected_diagnostic_dataset,
)
from .utils import runtime_environment, seed_everything
from .vdd_diagnostic import inspect_vdd, resolve_vdd_root


def implementation_root(start: str | Path | None = None) -> Path:
    """Resolve the Paper 001 implementation directory from common entry points."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [
        current,
        current / "papers/001-journal-abnormal-activity-recognition/implementation",
        Path("/content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation"),
    ]
    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "requirements.txt").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Paper 001 implementation directory was not found")


def validate_runtime(require_cuda: bool = True) -> dict[str, object]:
    """Return runtime provenance and optionally require an available CUDA device."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; connect the notebook to the Colab A100")
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
    train, validation, test, split = load_split_subsets(
        dataset, manifest_path, seed=42
    )
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
    indices = list(train.indices[: min(3, len(train))])
    clips = torch.stack([dataset[index][0] for index in indices])
    labels = torch.tensor([dataset.samples[index][1] for index in indices])
    with torch.inference_mode():
        logits = model(clips)
    return {
        "evidence_role": "random_weight_pipeline_check_only",
        "test_access": "locked",
        "dataset": prepared["specification"].key,
        "classes": dataset.class_names,
        "input_shape": list(clips.shape),
        "logit_shape": list(logits.shape),
        "labels": labels.tolist(),
        "predictions": logits.argmax(dim=1).tolist(),
        "trainable_parameters": count_trainable_parameters(model),
        "model": model.configuration(),
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
        str(root / "configs" / "aad_controlled_experiment_plan.json"),
        "--list-plan",
    ]


def controlled_stage_command(
    root: Path,
    stage: str,
    candidate_expansion_decision: str,
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
    return [
        sys.executable,
        "-m",
        "src.experiments",
        "--plan-config",
        str(root / "configs" / "aad_controlled_experiment_plan.json"),
        "--run-plan-stage",
        stage,
        "--plan-dataset-dir",
        str(aad_root),
        "--plan-runs-dir",
        str(root / "runs"),
    ]


def run_command(command: list[str], root: Path) -> None:
    """Stream one existing module command and fail on a non-zero exit."""
    subprocess.run(command, cwd=root, check=True)


def printable(value: object) -> str:
    """Format notebook summaries as stable readable JSON."""
    return json.dumps(value, indent=2, default=str, sort_keys=True)
