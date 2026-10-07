"""
experiments.py

Exploratory comparison runner for ConvLSTM and video-model baselines.
Saves all results to JSON.

Author: Sanele Hlabisa

.venv/bin/python -m src.experiments \
    --plan-config configs/aad_controlled_experiment_plan.json \
    --list-plan
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models.video as video_models

from .dataset import (
    AHARDataset,
    AugmentSubset,
    VideoAugmentation,
    load_split_subsets,
    resolve_split_manifest_path,
)
from .experiment_config import (
    CandidateManifest,
    ExperimentConfig,
    add_config_arguments,
    resolve_config_arguments,
)
from .model import CustomConvLSTM, PaperConvLSTM, count_trainable_parameters
from .metrics import (
    ValidationLossSelector,
    evaluate_classifier,
    metric_protocol,
    rank_validation_results,
    train_classifier_epoch,
    validate_selected_checkpoint,
)
from .utils import (
    RunContext,
    collect_predictions,
    data_loader_generator,
    plot_confusion_matrix,
    safe_filename,
    seed_data_loader_worker,
    seed_everything,
    write_json,
)

EXPERIMENT_DEFAULT_CONFIG = ExperimentConfig(
    sequence_length=16,
    height=32,
    width=32,
    epochs=24,
    batch_size=16,
    weight_decay=1e-3,
    num_workers=2,
    pin_memory=True,
    scheduler="none",
)

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
add_config_arguments(parser)
parser.add_argument(
    "--list-models",
    action="store_true",
    default=False,
    help="List approved baseline roles without loading data or starting training",
)
parser.add_argument(
    "--candidates-config",
    type=Path,
    help="Use a validated custom-only architecture candidate manifest",
)
parser.add_argument(
    "--model",
    dest="models",
    action="append",
    help="Repeat to run only explicitly named registered models",
)
parser.add_argument(
    "--confirmation-candidate",
    help="Compare one named custom candidate with the three practical baselines",
)
parser.add_argument("--plan-config", type=Path)
parser.add_argument("--list-plan", action="store_true", default=False)
parser.add_argument(
    "--run-plan-stage",
    choices=(
        "architecture-screen",
        "baseline-confirmation",
        "published-topology",
        "focused-ablations",
    ),
)
parser.add_argument("--plan-dataset-dir")
parser.add_argument("--plan-runs-dir")
parser.add_argument("--reference-candidate")
parser.add_argument("--run-label")
parser.add_argument("--trial-name")
parser.add_argument("--changed-factor")


def _require_exact_fields(
    values: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    """Reject missing or unknown fields in one controlled-plan section."""
    missing = sorted(expected - set(values))
    unknown = sorted(set(values) - expected)
    if missing:
        raise ValueError(f"{label} missing field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} unknown field(s): {', '.join(unknown)}")


def _resolve_plan_path(plan_path: Path, configured_path: object) -> Path:
    """Resolve one implementation-relative path from the controlled plan."""
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("controlled plan paths must be non-empty strings")
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = plan_path.resolve().parent.parent / path
    return path.resolve()


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    """Load one JSON object with a concise validation error."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a JSON object")
    return values


def _validate_protocol(
    config: ExperimentConfig,
    *,
    epochs: int,
    sequence_length: int,
    height: int,
    width: int,
    batch_size: int | None = None,
) -> None:
    """Validate fixed settings shared by one approved study stage."""
    expected = {
        "epochs": epochs,
        "sequence_length": sequence_length,
        "height": height,
        "width": width,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "augment": True,
        "early_stopping_patience": 10,
        "split_seed": 42,
        "scheduler": "reduce_on_plateau",
    }
    if batch_size is not None:
        expected["batch_size"] = batch_size
    for field_name, expected_value in expected.items():
        if getattr(config, field_name) != expected_value:
            raise ValueError(
                f"controlled {field_name} must be {expected_value!r}, got "
                f"{getattr(config, field_name)!r}"
            )


def load_controlled_plan(
    plan_path: str | Path,
) -> tuple[dict[str, object], CandidateManifest]:
    """Load and strictly validate the approved controlled experiment plan."""
    resolved_plan_path = Path(plan_path).expanduser().resolve()
    plan = _load_json_object(resolved_plan_path, "controlled plan")
    _require_exact_fields(
        plan,
        {
            "plan_id",
            "candidate_manifest",
            "reference_candidate",
            "screening",
            "confirmation",
            "published_topology",
            "ablations",
        },
        "controlled plan",
    )
    if plan["plan_id"] != "aad_controlled_experiment_suite":
        raise ValueError("unexpected controlled plan_id")

    candidate_path = _resolve_plan_path(resolved_plan_path, plan["candidate_manifest"])
    candidate_manifest = CandidateManifest.from_json(candidate_path)
    candidate_names = [candidate.name for candidate in candidate_manifest.candidates]
    reference_candidate = plan["reference_candidate"]
    if reference_candidate not in candidate_names:
        raise ValueError("reference_candidate must name a declared custom candidate")

    screening = plan["screening"]
    confirmation = plan["confirmation"]
    published = plan["published_topology"]
    ablations = plan["ablations"]
    for label, section in (
        ("screening", screening),
        ("confirmation", confirmation),
        ("published_topology", published),
        ("ablations", ablations),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"{label} must be a JSON object")

    _require_exact_fields(screening, {"config", "model_scope"}, "screening")
    _require_exact_fields(confirmation, {"config", "seeds", "models"}, "confirmation")
    _require_exact_fields(
        published,
        {"config", "models", "comparable_to_confirmation", "resource_note"},
        "published_topology",
    )
    _require_exact_fields(ablations, {"config", "seeds", "trials"}, "ablations")

    if screening["model_scope"] != "all_custom_candidates":
        raise ValueError("screening must include all custom candidates")
    if confirmation["seeds"] != [42, 2026] or ablations["seeds"] != [42, 2026]:
        raise ValueError("confirmation and ablation seeds must be [42, 2026]")
    if confirmation["models"] != [
        "reference_candidate",
        "r3d_18",
        "mc3_18",
        "r2plus1d_18",
    ]:
        raise ValueError("confirmation models must be the reference and 3D CNNs")
    if published["models"] != ["paper_convlstm_published"]:
        raise ValueError("published_topology must contain only PaperConvLSTM")
    if published["comparable_to_confirmation"] is not False:
        raise ValueError("published topology must be marked non-comparable")
    if (
        not isinstance(published["resource_note"], str)
        or not published["resource_note"].strip()
    ):
        raise ValueError("published_topology.resource_note must be non-empty")

    screening_config = ExperimentConfig.from_json(
        _resolve_plan_path(resolved_plan_path, screening["config"])
    )
    confirmation_config = ExperimentConfig.from_json(
        _resolve_plan_path(resolved_plan_path, confirmation["config"])
    )
    paper_config = ExperimentConfig.from_json(
        _resolve_plan_path(resolved_plan_path, published["config"])
    )
    ablation_config = ExperimentConfig.from_json(
        _resolve_plan_path(resolved_plan_path, ablations["config"])
    )
    _validate_protocol(
        screening_config,
        epochs=24,
        sequence_length=16,
        height=32,
        width=32,
        batch_size=16,
    )
    _validate_protocol(
        confirmation_config,
        epochs=64,
        sequence_length=16,
        height=32,
        width=32,
        batch_size=16,
    )
    _validate_protocol(
        ablation_config,
        epochs=64,
        sequence_length=16,
        height=32,
        width=32,
        batch_size=16,
    )
    _validate_protocol(
        paper_config,
        epochs=64,
        sequence_length=50,
        height=50,
        width=50,
        batch_size=1,
    )

    reference = next(
        candidate
        for candidate in candidate_manifest.candidates
        if candidate.name == reference_candidate
    )
    if confirmation_config.convlstm_layers != reference.convlstm_layers:
        raise ValueError("confirmation config must match the named reference candidate")
    if ablation_config.to_dict() != confirmation_config.to_dict():
        raise ValueError("ablation and confirmation reference configs must match")

    expected_trials = {
        "reference": (None, {}),
        "weight_decay_0": ("weight_decay", {"weight_decay": 0.0}),
        "weight_decay_0_0001": ("weight_decay", {"weight_decay": 0.0001}),
        "augmentation_off": ("augmentation", {"augment": False}),
        "spatial_64": ("spatial_size", {"height": 64, "width": 64}),
        "sequence_32": ("sequence_length", {"sequence_length": 32}),
    }
    trials = ablations["trials"]
    if not isinstance(trials, list) or len(trials) != len(expected_trials):
        raise ValueError("ablations must contain the six approved trials")
    seen_trials: set[str] = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("each ablation trial must be a JSON object")
        _require_exact_fields(
            trial, {"name", "changed_factor", "overrides"}, "ablation trial"
        )
        name = trial["name"]
        if name not in expected_trials or name in seen_trials:
            raise ValueError("ablation trial names must be unique and approved")
        seen_trials.add(name)
        expected_factor, expected_overrides = expected_trials[name]
        if (
            trial["changed_factor"] != expected_factor
            or trial["overrides"] != expected_overrides
        ):
            raise ValueError(
                f"ablation trial {name} must change exactly its approved factor"
            )
        resolved_values = ablation_config.to_dict()
        resolved_values.update(expected_overrides)
        resolved_trial = ExperimentConfig.from_mapping(resolved_values)
        if resolved_trial.learning_rate != 0.001 or resolved_trial.epochs != 64:
            raise ValueError("ablation learning rate and epoch budget must stay fixed")
    if seen_trials != set(expected_trials):
        raise ValueError("ablations must include every approved trial")
    return plan, candidate_manifest


def _append_config_overrides(command: list[str], overrides: dict[str, object]) -> None:
    """Append explicit experiment overrides to one command."""
    for field_name, value in overrides.items():
        option = f"--{field_name}"
        if isinstance(value, bool):
            command.append(option if value else f"--no-{field_name.replace('_', '-')}")
        else:
            command.extend([option, str(value)])


def build_plan_commands(
    plan_path: str | Path,
    stage: str,
    reference_candidate: str | None = None,
    dataset_dir: str | None = None,
    runs_dir: str | None = None,
) -> list[tuple[str, list[str]]]:
    """Build exact leaf commands for one approved controlled-plan stage."""
    resolved_plan_path = Path(plan_path).expanduser().resolve()
    plan, candidate_manifest = load_controlled_plan(resolved_plan_path)
    reference = reference_candidate or str(plan["reference_candidate"])
    candidate_names = [candidate.name for candidate in candidate_manifest.candidates]
    if reference not in candidate_names:
        raise ValueError("reference candidate is not present in the manifest")
    candidate_path = _resolve_plan_path(resolved_plan_path, plan["candidate_manifest"])

    def base_command(config_value: object, label: str) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "src.experiments",
            "--config",
            str(_resolve_plan_path(resolved_plan_path, config_value)),
            "--run-label",
            label,
        ]
        if dataset_dir:
            command.extend(["--dataset_dir", dataset_dir])
        if runs_dir:
            command.extend(["--runs_dir", runs_dir])
        return command

    commands: list[tuple[str, list[str]]] = []
    if stage == "architecture-screen":
        screening = plan["screening"]
        command = base_command(screening["config"], "architecture-screen")
        command.extend(["--candidates-config", str(candidate_path)])
        commands.append(("architecture_screen", command))
    elif stage == "baseline-confirmation":
        confirmation = plan["confirmation"]
        for seed in confirmation["seeds"]:
            command = base_command(confirmation["config"], "baseline-confirmation")
            command.extend(
                [
                    "--candidates-config",
                    str(candidate_path),
                    "--confirmation-candidate",
                    reference,
                    "--seed",
                    str(seed),
                ]
            )
            commands.append((f"baseline_confirmation_seed_{seed}", command))
    elif stage == "published-topology":
        published = plan["published_topology"]
        command = base_command(published["config"], "published-topology")
        command.extend(["--model", "paper_convlstm_published"])
        commands.append(("published_topology", command))
    elif stage == "focused-ablations":
        ablations = plan["ablations"]
        for trial in ablations["trials"]:
            for seed in ablations["seeds"]:
                command = base_command(ablations["config"], "focused-ablation")
                command.extend(
                    [
                        "--candidates-config",
                        str(candidate_path),
                        "--model",
                        reference,
                        "--seed",
                        str(seed),
                        "--trial-name",
                        str(trial["name"]),
                        "--changed-factor",
                        str(trial["changed_factor"] or "reference"),
                    ]
                )
                _append_config_overrides(command, trial["overrides"])
                commands.append((f"{trial['name']}_seed_{seed}", command))
    else:
        raise ValueError(f"unknown controlled plan stage: {stage}")
    return commands


def controlled_plan_rows(plan_path: str | Path) -> list[dict[str, object]]:
    """Return readable rows for every approved controlled-plan stage.

    Parameters:
        plan_path: Path to the validated controlled experiment plan.

    Returns:
        Stage, trial, model role, factor, protocol, and expected-run rows.
    """
    resolved_plan_path = Path(plan_path).expanduser().resolve()
    plan, candidate_manifest = load_controlled_plan(resolved_plan_path)

    def load_config(section: dict[str, object]) -> ExperimentConfig:
        return ExperimentConfig.from_json(
            _resolve_plan_path(resolved_plan_path, section["config"])
        )

    def input_size(config: ExperimentConfig) -> str:
        return f"T={config.sequence_length}, {config.height}x{config.width}"

    screening = plan["screening"]
    confirmation = plan["confirmation"]
    published = plan["published_topology"]
    ablations = plan["ablations"]
    screening_config = load_config(screening)
    confirmation_config = load_config(confirmation)
    published_config = load_config(published)
    ablation_config = load_config(ablations)
    reference = str(plan["reference_candidate"])

    rows: list[dict[str, object]] = [
        {
            "stage": "architecture-screen",
            "trial": "all_custom_candidates",
            "model_role": f"{len(candidate_manifest.candidates)} custom candidates",
            "changed_factor": "architecture width/depth",
            "seeds": str(screening_config.seed),
            "input_size": input_size(screening_config),
            "epochs": screening_config.epochs,
            "augmentation": screening_config.augment,
            "weight_decay": screening_config.weight_decay,
            "expected_runs": 1,
        },
        {
            "stage": "baseline-confirmation",
            "trial": "reference_and_practical_baselines",
            "model_role": (f"{reference} + r3d_18 + mc3_18 + r2plus1d_18"),
            "changed_factor": "model family",
            "seeds": ", ".join(str(seed) for seed in confirmation["seeds"]),
            "input_size": input_size(confirmation_config),
            "epochs": confirmation_config.epochs,
            "augmentation": confirmation_config.augment,
            "weight_decay": confirmation_config.weight_decay,
            "expected_runs": len(confirmation["seeds"]),
        },
        {
            "stage": "published-topology",
            "trial": "source_paper_topology",
            "model_role": "PaperConvLSTM (separate protocol)",
            "changed_factor": "published topology",
            "seeds": str(published_config.seed),
            "input_size": input_size(published_config),
            "epochs": published_config.epochs,
            "augmentation": published_config.augment,
            "weight_decay": published_config.weight_decay,
            "expected_runs": 1,
        },
    ]
    for trial in ablations["trials"]:
        resolved_values = ablation_config.to_dict()
        resolved_values.update(trial["overrides"])
        trial_config = ExperimentConfig.from_mapping(resolved_values)
        rows.append(
            {
                "stage": "focused-ablations",
                "trial": trial["name"],
                "model_role": reference,
                "changed_factor": trial["changed_factor"] or "reference",
                "seeds": ", ".join(str(seed) for seed in ablations["seeds"]),
                "input_size": input_size(trial_config),
                "epochs": trial_config.epochs,
                "augmentation": trial_config.augment,
                "weight_decay": trial_config.weight_decay,
                "expected_runs": len(ablations["seeds"]),
            }
        )
    return rows


def inspect_experiment_run_directories(
    run_directories: list[str | Path],
) -> list[dict[str, object]]:
    """Inspect exact experiment run directories without selecting results.

    Parameters:
        run_directories: Explicit run directories created by one requested stage.

    Returns:
        Missing, partial, or complete manifest and validation-summary records.
    """
    records: list[dict[str, object]] = []
    for value in run_directories:
        run_dir = Path(value).expanduser().resolve()
        manifest_path = run_dir / "run.json"
        summary_path = run_dir / "summary.json"
        if not manifest_path.is_file():
            records.append(
                {
                    "run_dir": str(run_dir),
                    "evidence_state": "missing",
                    "manifest": None,
                    "summary": None,
                }
            )
            continue

        manifest = _load_json_object(manifest_path, "run manifest")
        summary = (
            _load_json_object(summary_path, "experiment summary")
            if summary_path.is_file()
            else None
        )
        if summary is not None:
            ranked = summary.get("ranked", [])
            if not isinstance(ranked, list) or any(
                not isinstance(result, dict) or result.get("partition") != "validation"
                for result in ranked
            ):
                raise ValueError(
                    f"experiment summary must rank validation results: {summary_path}"
                )
        evidence_state = (
            "complete"
            if manifest.get("status") == "complete" and summary is not None
            else "partial"
        )
        records.append(
            {
                "run_dir": str(run_dir),
                "evidence_state": evidence_state,
                "manifest": manifest,
                "summary": summary,
            }
        )
    return records


def print_controlled_plan(plan_path: str | Path) -> None:
    """Print every approved stage and leaf command without starting a run."""
    plan, candidate_manifest = load_controlled_plan(plan_path)
    print(f"Controlled plan: {plan['plan_id']}")
    print(f"Reference candidate: {plan['reference_candidate']}")
    print(f"Custom candidates: {len(candidate_manifest.candidates)}")
    print(
        "Screening models: "
        + ", ".join(candidate.name for candidate in candidate_manifest.candidates)
    )
    print(
        "Screening protocol: seed=42, T=16, 32x32, epochs=24, lr=0.001, wd=0.001, augment=on"
    )
    print("Confirmation models: reference_candidate, r3d_18, mc3_18, " "r2plus1d_18")
    print("Confirmation protocol: seeds=42/2026, T=16, 32x32, epochs=64")
    print("Published topology: PaperConvLSTM only, T=50, 50x50, separate protocol")
    print(
        "Ablation factors: weight_decay=0/0.0001/0.001, augmentation=off/on, "
        "spatial_size=32x32/64x64, sequence_length=16/32"
    )
    print("Selection: validation only; test access: locked")
    for stage in (
        "architecture-screen",
        "baseline-confirmation",
        "published-topology",
        "focused-ablations",
    ):
        commands = build_plan_commands(plan_path, stage)
        print(f"\n{stage} ({len(commands)} run(s))")
        for name, command in commands:
            print(f"- {name}: {shlex.join(command)}")


def execute_plan_stage(
    plan_path: str | Path,
    stage: str,
    reference_candidate: str | None,
    dataset_dir: str | None,
    runs_dir: str | None,
) -> None:
    """Execute one explicitly requested controlled stage as checked leaf runs."""
    implementation_dir = Path(plan_path).expanduser().resolve().parent.parent
    commands = build_plan_commands(
        plan_path,
        stage,
        reference_candidate=reference_candidate,
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
    )
    for name, command in commands:
        print(f"\n[{name}] $ {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=implementation_dir, check=True)


def _resolved_arguments(
    values: dict[str, object],
) -> tuple[argparse.Namespace, ExperimentConfig, CandidateManifest | None, bool]:
    """Combine validated configuration with comparison-only flags."""
    config, operations, print_only, _ = resolve_config_arguments(
        values,
        EXPERIMENT_DEFAULT_CONFIG,
    )
    candidates_path = operations.pop("candidates_config", None)
    candidate_manifest = (
        CandidateManifest.from_json(candidates_path)
        if candidates_path is not None
        else None
    )
    combined = config.to_dict()
    combined.update(operations)
    return argparse.Namespace(**combined), config, candidate_manifest, print_only


class Video3DModelWrapper(nn.Module):
    """Wraps PyTorch 3D ResNet models to match our (B, T, C, H, W) input format."""

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.model = base_model
        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x is (B, T, C, H, W) -> PyTorch 3D CNNs expect (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


def model_registry(
    experiment_config: ExperimentConfig | None = None,
    candidate_manifest: CandidateManifest | None = None,
    confirmation_candidate: str | None = None,
    selected_models: list[str] | None = None,
) -> list[dict[str, object]]:
    """Describe the approved comparison models without allocating them.

    Parameters:
        experiment_config: Optional settings for the custom reference entry.
        candidate_manifest: Optional custom-only architecture screen.
        confirmation_candidate: Custom candidate compared with practical baselines.
        selected_models: Optional ordered subset of registered model names.

    Returns:
        Model names, families, classes, and comparison roles.
    """
    custom_entries: list[dict[str, object]] = []
    if candidate_manifest is not None:
        custom_entries = [
            {
                "name": candidate.name,
                "family": "ConvLSTM",
                "model_class": "CustomConvLSTM",
                "role": "controlled custom architecture candidate",
                "research_question": candidate.research_question,
                "convlstm_layers": candidate.to_dict()["convlstm_layers"],
                "hidden_classifier_width": candidate.hidden_classifier_width,
            }
            for candidate in candidate_manifest.candidates
        ]
    custom_name = "custom_convlstm_reference_8_k3"
    if experiment_config is not None and experiment_config.convlstm_layers != (
        (8, (3, 3)),
    ):
        custom_name = "custom_convlstm_configured"
    standard_entries = [
        {
            "name": "paper_convlstm_published",
            "family": "ConvLSTM",
            "model_class": "PaperConvLSTM",
            "role": "source-paper ConvLSTM topology baseline",
        },
        {
            "name": custom_name,
            "family": "ConvLSTM",
            "model_class": "CustomConvLSTM",
            "role": "study custom reference; not the selected final model",
        },
        {
            "name": "r3d_18",
            "family": "3D-CNN",
            "model_class": "torchvision.models.video.r3d_18",
            "role": "study practical baseline trained from scratch",
        },
        {
            "name": "mc3_18",
            "family": "3D-CNN",
            "model_class": "torchvision.models.video.mc3_18",
            "role": "study practical baseline trained from scratch",
        },
        {
            "name": "r2plus1d_18",
            "family": "3D-CNN",
            "model_class": "torchvision.models.video.r2plus1d_18",
            "role": "study practical baseline trained from scratch",
        },
    ]
    if candidate_manifest is None:
        registry = standard_entries
    elif confirmation_candidate is None:
        registry = custom_entries
    else:
        selected_candidate = next(
            (
                entry
                for entry in custom_entries
                if entry["name"] == confirmation_candidate
            ),
            None,
        )
        if selected_candidate is None:
            raise ValueError(
                f"unknown confirmation candidate: {confirmation_candidate}"
            )
        registry = [selected_candidate, *standard_entries[2:]]

    if selected_models is None:
        return registry
    if len(selected_models) != len(set(selected_models)):
        raise ValueError("--model names must be unique")
    by_name = {str(entry["name"]): entry for entry in registry}
    unknown = [name for name in selected_models if name not in by_name]
    if unknown:
        raise ValueError("unknown selected model(s): " + ", ".join(unknown))
    return [by_name[name] for name in selected_models]


def build_registered_model(
    model_name: str,
    num_classes: int,
    input_shape: tuple[int, int, int],
    sequence_length: int,
    experiment_config: ExperimentConfig | None = None,
    candidate_manifest: CandidateManifest | None = None,
) -> nn.Module:
    """Build one approved model by its registry name.

    Parameters:
        model_name: Exact name returned by `model_registry`.
        num_classes: Number of dataset classes.
        input_shape: Frame shape `(channels, height, width)`.
        sequence_length: Frames supplied to the model.
        experiment_config: Resolved settings for the custom reference.
        candidate_manifest: Optional custom-only architecture screen.

    Returns:
        The requested untrained model.
    """
    if candidate_manifest is not None:
        candidate = next(
            (item for item in candidate_manifest.candidates if item.name == model_name),
            None,
        )
        if candidate is not None:
            return CustomConvLSTM(
                num_classes,
                layers=list(candidate.convlstm_layers),
                hidden_classifier_width=candidate.hidden_classifier_width,
            )
    if model_name == "paper_convlstm_published":
        return PaperConvLSTM(
            num_classes,
            input_shape=input_shape,
            sequence_length=sequence_length,
        )
    if model_name in {
        "custom_convlstm_reference_8_k3",
        "custom_convlstm_configured",
    }:
        config = experiment_config or EXPERIMENT_DEFAULT_CONFIG
        return CustomConvLSTM(
            num_classes,
            layers=list(config.convlstm_layers),
            hidden_classifier_width=config.hidden_classifier_width,
        )
    if model_name == "r3d_18":
        return Video3DModelWrapper(
            video_models.r3d_18(weights=None),
            num_classes,
        )
    if model_name == "mc3_18":
        return Video3DModelWrapper(
            video_models.mc3_18(weights=None),
            num_classes,
        )
    if model_name == "r2plus1d_18":
        return Video3DModelWrapper(
            video_models.r2plus1d_18(weights=None),
            num_classes,
        )
    raise ValueError(f"unknown registered model: {model_name}")


def _format_layers(raw_layers: object) -> str:
    """Format one validated layer stack for concise terminal output."""
    if not isinstance(raw_layers, list):
        return "configured by shared experiment config"
    return " -> ".join(
        f"{layer[0]}x{layer[1][0]}x{layer[1][1]}" for layer in raw_layers
    )


def print_model_registry(
    experiment_config: ExperimentConfig | None = None,
    candidate_manifest: CandidateManifest | None = None,
    confirmation_candidate: str | None = None,
    selected_models: list[str] | None = None,
) -> None:
    """Print approved model names and roles without allocating models.

    Parameters:
        experiment_config: Optional settings for the custom reference entry.
        candidate_manifest: Optional custom-only architecture screen.
        confirmation_candidate: Custom candidate compared with practical baselines.
        selected_models: Optional ordered subset of registered model names.

    Returns:
        None.
    """
    if confirmation_candidate is not None:
        heading = "Approved practical-baseline confirmation models"
    elif candidate_manifest is not None:
        heading = "Approved custom architecture candidates"
    else:
        heading = "Approved comparison models"
    print(heading)
    for entry in model_registry(
        experiment_config,
        candidate_manifest,
        confirmation_candidate,
        selected_models,
    ):
        print(
            f"- {entry['name']}: {entry['model_class']} | "
            f"{entry['family']} | {entry['role']}"
        )
        if "research_question" in entry:
            print(f"  architecture: {_format_layers(entry['convlstm_layers'])}")
            print(f"  question: {entry['research_question']}")


def _save_selected_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    model_config: dict[str, object],
    model_registry_entry: dict[str, object],
    validation_metrics: dict[str, float],
    selected_epoch: int,
    dataset_name: str,
    split_manifest_hash: str,
    seed: int,
    experiment_config: dict[str, object],
    candidate_manifest: dict[str, object] | None,
) -> None:
    """Save one experiment model selected by validation loss.

    Parameters:
        model: Model whose selected weights should be saved.
        optimizer: Optimizer state associated with the selected epoch.
        checkpoint_path: Destination checkpoint path.
        model_config: Configuration identifying the model architecture.
        model_registry_entry: Approved registry record for the model.
        validation_metrics: Metrics from the full validation partition.
        selected_epoch: One-based selected epoch.
        dataset_name: Dataset used for the comparison.
        split_manifest_hash: Exact split manifest hash used by the run.
        seed: Experiment run seed.
        experiment_config: Exact normalised experiment configuration.
        candidate_manifest: Exact manifest content and hash, when supplied.

    Returns:
        None.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "checkpoint_role": "validation_selected_lowest_loss",
        "selection_partition": "validation",
        "selection_metric": "loss",
        "selection_value": validation_metrics["loss"],
        "selected_epoch": selected_epoch,
        "epoch": selected_epoch,
        "loss": validation_metrics["loss"],
        "validation_metrics": validation_metrics,
        "metric_protocol": metric_protocol(),
        "dataset_name": dataset_name,
        "split_manifest_hash": split_manifest_hash,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "trainable_parameters": count_trainable_parameters(model),
        "model_config": model_config,
        "model_registry_entry": model_registry_entry,
        "experiment_config": experiment_config,
        "candidate_manifest": candidate_manifest,
    }
    torch.save(
        {
            **metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    write_json(path.with_suffix(".json"), metadata)


def build_experiment_criterion() -> nn.CrossEntropyLoss:
    """Build the loss declared by the controlled experiment configurations.

    Parameters:
        None.

    Returns:
        Plain cross-entropy without label smoothing.
    """
    return nn.CrossEntropyLoss()


def main(argv: list[str] | None = None) -> None:
    """Resolve configuration and run validation-only model comparisons."""
    args, experiment_config, candidate_manifest, print_only = _resolved_arguments(
        vars(parser.parse_args(argv))
    )
    if print_only:
        print(experiment_config.to_json())
        return
    plan_path = getattr(args, "plan_config", None)
    list_plan = getattr(args, "list_plan", False)
    plan_stage = getattr(args, "run_plan_stage", None)
    if plan_path is not None:
        if list_plan:
            print_controlled_plan(plan_path)
            return
        if plan_stage is not None:
            execute_plan_stage(
                plan_path,
                plan_stage,
                getattr(args, "reference_candidate", None),
                getattr(args, "plan_dataset_dir", None),
                getattr(args, "plan_runs_dir", None),
            )
            return
        raise ValueError("--plan-config requires --list-plan or --run-plan-stage")
    if list_plan or plan_stage is not None:
        raise ValueError("--list-plan and --run-plan-stage require --plan-config")

    confirmation_candidate = getattr(args, "confirmation_candidate", None)
    selected_models = getattr(args, "models", None)
    if confirmation_candidate is not None and candidate_manifest is None:
        raise ValueError("--confirmation-candidate requires --candidates-config")
    registry = model_registry(
        experiment_config,
        candidate_manifest,
        confirmation_candidate,
        selected_models,
    )
    if args.list_models:
        print_model_registry(
            experiment_config,
            candidate_manifest,
            confirmation_candidate,
            selected_models,
        )
        return
    if any(entry["model_class"] == "PaperConvLSTM" for entry in registry) and (
        args.sequence_length,
        args.height,
        args.width,
    ) != (50, 50, 50):
        raise ValueError(
            "PaperConvLSTM must use its native 50-frame 50x50 protocol; "
            "run it as the separate published-topology stage"
        )
    candidate_provenance = (
        candidate_manifest.provenance() if candidate_manifest is not None else None
    )
    deterministic_settings = seed_everything(args.seed)
    deterministic_settings["data_loader_seeds"] = {
        "train": args.seed,
        "validation": args.seed + 1,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = resolve_split_manifest_path(
        args.dataset_dir, args.split_manifest, args.split_seed
    )

    run_label = getattr(args, "run_label", None)
    if run_label is None:
        run_label = (
            "architecture-screen" if candidate_manifest else "baseline-comparison"
        )
    trial_metadata = {
        "trial_name": getattr(args, "trial_name", None),
        "changed_factor": getattr(args, "changed_factor", None),
    }
    run = RunContext(
        args.runs_dir,
        purpose="experiments",
        dataset_path=args.dataset_dir,
        label=run_label,
        arguments=vars(args),
        metadata={
            "experiment_config": experiment_config.to_dict(),
            "candidate_manifest": candidate_provenance,
            "input_dimensions": {
                "sequence_length": args.sequence_length,
                "channels": 3,
                "height": args.height,
                "width": args.width,
            },
            "augmentation": args.augment,
            "split_ratios": {
                "train": args.train_ratio,
                "validation": args.val_ratio,
                "test": args.test_ratio,
            },
            "seed": args.seed,
            "split_seed": args.split_seed,
            "split_manifest": str(manifest_path.resolve()),
            "deterministic_settings": deterministic_settings,
            "device": str(device),
            "study_trial": trial_metadata,
        },
    )
    run_dir = run.run_dir
    print(f"Run directory: {run_dir}")

    resolved_config_path = experiment_config.save_json(run_dir / "resolved_config.json")
    resolved_candidates_path = None
    if candidate_provenance is not None:
        resolved_candidates_path = write_json(
            run_dir / "resolved_candidates.json",
            candidate_provenance,
        )

    # ---- Dataset ----
    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    num_classes = dataset.num_classes
    print(
        f"Loaded {len(dataset)} samples | {num_classes} classes: {dataset.class_names}"
    )

    train_set, val_set, _, split_metadata = load_split_subsets(
        dataset,
        manifest_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    n_total = len(dataset)
    n_train, n_val = len(train_set), len(val_set)
    n_test = n_total - n_train - n_val
    print(f"Train: {n_train} | Val: {n_val} | Test locked: {n_test}")

    train_data = (
        AugmentSubset(train_set, VideoAugmentation()) if args.augment else train_set
    )
    augmentation_state = "enabled" if args.augment else "disabled"
    augmentation_detail = (
        " (one transform or none per clip)" if args.augment else ""
    )
    print(f"Online augmentation: {augmentation_state}{augmentation_detail}")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_data_loader_worker,
    )
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=data_loader_generator(args.seed),
        **loader_kw,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        generator=data_loader_generator(args.seed + 1),
        **loader_kw,
    )
    # ---- Model configs ----
    input_shape = (3, args.height, args.width)
    shared_configuration = {
        "experiment_config": experiment_config.to_dict(),
        "candidate_manifest": candidate_provenance,
        "models": registry,
        "class_names": dataset.class_names,
        "dataset_size": n_total,
        "split_sizes": {"train": n_train, "validation": n_val, "test": n_test},
        "split": split_metadata,
        "deterministic_settings": deterministic_settings,
        "input_dimensions": {
            "sequence_length": args.sequence_length,
            "channels": 3,
            "height": args.height,
            "width": args.width,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "scheduler": args.scheduler,
            "metric_protocol": metric_protocol(),
            "checkpoint_selection": "lowest_validation_loss",
            "early_stopping_patience": args.early_stopping_patience,
            "ranking": ["validation_macro_f1", "validation_accuracy", "parameters"],
            "test_access": "locked",
        },
        "augmentation": args.augment,
        "study_trial": trial_metadata,
    }
    config_path = write_json(run_dir / "config.json", shared_configuration)
    run.update(
        {
            "class_names": dataset.class_names,
            "model_configuration": registry,
            "experiment_config": experiment_config.to_dict(),
            "candidate_manifest": candidate_provenance,
            "split_sizes": shared_configuration["split_sizes"],
            "split": split_metadata,
            "deterministic_settings": deterministic_settings,
            "study_trial": trial_metadata,
        }
    )
    print(f"\nRunning {len(registry)} configurations...\n")
    all_results: list[dict[str, object]] = []

    for i, entry in enumerate(registry):
        name = entry["name"]
        role = entry["role"]
        model = build_registered_model(
            str(entry["name"]),
            num_classes,
            input_shape,
            args.sequence_length,
            experiment_config,
            candidate_manifest,
        )
        model = model.to(device)
        num_params = count_trainable_parameters(model)
        model_dir = run_dir / "models" / safe_filename(name)
        print(f"[{i+1}/{len(registry)}] {name} | params={num_params:,}")

        opt = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.9, patience=5
            )
            if args.scheduler == "reduce_on_plateau"
            else None
        )
        criterion = build_experiment_criterion()
        model_config = (
            model.configuration()
            if isinstance(model, CustomConvLSTM)
            else {
                "model_name": entry["model_class"],
                "registry_name": entry["name"],
                "num_classes": num_classes,
                "input_dimensions": shared_configuration["input_dimensions"],
            }
        )
        selector = ValidationLossSelector(args.early_stopping_patience)
        history: list[dict[str, object]] = []
        selected_checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
        t0 = timer()

        for epoch in tqdm(range(args.epochs), leave=False, desc=name):
            training_metrics = train_classifier_epoch(
                model,
                train_loader,
                criterion,
                opt,
                device,
                num_classes,
            )
            validation_metrics = evaluate_classifier(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
            )
            if scheduler is not None:
                scheduler.step(validation_metrics["loss"])
            selected = selector.update(validation_metrics["loss"], epoch + 1)
            history.append(
                {
                    "epoch": epoch + 1,
                    "training_metrics": training_metrics,
                    "validation_metrics": validation_metrics,
                    "selected_checkpoint": selected,
                }
            )
            if selected:
                _save_selected_checkpoint(
                    model,
                    opt,
                    str(selected_checkpoint_path),
                    model_config,
                    entry,
                    validation_metrics,
                    epoch + 1,
                    dataset.dataset_dir.resolve().name,
                    str(split_metadata["manifest_hash"]),
                    args.seed,
                    experiment_config.to_dict(),
                    candidate_provenance,
                )
            if selector.should_stop:
                break

        elapsed = timer() - t0
        selection_state = selector.state(len(history))
        checkpoint = torch.load(
            selected_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        checkpoint["early_stopping"] = selection_state
        torch.save(checkpoint, selected_checkpoint_path)
        write_json(
            selected_checkpoint_path.with_suffix(".json"),
            {
                key: value
                for key, value in checkpoint.items()
                if key not in {"model_state_dict", "optimizer_state_dict"}
            },
        )
        checkpoint_selection = validate_selected_checkpoint(
            checkpoint,
            dataset.dataset_dir.resolve().name,
            str(split_metadata["manifest_hash"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        selected_validation_metrics = evaluate_classifier(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
        )
        all_true, all_pred = collect_predictions(model, val_loader, device)

        confusion_artifacts = plot_confusion_matrix(
            all_true,
            all_pred,
            dataset.class_names,
            dataset_name=f"{name}_validation",
            save_path=model_dir / "metrics" / "validation_confusion_matrix.png",
        )
        history_path = write_json(
            model_dir / "metrics" / "history.json",
            {
                "metric_protocol": metric_protocol(),
                "selection": selection_state,
                "epochs": history,
            },
        )

        result = {
            "name": name,
            "family": entry["family"],
            "model_class": entry["model_class"],
            "role": role,
            "num_params": num_params,
            "experiment_config": experiment_config.to_dict(),
            "candidate_manifest": candidate_provenance,
            "study_trial": trial_metadata,
            "partition": "validation",
            "metric_protocol": metric_protocol(),
            "validation_metrics": selected_validation_metrics,
            "checkpoint_selection": checkpoint_selection,
            "early_stopping": selection_state,
            "train_time_s": round(elapsed, 1),
            "validation_confusion_matrix": confusion_artifacts,
            "selected_checkpoint": str(selected_checkpoint_path),
            "selected_checkpoint_metadata": str(
                selected_checkpoint_path.with_suffix(".json")
            ),
            "history": str(history_path),
            "test_access": "locked",
        }
        metrics_path = write_json(model_dir / "metrics" / "metrics.json", result)
        result["metrics_path"] = str(metrics_path)
        all_results.append(result)
        print(
            f"  val_f1={selected_validation_metrics['macro_f1']:.4f}  "
            f"val_acc={selected_validation_metrics['accuracy']:.4f}  "
            f"selected_epoch={checkpoint_selection['selected_epoch']}  "
            f"time={elapsed:.0f}s"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Rank and save ----
    ranked = rank_validation_results(all_results)
    shared_configuration["completed_models"] = {
        str(result["name"]): {
            "checkpoint_selection": result["checkpoint_selection"],
            "early_stopping": result["early_stopping"],
        }
        for result in all_results
    }
    write_json(config_path, shared_configuration)

    print("\nTop 5 (validation macro-F1, accuracy, fewest parameters):")
    print("-" * 70)
    for r in ranked[:5]:
        validation_metrics = r["validation_metrics"]
        print(
            f"  {r['name']:<30} val_f1={validation_metrics['macro_f1']:.4f}  "
            f"val_acc={validation_metrics['accuracy']:.4f}  "
            f"params={r['num_params']:,}"
        )

    summary = {
        "experiment_config": experiment_config.to_dict(),
        "candidate_manifest": candidate_provenance,
        "study_trial": trial_metadata,
        "ranking": [
            "validation_macro_f1",
            "validation_accuracy",
            "parameters",
        ],
        "ranked": ranked,
        "top_five": ranked[:5],
        "all": all_results,
    }
    out_path = write_json(run_dir / "summary.json", summary)
    run.update(
        {
            "metric_protocol": metric_protocol(),
            "ranking": [
                "validation_macro_f1",
                "validation_accuracy",
                "parameters",
            ],
            "completed_models": shared_configuration["completed_models"],
            "experiment_config": experiment_config.to_dict(),
            "candidate_manifest": candidate_provenance,
            "study_trial": trial_metadata,
            "test_access": "locked",
        }
    )
    run.complete(
        artifacts={
            "resolved_configuration": str(resolved_config_path),
            "resolved_candidates": (
                str(resolved_candidates_path) if resolved_candidates_path else None
            ),
            "configuration": str(config_path),
            "ranked_summary": str(out_path),
            "model_directories": {
                result["name"]: str(run_dir / "models" / safe_filename(result["name"]))
                for result in all_results
            },
        },
        results={
            "experiment_config": experiment_config.to_dict(),
            "candidate_manifest": candidate_provenance,
            "study_trial": trial_metadata,
            "ranked": ranked,
        },
    )
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
