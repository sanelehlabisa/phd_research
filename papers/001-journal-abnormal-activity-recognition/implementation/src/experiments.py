"""
experiments.py

Exploratory comparison runner for ConvLSTM and video-model baselines.
Saves all results to JSON.

Author: Sanele Hlabisa

.venv/bin/python -m src.experiments \
    --config configs/aad_screening_reference.json \
    --candidates-config configs/aad_architecture_candidates.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
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
) -> list[dict[str, object]]:
    """Describe the approved comparison models without allocating them.

    Parameters:
        experiment_config: Optional settings for the custom reference entry.
        candidate_manifest: Optional custom-only architecture screen.

    Returns:
        Model names, families, classes, and comparison roles.
    """
    if candidate_manifest is not None:
        return [
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
    return [
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
        if candidate is None:
            raise ValueError(f"unknown candidate model: {model_name}")
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
) -> None:
    """Print approved model names and roles without allocating models.

    Parameters:
        experiment_config: Optional settings for the custom reference entry.
        candidate_manifest: Optional custom-only architecture screen.

    Returns:
        None.
    """
    heading = (
        "Approved custom architecture candidates"
        if candidate_manifest is not None
        else "Approved comparison models"
    )
    print(heading)
    for entry in model_registry(experiment_config, candidate_manifest):
        print(
            f"- {entry['name']}: {entry['model_class']} | "
            f"{entry['family']} | {entry['role']}"
        )
        if candidate_manifest is not None:
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


def main(argv: list[str] | None = None) -> None:
    """Resolve configuration and run validation-only model comparisons."""
    args, experiment_config, candidate_manifest, print_only = _resolved_arguments(
        vars(parser.parse_args(argv))
    )
    if print_only:
        print(experiment_config.to_json())
        return
    registry = model_registry(experiment_config, candidate_manifest)
    if args.list_models:
        print_model_registry(experiment_config, candidate_manifest)
        return
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

    run = RunContext(
        args.runs_dir,
        purpose="experiments",
        dataset_path=args.dataset_dir,
        label=("architecture-screen" if candidate_manifest else "baseline-comparison"),
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
    print(
        f"Augmentation: {augmentation_state} | "
        f"Train samples: {len(train_set)} → {len(train_data)}"
    )

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
            "loss": "CrossEntropyLoss(label_smoothing=0.1)",
            "scheduler": args.scheduler,
            "metric_protocol": metric_protocol(),
            "checkpoint_selection": "lowest_validation_loss",
            "early_stopping_patience": args.early_stopping_patience,
            "ranking": ["validation_macro_f1", "validation_accuracy", "parameters"],
            "test_access": "locked",
        },
        "augmentation": args.augment,
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
        }
    )
    print(f"\nRunning {len(registry)} configurations...\n")
    all_results: list[dict[str, object]] = []

    for i, entry in enumerate(registry):
        name = entry["name"]
        role = entry["role"]
        if entry["model_class"] == "PaperConvLSTM" and (
            args.sequence_length,
            args.height,
            args.width,
        ) != (50, 50, 50):
            name = f"{name}_reduced_input"
            role = f"{role}; reduced-input topology check"
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
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
            "ranked": ranked,
        },
    )
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
