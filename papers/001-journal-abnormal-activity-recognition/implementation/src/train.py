"""
train.py

Training script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR).

Author: Sanele Hlabisa

.venv/bin/python -m src.train \
    --config configs/aad_screening_reference.json
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import (
    AHARDataset,
    AugmentSubset,
    CachedAHARDataset,
    VideoAugmentation,
    load_split_subsets,
    resolve_split_manifest_path,
)
from .experiment_config import (
    ExperimentConfig,
    add_config_arguments,
    resolve_config_arguments,
)
from .model import (
    CustomConvLSTM,
    count_trainable_parameters,
    custom_model_from_checkpoint,
    parse_layer_arguments,
)
from .metrics import (
    ValidationLossSelector,
    evaluate_classifier,
    metric_protocol,
    train_classifier_epoch,
    validate_selected_checkpoint,
)
from .utils import (
    RunContext,
    collect_predictions,
    data_loader_generator,
    plot_confusion_matrix,
    plot_training_curves,
    seed_data_loader_worker,
    seed_everything,
    write_json,
)

TRAIN_DEFAULT_CONFIG = ExperimentConfig()

parser = argparse.ArgumentParser(
    description="Train ConvLSTM for AHAR",
    argument_default=argparse.SUPPRESS,
)
add_config_arguments(parser)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume training same dataset, no layer changes",
)
parser.add_argument(
    "--finetune_last",
    action="store_true",
    default=False,
    help="Freeze all except the classifier",
)
parser.add_argument(
    "--finetune_full",
    action="store_true",
    default=False,
    help="Load weights, unfreeze everything, train all layers",
)


def _resolved_arguments(
    values: dict[str, object],
) -> tuple[argparse.Namespace, ExperimentConfig, bool]:
    """Combine validated configuration with operational training flags."""
    explicit_layers = "convlstm_layers" in values
    config, operations, print_only, config_loaded = resolve_config_arguments(
        values,
        TRAIN_DEFAULT_CONFIG,
    )
    combined = config.to_dict()
    combined["convlstm_layer"] = (
        [[filters, kernel[0], kernel[1]] for filters, kernel in config.convlstm_layers]
        if config_loaded or explicit_layers
        else None
    )
    combined.update(operations)
    return argparse.Namespace(**combined), config, print_only


def _save_custom_checkpoint(
    model: CustomConvLSTM,
    optimizer: optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
    checkpoint_path: Path,
    dataset_name: str,
    split_manifest_hash: str,
    seed: int,
    experiment_config: dict[str, object],
) -> None:
    """Save one validation-selected checkpoint with complete provenance.

    Parameters:
        model: Custom model being trained.
        optimizer: Optimizer whose state should be saved.
        epoch: Current training epoch.
        validation_metrics: Metrics from the complete validation partition.
        checkpoint_path: Destination checkpoint path.
        dataset_name: Dataset used for training and validation.
        split_manifest_hash: Exact split manifest hash used by the run.
        seed: Training run seed.
        experiment_config: Exact normalised experiment configuration.

    Returns:
        None.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = model.configuration()
    metadata: dict[str, object] = {
        "checkpoint_role": "validation_selected_lowest_loss",
        "selection_partition": "validation",
        "selection_metric": "loss",
        "selection_value": validation_metrics["loss"],
        "selected_epoch": epoch,
        "epoch": epoch,
        "loss": validation_metrics["loss"],
        "validation_metrics": validation_metrics,
        "metric_protocol": metric_protocol(),
        "dataset_name": dataset_name,
        "split_manifest_hash": split_manifest_hash,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "trainable_parameters": count_trainable_parameters(model),
        "model_config": configuration,
        "experiment_config": experiment_config,
    }
    torch.save(
        {
            **metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    write_json(checkpoint_path.with_suffix(".json"), metadata)
    print(f"✅ Saved checkpoint to {checkpoint_path}")


def train_notebook_model(
    prepared,
    config: ExperimentConfig,
    label: str,
    on_epoch=None,
    *,
    model_spec=None,
    deadline=None,
):
    """Train an exploratory model on explicit train/validation subsets only."""
    from time import perf_counter
    from notebooks.utils.data import data_identity
    from notebooks.utils import config as settings
    from notebooks.utils.models import build_model
    import time

    def check_budget():
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(
                "Diagnostic compute budget exhausted; evidence is partial"
            )

    check_budget()

    deterministic = seed_everything(config.seed)
    dataset = prepared["dataset"]
    identity = data_identity(prepared)
    if (config.height, config.width) != dataset.frame_size:
        raise ValueError("configuration and dataset frame sizes differ")
    if config.sequence_length != dataset.sequence_length or config.num_workers != 0:
        raise ValueError(
            "Notebook training requires the prepared frame count and num_workers=0"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RunContext(
        config.runs_dir,
        "train",
        dataset.dataset_dir,
        label,
        config.to_dict(),
        {
            "evidence_role": "exploratory_diagnostic",
            "data_identity": identity,
            "determinism": deterministic,
        },
    )
    if prepared.get("suite_run_dir") is not None:
        run_directory = Path(run.run_dir)
        if not run_directory.is_absolute():
            run_directory = Path(prepared["root"]) / run_directory
        prepared.setdefault("suite_train_run_dirs", []).append(run_directory)
    config.save_json(run.run_dir / "config.json")
    write_json(run.run_dir / "split.json", prepared["split"])
    model_spec = model_spec or dict(
        name="custom",
        layers=config.to_dict()["convlstm_layers"],
        dropout=settings.NOTEBOOK_DROPOUT,
        hidden_classifier_width=config.hidden_classifier_width,
    )
    try:
        model = build_model(
            model_spec,
            dataset.num_classes,
            (3, config.height, config.width),
            config.sequence_length,
        ).to(device)
    except BaseException as error:
        run.update({"status": "failed", "error": f"Model allocation: {error}"})
        raise
    run.update(
        {
            "notebook_model": model_spec,
            "scheduler_parameters": dict(
                factor=settings.LR_FACTOR,
                patience=settings.LR_PATIENCE,
                min_lr=settings.MIN_LR,
            ),
        }
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=settings.LR_PATIENCE,
            factor=settings.LR_FACTOR,
            min_lr=settings.MIN_LR,
        )
        if config.scheduler == "reduce_on_plateau"
        else None
    )
    criterion = nn.CrossEntropyLoss()
    training = (
        AugmentSubset(prepared["train"], VideoAugmentation())
        if config.augment
        else prepared["train"]
    )
    loader = DataLoader(
        training,
        batch_size=config.batch_size,
        shuffle=True,
        generator=data_loader_generator(config.seed),
        num_workers=0,
        pin_memory=config.pin_memory,
    )
    validation = DataLoader(
        prepared["validation"],
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=config.pin_memory,
    )
    selector = ValidationLossSelector(config.early_stopping_patience)
    checkpoint_path = run.run_dir / "checkpoints" / "best_model.pth"
    history = []
    last_print = 0.0
    gradient_norm = None

    def progress(done, total, loss):
        nonlocal last_print, gradient_norm
        check_budget()
        if model.training:
            gradients = [
                p.grad.detach().norm() for p in model.parameters() if p.grad is not None
            ]
            gradient_norm = (
                float(torch.stack(gradients).norm().item()) if gradients else 0.0
            )
            if not math.isfinite(gradient_norm):
                raise ValueError(
                    "Non-finite gradients; inspect saved input diagnostics"
                )
        now = perf_counter()
        if now - last_print >= 10 or done == total:
            print(
                f"  {label}: {done}/{total} clips | batch loss={loss:.4f}", flush=True
            )
            last_print = now

    print(
        f"Training {label} on {device}: {config.epochs} epochs, "
        f"{config.height}x{config.width}, {config.convlstm_layers}",
        flush=True,
    )
    print(f"Artifacts: {run.run_dir}", flush=True)
    try:
        for epoch in range(1, config.epochs + 1):
            check_budget()
            epoch_started = perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            tracked_parameter = next(model.parameters())
            before_update = tracked_parameter.detach().flatten()[:1024].clone()
            dataset.set_epoch(epoch, config.seed, training=True)
            print(f"Epoch {epoch}/{config.epochs}", flush=True)
            train_metrics = train_classifier_epoch(
                model,
                loader,
                criterion,
                optimizer,
                device,
                dataset.num_classes,
                progress,
            )
            dataset.set_epoch(epoch, config.seed, training=False)
            print("  Validating complete validation partition...", flush=True)
            validation_metrics = evaluate_classifier(
                model, validation, criterion, device, dataset.num_classes, progress
            )
            history.append(
                dict(
                    epoch=epoch,
                    train=train_metrics,
                    validation=validation_metrics,
                    learning_rate=learning_rate,
                    gradient_norm=gradient_norm,
                    parameter_sample_update_norm=float(
                        (tracked_parameter.detach().flatten()[:1024] - before_update)
                        .norm()
                        .item()
                    ),
                    seconds=perf_counter() - epoch_started,
                )
            )
            if selector.update(validation_metrics["loss"], epoch):
                _save_custom_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    validation_metrics,
                    checkpoint_path,
                    dataset.dataset_dir.resolve().name,
                    identity["split_manifest_hash"],
                    config.seed,
                    config.to_dict(),
                )
            write_json(run.run_dir / "history.json", history)
            print(
                f"  lr={learning_rate:g} | train loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%} | "
                f"val loss={validation_metrics['loss']:.4f}, acc={validation_metrics['accuracy']:.1%}",
                flush=True,
            )
            if on_epoch is not None:
                on_epoch(history, run.run_dir)
            if scheduler is not None:
                scheduler.step(validation_metrics["loss"])
            if selector.should_stop:
                break
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        clean_training = evaluate_classifier(
            model,
            DataLoader(prepared["train"], batch_size=config.batch_size),
            criterion,
            device,
            dataset.num_classes,
            progress,
        )
        checkpoint.update(
            early_stopping=selector.state(len(history)),
            data_identity=identity,
            notebook_model=model_spec,
            scheduler_parameters=dict(
                factor=settings.LR_FACTOR,
                patience=settings.LR_PATIENCE,
                min_lr=settings.MIN_LR,
            ),
        )
        torch.save(checkpoint, checkpoint_path)
        write_json(
            checkpoint_path.with_suffix(".json"),
            {
                k: v
                for k, v in checkpoint.items()
                if k not in {"model_state_dict", "optimizer_state_dict"}
            },
        )
        result = dict(
            name=label,
            partition="validation",
            run_dir=str(run.run_dir),
            selected_checkpoint=str(checkpoint_path),
            config=config.to_dict(),
            data_identity=identity,
            actual_epochs=len(history),
            validation_metrics=checkpoint["validation_metrics"],
            num_params=count_trainable_parameters(model),
            model_spec=model_spec,
            training_seconds=sum(row["seconds"] for row in history),
            clean_training_metrics=clean_training,
        )
        run.complete(
            {
                "checkpoint": str(checkpoint_path),
                "history": str(run.run_dir / "history.json"),
            },
            result,
        )
        return result
    except BaseException as error:
        run.update(
            {
                "status": (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        dataset.training_windows = False
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    """Resolve configuration and run validation-selected training."""
    args, experiment_config, print_only = _resolved_arguments(
        vars(parser.parse_args(argv))
    )
    if print_only:
        print(experiment_config.to_json())
        return
    deterministic_settings = seed_everything(args.seed)
    deterministic_settings["data_loader_seeds"] = {
        "train": args.seed,
        "validation": args.seed + 1,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    manifest_path = resolve_split_manifest_path(
        args.dataset_dir, args.split_manifest, args.split_seed
    )

    run = RunContext(
        args.runs_dir,
        purpose="train",
        dataset_path=args.dataset_dir,
        label="custom-convlstm",
        arguments=vars(args),
        metadata={
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
            "input_checkpoint": args.checkpoint_path,
        },
    )
    run_dir = run.run_dir
    print(f"📁 Run directory → {run_dir}")

    resolved_config_path = experiment_config.save_json(run_dir / "resolved_config.json")

    _probe = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    DatasetClass = CachedAHARDataset if len(_probe) <= 2000 else AHARDataset
    effective_dir = args.dataset_dir
    del _probe

    if DatasetClass is CachedAHARDataset:
        print("Small dataset - caching into RAM")

    dataset = DatasetClass(
        effective_dir, args.sequence_length, (args.width, args.height)
    )
    dataset_name = Path(args.dataset_dir).name
    num_classes = dataset.num_classes
    print(f"{len(dataset)} samples | {num_classes} classes")

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
        f"🎞️  Augmentation: {augmentation_state} | "
        f"Train samples: {len(train_set)} → {len(train_data)}"
    )

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
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
    layers = parse_layer_arguments(args.convlstm_layer)
    model = CustomConvLSTM(
        num_classes=num_classes,
        layers=layers,
        hidden_classifier_width=args.hidden_classifier_width,
    ).to(device)

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        print(f"⏳ Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        try:
            loaded_model = custom_model_from_checkpoint(checkpoint).to(device)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise SystemExit(f"Checkpoint is incompatible: {error}") from error

        if args.convlstm_layer is not None and loaded_model.layers != layers:
            raise SystemExit("Checkpoint layers do not match --convlstm-layer values")
        if (
            args.hidden_classifier_width is not None
            and loaded_model.hidden_classifier_width != args.hidden_classifier_width
        ):
            raise SystemExit(
                "Checkpoint hidden width does not match --hidden-classifier-width"
            )

        checkpoint_classes = loaded_model.num_classes
        if checkpoint_classes != num_classes:
            loaded_model.classifier = nn.Linear(
                loaded_model.classifier.in_features, num_classes
            ).to(device)
            loaded_model.num_classes = num_classes
            print(f"🔁 Output layer: {checkpoint_classes} → {num_classes} classes")

        for parameter in loaded_model.parameters():
            parameter.requires_grad = True
        if args.resume:
            print("▶️  Resuming - all layers trainable")
        else:
            print("🔓 Training all layers")
        if args.finetune_last:
            classifier_ids = {
                id(parameter) for parameter in loaded_model.classifier.parameters()
            }
            for parameter in loaded_model.parameters():
                parameter.requires_grad = id(parameter) in classifier_ids
            print("🔒 Fine-tuning the classifier only")
        model = loaded_model
        print(
            f"✅ Loaded epoch={checkpoint.get('epoch', 0)} | "
            f"params={count_trainable_parameters(model):,}"
        )
    else:
        print(
            "⚠️  No checkpoint - scratch | "
            f"params={count_trainable_parameters(model):,}"
        )

    resolved_configuration = {
        "experiment_config": experiment_config.to_dict(),
        "model": model.configuration(),
        "trainable_parameters": count_trainable_parameters(model),
        "class_names": dataset.class_names,
        "dataset_size": n_total,
        "split_sizes": {
            "train": n_train,
            "validation": n_val,
            "test": n_test,
        },
        "split": split_metadata,
        "deterministic_settings": deterministic_settings,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "Adam",
            "scheduler": args.scheduler,
            "metric_protocol": metric_protocol(),
            "checkpoint_selection": "lowest_validation_loss",
            "early_stopping_patience": args.early_stopping_patience,
            "test_access": "locked",
        },
    }
    config_path = write_json(run_dir / "config.json", resolved_configuration)
    run.update(
        {
            "class_names": dataset.class_names,
            "model_configuration": model.configuration(),
            "trainable_parameters": count_trainable_parameters(model),
            "split_sizes": resolved_configuration["split_sizes"],
            "split": split_metadata,
            "deterministic_settings": deterministic_settings,
        }
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=5
        )
        if args.scheduler == "reduce_on_plateau"
        else None
    )
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    epoch_history: list[dict[str, object]] = []
    selector = ValidationLossSelector(args.early_stopping_patience)
    history_path = run_dir / "metrics" / "history.json"
    best_path = run_dir / "checkpoints" / "best_model.pth"

    print("🚀 Training...")
    start = timer()

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n🧠 Epoch {epoch+1}/{args.epochs}  lr={current_lr:.2e}")

        train_metrics = train_classifier_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
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

        train_losses.append(train_metrics["loss"])
        val_losses.append(validation_metrics["loss"])
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(validation_metrics["accuracy"])
        selected = selector.update(validation_metrics["loss"], epoch + 1)
        epoch_history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "training_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "selected_checkpoint": selected,
            }
        )
        write_json(
            history_path,
            {
                "metric_protocol": metric_protocol(),
                "selection": selector.state(len(epoch_history)),
                "epochs": epoch_history,
            },
        )
        print(
            "  Loss → "
            f"Train: {train_metrics['loss']:.4f} "
            f"Val: {validation_metrics['loss']:.4f} | Acc → "
            f"Train: {train_metrics['accuracy']:.4f} "
            f"Val: {validation_metrics['accuracy']:.4f} | "
            f"Val macro-F1: {validation_metrics['macro_f1']:.4f}"
        )

        if selected:
            _save_custom_checkpoint(
                model,
                optimizer,
                epoch + 1,
                validation_metrics,
                checkpoint_path=best_path,
                dataset_name=dataset_name,
                split_manifest_hash=str(split_metadata["manifest_hash"]),
                seed=args.seed,
                experiment_config=experiment_config.to_dict(),
            )
            print(
                "  ⭐ Best model updated "
                f"(val_loss={validation_metrics['loss']:.4f})"
            )
        if selector.should_stop:
            print(
                "  🛑 Early stopping: "
                f"{args.early_stopping_patience} epochs without improvement"
            )
            break

    print(f"\n⏱  Done in {timer() - start:.1f}s")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_path=run_dir / "plots" / "training_curves.png",
        show=False,
    )

    selection_state = selector.state(len(epoch_history))
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    checkpoint["early_stopping"] = selection_state
    torch.save(checkpoint, best_path)
    write_json(
        best_path.with_suffix(".json"),
        {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state_dict", "optimizer_state_dict"}
        },
    )
    checkpoint_selection = validate_selected_checkpoint(
        checkpoint,
        dataset_name,
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
    true_labels, predicted_labels = collect_predictions(model, val_loader, device)
    confusion_artifacts = plot_confusion_matrix(
        true_labels,
        predicted_labels,
        dataset.class_names,
        dataset_name=f"{dataset_name}_validation",
        save_path=run_dir / "metrics" / "validation_confusion_matrix.png",
    )
    final_metrics = {
        "evaluated_model": "validation_selected_checkpoint",
        "partition": "validation",
        "metric_protocol": metric_protocol(),
        "validation_metrics": selected_validation_metrics,
        "checkpoint_selection": checkpoint_selection,
        "early_stopping": selection_state,
        "test_access": "locked",
    }
    final_metrics_path = write_json(run_dir / "metrics" / "final.json", final_metrics)
    resolved_configuration["checkpoint_selection"] = checkpoint_selection
    resolved_configuration["early_stopping"] = selection_state
    write_json(config_path, resolved_configuration)
    run.update(
        {
            "metric_protocol": metric_protocol(),
            "checkpoint_selection": checkpoint_selection,
            "early_stopping": selection_state,
            "test_access": "locked",
        }
    )
    run.complete(
        artifacts={
            "resolved_configuration": str(resolved_config_path),
            "configuration": str(config_path),
            "history": str(history_path),
            "best_checkpoint": str(run_dir / "checkpoints" / "best_model.pth"),
            "best_checkpoint_metadata": str(
                run_dir / "checkpoints" / "best_model.json"
            ),
            "training_curves": str(run_dir / "plots" / "training_curves.png"),
            "final_metrics": str(final_metrics_path),
            "validation_confusion_matrix": confusion_artifacts,
        },
        results=final_metrics,
    )
    print(f"📁 Run artifacts → {run_dir}")


if __name__ == "__main__":
    main()
