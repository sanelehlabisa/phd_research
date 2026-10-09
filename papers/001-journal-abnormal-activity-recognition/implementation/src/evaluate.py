"""
evaluate.py

Evaluation script for AHAR.

Author: Sanele Hlabisa

.venv/bin/python -m src.evaluate \
    --config configs/evaluate/aad_evaluation_reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from .dataset import (
    AHARDataset,
    load_split_subsets,
    resolve_split_manifest_path,
)
from .experiment_config import ExperimentConfig
from .dataset_source import resolve_dataset
from .metrics import evaluate_classifier, metric_protocol, validate_selected_checkpoint
from .model import custom_model_from_checkpoint
from .utils import (
    RunContext,
    data_loader_generator,
    plot_confusion_matrix,
    save_prediction_examples,
    seed_data_loader_worker,
    seed_everything,
    write_json,
)

parser = argparse.ArgumentParser(
    description="Evaluate the best checkpoint from a training run for AHAR",
)
parser.add_argument("--config", type=Path, required=True)


def model_from_checkpoint(checkpoint):
    """Restore a custom or registered practical baseline without pretrained weights."""
    description = checkpoint.get("model_config", {})
    registry = description.get("registry_name")
    if registry is None:
        return custom_model_from_checkpoint(checkpoint)
    if registry not in {
        "r3d_18",
        "mc3_18",
        "r2plus1d_18",
        "swin3d_t",
        "swin3d_s",
        "paper_convlstm_published",
    }:
        raise ValueError(f"unsupported final model registry: {registry}")
    from .experiments import build_registered_model

    dimensions = description["input_dimensions"]
    with torch.device("meta"):
        model = build_registered_model(
            registry,
            description["num_classes"],
            (3, dimensions["height"], dimensions["width"]),
            dimensions["sequence_length"],
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
    return model


@torch.inference_mode()
def prediction_records(model, loader, device, class_names, partition="test"):
    """Keep full test predictions and sample losses for independent recalculation."""
    records = []
    subset = loader.dataset
    model.eval()
    for inputs, labels in loader:
        logits = model(inputs.to(device))
        losses = nn.functional.cross_entropy(
            logits, labels.to(device), reduction="none"
        )
        probabilities = logits.softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        for target, prediction, score, loss in zip(
            labels.tolist(), predicted.tolist(), confidence.tolist(), losses.tolist()
        ):
            source = subset.dataset.samples[subset.indices[len(records)]][0]
            records.append(
                {
                    "source": str(source),
                    "partition": partition,
                    "target": target,
                    "predicted": prediction,
                    "true_class": class_names[target],
                    "predicted_class": class_names[prediction],
                    "confidence": score,
                    "correct": target == prediction,
                    "loss": loss,
                }
            )
    return records


def per_class_metrics(records, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        [r["target"] for r in records],
        [r["predicted"] for r in records],
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return [
        {
            "class": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    ]


def main(argv=None) -> Path:
    args = parser.parse_args(argv)
    config_file = args.config
    try:
        values = json.loads(config_file.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"Evaluation config not found: {config_file}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {config_file}: {error.msg}") from error
    if not isinstance(values, dict):
        raise SystemExit("Evaluation config must contain a JSON object")

    training_run_value = values.pop("training_run_dir", None)
    checkpoint_value = values.pop("checkpoint_path", None)
    if (training_run_value is None) == (checkpoint_value is None):
        raise SystemExit("Set exactly one of training_run_dir or checkpoint_path")
    selected_path = training_run_value if checkpoint_value is None else checkpoint_value
    if not isinstance(selected_path, str) or not selected_path.strip():
        raise SystemExit("Checkpoint/training path must be non-empty")
    training_run_dir = (
        Path(selected_path).expanduser() if checkpoint_value is None else None
    )
    checkpoint_path = (
        training_run_dir / "checkpoints" / "best_model.pth"
        if training_run_dir is not None
        else Path(checkpoint_value).expanduser()
    )
    try:
        config = ExperimentConfig.from_mapping(values, defaults=ExperimentConfig())
    except ValueError as error:
        raise SystemExit(f"Invalid evaluation config: {error}") from error
    resolved_dataset = resolve_dataset(
        config.dataset_name,
        config.dataset_dir,
        Path(__file__).resolve().parent.parent,
    )
    config = ExperimentConfig.from_mapping(
        {**config.to_dict(), "dataset_dir": str(resolved_dataset)}
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"best checkpoint not found in training run: {checkpoint_path}"
        )

    deterministic_settings = seed_everything(config.seed)
    deterministic_settings["data_loader_seeds"] = {"test": config.seed + 2}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    dataset_name = config.dataset_name
    manifest_path = resolve_split_manifest_path(
        config.dataset_dir, config.split_manifest, config.split_seed
    )
    run = RunContext(
        config.runs_dir,
        purpose="evaluate",
        dataset_path=config.dataset_dir,
        label="selected-model",
        arguments={
            "config_file": str(config_file) if config_file is not None else None,
            "experiment_config": config.to_dict(),
            "training_run_dir": str(training_run_dir),
            "checkpoint_path": str(checkpoint_path),
        },
        metadata={
            "input_dimensions": {
                "sequence_length": config.sequence_length,
                "channels": 3,
                "height": config.height,
                "width": config.width,
            },
            "augmentation": False,
            "split_ratios": {
                "train": config.train_ratio,
                "validation": config.val_ratio,
                "test": config.test_ratio,
            },
            "seed": config.seed,
            "split_seed": config.split_seed,
            "split_manifest": str(manifest_path.resolve()),
            "deterministic_settings": deterministic_settings,
            "device": str(device),
            "input_checkpoint": str(checkpoint_path),
            "training_run_dir": str(training_run_dir),
        },
    )
    run_dir = run.run_dir
    print(f"📁 Run directory → {run_dir}")

    dataset = AHARDataset(
        config.dataset_dir,
        config.sequence_length,
        (config.width, config.height),
    )
    num_classes = dataset.num_classes
    print(f"📦 {len(dataset)} samples | {num_classes} classes: {dataset.class_names}")

    train_set, val_set, test_set, split_metadata = load_split_subsets(
        dataset,
        manifest_path,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.split_seed,
    )
    n_total = len(dataset)
    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)
    print(f"📊 Test split: {n_test} samples")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        checkpoint_selection = validate_selected_checkpoint(
            checkpoint,
            dataset_name,
            str(split_metadata["manifest_hash"]),
        )
        model = model_from_checkpoint(checkpoint).to(device)
        if checkpoint["model_config"]["num_classes"] != num_classes:
            raise ValueError(
                "checkpoint class count does not match the dataset class count"
            )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"Checkpoint is incompatible: {error}") from error
    checkpoint_config = checkpoint.get("experiment_config")
    if isinstance(checkpoint_config, dict):
        expected_input = {
            "sequence_length": checkpoint_config.get("sequence_length"),
            "height": checkpoint_config.get("height"),
            "width": checkpoint_config.get("width"),
        }
        actual_input = {
            "sequence_length": config.sequence_length,
            "height": config.height,
            "width": config.width,
        }
        if all(value is not None for value in expected_input.values()) and (
            expected_input != actual_input
        ):
            raise SystemExit(
                "Evaluation input dimensions do not match the checkpoint config: "
                f"checkpoint={expected_input}, evaluation={actual_input}. Use the "
                "training JSON config or explicitly override its dimensions."
            )
    prediction_samples = config.prediction_samples_per_category
    print(
        "📂 Validation-selected checkpoint → "
        f"epoch={checkpoint_selection['selected_epoch']}, "
        f"loss={checkpoint_selection['selection_value']:.4f}"
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_data_loader_worker,
        generator=data_loader_generator(config.seed + 2),
    )

    configuration = {
        "model": checkpoint["model_config"],
        "class_names": dataset.class_names,
        "dataset_size": n_total,
        "split_sizes": {"train": n_train, "validation": n_val, "test": n_test},
        "split": split_metadata,
        "deterministic_settings": deterministic_settings,
        "metric_protocol": metric_protocol(),
        "evaluation_partition": "test",
        "prediction_samples_per_category": prediction_samples,
        "experiment_config": config.to_dict(),
        "config_file": str(config_file) if config_file is not None else None,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            **checkpoint_selection,
        },
    }
    config_path = write_json(run_dir / "config.json", configuration)
    checkpoint_reference_path = write_json(
        run_dir / "checkpoint.json", configuration["checkpoint"]
    )
    run.update(
        {
            "class_names": dataset.class_names,
            "model_configuration": checkpoint["model_config"],
            "split_sizes": configuration["split_sizes"],
            "split": split_metadata,
            "deterministic_settings": deterministic_settings,
            "metric_protocol": metric_protocol(),
            "evaluation_partition": "test",
            "checkpoint_selection": checkpoint_selection,
        }
    )

    criterion = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    results = evaluate_classifier(
        model,
        test_loader,
        criterion,
        device,
        num_classes,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    metric_pass_seconds = perf_counter() - started
    print("\n🏁 Final test results")
    print("─" * 32)
    for name, value in results.items():
        print(f"  {name:<12}: {value:.4f}")

    records = prediction_records(model, test_loader, device, dataset.class_names)
    records_path = write_json(run_dir / "metrics" / "test_predictions.json", records)
    all_true = [record["target"] for record in records]
    all_pred = [record["predicted"] for record in records]
    from .study_reporting import extended_metrics

    results.update(extended_metrics(all_pred, all_true, num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_pred, labels=list(range(num_classes)), zero_division=0
    )
    confusion_artifacts = plot_confusion_matrix(
        all_true,
        all_pred,
        dataset.class_names,
        dataset_name=f"{dataset_name}_test",
        save_path=run_dir / "metrics" / "test_confusion_matrix.png",
    )

    print("\n🎬 Saving prediction clips...")
    prediction_examples = save_prediction_examples(
        model,
        test_set,
        dataset.class_names,
        device,
        run_dir / "predictions",
        prediction_samples,
        partition="test",
        seed=config.seed,
    )

    report = {
        "dataset": dataset_name,
        "dataset_mode": "classification",
        "partition": "test",
        "metric_protocol": metric_protocol(),
        "classes": dataset.class_names,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            **checkpoint_selection,
        },
        "metrics": {k: round(v, 6) for k, v in results.items()},
        "per_class": [
            {
                "class": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(dataset.class_names)
        ],
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "metric_pass_seconds": metric_pass_seconds,
        "runtime_note": "Full metric pass including decoding/data loading; not pure model latency.",
        "experiment_config": config.to_dict(),
        "split": split_metadata,
        "predictions": str(records_path),
        "prediction_samples_per_category": prediction_samples,
        "prediction_examples": prediction_examples,
        "artifacts": {
            "confusion_matrix": confusion_artifacts,
            "prediction_clips": prediction_examples["records"],
        },
    }
    json_path = write_json(run_dir / "metrics" / "final.json", report)
    run.complete(
        artifacts={
            "configuration": str(config_path),
            "checkpoint_reference": str(checkpoint_reference_path),
            "metrics": str(json_path),
            "confusion_matrix": confusion_artifacts,
            "prediction_examples": prediction_examples,
            "predictions": str(records_path),
        },
        results=report["metrics"],
    )
    print(f"📄 Report → {json_path}")
    return run_dir


if __name__ == "__main__":
    main()
