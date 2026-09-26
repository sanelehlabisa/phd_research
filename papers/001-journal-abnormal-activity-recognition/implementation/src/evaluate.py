"""
evaluate.py

Evaluation script for AHAR.

Author: Sanele Hlabisa

.venv/bin/python -m src.evaluate \
    --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
    --checkpoint_path "runs/train/<run>/checkpoints/best_model.pth" \
    --convlstm-layer 8 3 3 \
    --convlstm-layer 16 3 3 \
    --runs_dir "runs" \
    --split_manifest "splits/abnormal-activities-dataset_seed42.json" \
    --seed 42 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --batch_size 32 \
    --sequence_length 16 \
    --height 32 \
    --width 32 \
    --num_workers 2 \
    --pin_memory \
    --num_samples 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    AHARDataset,
    DEFAULT_SPLIT_SEED,
    load_split_subsets,
    resolve_split_manifest_path,
)
from .metrics import evaluate_classifier, metric_protocol, validate_selected_checkpoint
from .model import custom_model_from_checkpoint, parse_layer_arguments
from .utils import (
    RunContext,
    collect_predictions,
    data_loader_generator,
    plot_confusion_matrix,
    save_prediction_clips,
    seed_data_loader_worker,
    seed_everything,
    write_json,
)

parser = argparse.ArgumentParser(description="Evaluate ConvLSTM for AHAR")
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--runs_dir", type=str, default="runs")
parser.add_argument("--split_manifest", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument(
    "--convlstm-layer",
    action="append",
    nargs=3,
    type=int,
    metavar=("FILTERS", "KERNEL_HEIGHT", "KERNEL_WIDTH"),
    help="Repeat for each CustomConvLSTM layer, for example: 8 3 3",
)
parser.add_argument("--hidden-classifier-width", type=int, default=None)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.15)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", action="store_true")
parser.add_argument("--num_samples", type=int, default=8)


def main() -> None:
    args = parser.parse_args()
    deterministic_settings = seed_everything(args.seed)
    deterministic_settings["data_loader_seeds"] = {"test": args.seed + 2}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    dataset_name = Path(args.dataset_dir).name
    manifest_path = resolve_split_manifest_path(
        args.dataset_dir, args.split_manifest, DEFAULT_SPLIT_SEED
    )
    run = RunContext(
        args.runs_dir,
        purpose="evaluate",
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
            "augmentation": False,
            "split_ratios": {
                "train": args.train_ratio,
                "validation": args.val_ratio,
                "test": 1.0 - args.train_ratio - args.val_ratio,
            },
            "seed": args.seed,
            "split_seed": DEFAULT_SPLIT_SEED,
            "split_manifest": str(manifest_path.resolve()),
            "deterministic_settings": deterministic_settings,
            "device": str(device),
            "input_checkpoint": args.checkpoint_path,
        },
    )
    run_dir = run.run_dir
    print(f"📁 Run directory → {run_dir}")

    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    num_classes = dataset.num_classes
    print(f"📦 {len(dataset)} samples | {num_classes} classes: {dataset.class_names}")

    train_set, val_set, test_set, split_metadata = load_split_subsets(
        dataset,
        manifest_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=DEFAULT_SPLIT_SEED,
    )
    n_total = len(dataset)
    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)
    print(f"📊 Test split: {n_test} samples")

    layers = parse_layer_arguments(args.convlstm_layer)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        checkpoint_selection = validate_selected_checkpoint(
            checkpoint,
            dataset_name,
            str(split_metadata["manifest_hash"]),
        )
        model = custom_model_from_checkpoint(checkpoint).to(device)
        if args.convlstm_layer is not None and model.layers != layers:
            raise ValueError("checkpoint layers do not match --convlstm-layer values")
        if (
            args.hidden_classifier_width is not None
            and model.hidden_classifier_width != args.hidden_classifier_width
        ):
            raise ValueError(
                "checkpoint hidden width does not match --hidden-classifier-width"
            )
        if model.num_classes != num_classes:
            raise ValueError(
                "checkpoint class count does not match the dataset class count"
            )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"Checkpoint is incompatible: {error}") from error
    print(
        "📂 Validation-selected checkpoint → "
        f"epoch={checkpoint_selection['selected_epoch']}, "
        f"loss={checkpoint_selection['selection_value']:.4f}"
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=seed_data_loader_worker,
        generator=data_loader_generator(args.seed + 2),
    )

    configuration = {
        "model": model.configuration(),
        "class_names": dataset.class_names,
        "dataset_size": n_total,
        "split_sizes": {"train": n_train, "validation": n_val, "test": n_test},
        "split": split_metadata,
        "deterministic_settings": deterministic_settings,
        "metric_protocol": metric_protocol(),
        "evaluation_partition": "test",
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
            "model_configuration": model.configuration(),
            "split_sizes": configuration["split_sizes"],
            "split": split_metadata,
            "deterministic_settings": deterministic_settings,
            "metric_protocol": metric_protocol(),
            "evaluation_partition": "test",
            "checkpoint_selection": checkpoint_selection,
        }
    )

    criterion = nn.CrossEntropyLoss()
    results = evaluate_classifier(
        model,
        test_loader,
        criterion,
        device,
        num_classes,
    )
    print("\n🏁 Final test results")
    print("─" * 32)
    for name, value in results.items():
        print(f"  {name:<12}: {value:.4f}")

    all_true, all_pred = collect_predictions(model, test_loader, device)
    confusion_artifacts = plot_confusion_matrix(
        all_true,
        all_pred,
        dataset.class_names,
        dataset_name=f"{dataset_name}_test",
        save_path=run_dir / "metrics" / "test_confusion_matrix.png",
    )

    print("\n🎬 Saving prediction clips...")
    clip_records = save_prediction_clips(
        model,
        test_set,
        dataset.class_names,
        device,
        run_dir / "predictions",
        args.num_samples,
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
        "artifacts": {
            "confusion_matrix": confusion_artifacts,
            "prediction_clips": clip_records,
        },
    }
    json_path = write_json(run_dir / "metrics" / "final.json", report)
    run.complete(
        artifacts={
            "configuration": str(config_path),
            "checkpoint_reference": str(checkpoint_reference_path),
            "metrics": str(json_path),
            "confusion_matrix": confusion_artifacts,
            "prediction_videos": clip_records,
        },
        results=report["metrics"],
    )
    print(f"📄 Report → {json_path}")


if __name__ == "__main__":
    main()
