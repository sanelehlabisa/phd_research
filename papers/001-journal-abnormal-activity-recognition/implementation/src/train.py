"""
train.py

Training script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR).

Author: Sanele Hlabisa

python -m src.train \
    --dataset_dir "datasets/processed/videos_violence-detection-dataset" \
    --model_dir "models" \
    --checkpoint_path "models/videos_violence-detection-dataset_best_model.pth" \
    --convlstm-layer 8 3 3 \
    --convlstm-layer 16 3 3 \
    --resume \
    --finetune_full \
    --batch_size 32 \
    --weight_decay 0.0001 \
    --learning_rate 0.001 \
    --epochs 64 \
    --sequence_length 64 \
    --height 64 \
    --width 64 \
    --aug_copies 4 \
    --num_workers 2 \
    --pin_memory
    
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from .dataset import AHARDataset, CachedAHARDataset, AugmentSubset
from .model import (
    CustomConvLSTM,
    count_trainable_parameters,
    custom_model_from_checkpoint,
    parse_layer_arguments,
)
from .utils import plot_training_curves, save_prediction_clips

parser = argparse.ArgumentParser(description="Train ConvLSTM for AHAR")
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training same dataset, no layer changes",
)
parser.add_argument(
    "--finetune_last",
    action="store_true",
    help="Freeze all except the classifier",
)
parser.add_argument(
    "--finetune_full",
    action="store_true",
    help="Load weights, unfreeze everything, train all layers",
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
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
parser.add_argument("--aug_copies", type=int, default=4)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", action="store_true")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:
    """
    Trains the model for a single epoch and returns the average loss and accuracy.

    Parameters:
        model (nn.Module): The neural network model being trained.
        loader (DataLoader): The DataLoader providing batches of training data.
        criterion (nn.Module): The loss function used for optimization.
        optimizer (optim.Optimizer): The optimizer updating the model weights.
        accuracy_fn (torchmetrics.Metric): The function used to calculate accuracy.
        device (torch.device): The hardware device running the calculations.

    Returns:
        metrics (tuple[float, float]): The average loss and average accuracy for the epoch.
    """
    model.train()

    total_loss = total_acc = 0.0
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits: torch.Tensor = model(X)
        loss: torch.Tensor = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluates the model on validation or test data for a single epoch.

    Parameters:
        model (nn.Module): The neural network model being evaluated.
        loader (DataLoader): The DataLoader providing batches of evaluation data.
        criterion (nn.Module): The loss function used to calculate the error.
        accuracy_fn (torchmetrics.Metric): The function used to calculate accuracy.
        device (torch.device): The hardware device running the calculations.

    Returns:
        metrics (tuple[float, float]): The average loss and average accuracy for the epoch.
    """
    model.eval()

    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits: torch.Tensor = model(X)
        loss: torch.Tensor = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


def _save_custom_checkpoint(
    model: CustomConvLSTM,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: Path,
) -> None:
    """Save weights with enough architecture metadata to rebuild the model.

    Parameters:
        model: Custom model being trained.
        optimizer: Optimizer whose state should be saved.
        epoch: Current training epoch.
        loss: Current validation loss.
        checkpoint_path: Destination checkpoint path.

    Returns:
        None.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = model.configuration()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": configuration,
            "epoch": epoch,
            "loss": loss,
        },
        checkpoint_path,
    )
    metadata = {
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "trainable_parameters": count_trainable_parameters(model),
        "model_config": configuration,
    }
    with checkpoint_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"✅ Saved checkpoint to {checkpoint_path}")


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"🖥  Using device: {device}")

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

    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # Streamlined pipeline for linear probing/head fine-tuning
    # Upgraded robust transform pipeline with TrivialAugmentWide compatibility
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    )
                ],
                p=0.5,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        size=(args.height, args.width), scale=(0.8, 1.0)
                    )
                ],
                p=0.4,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(
                [transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.RandomErasing(
                p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0
            ),
        ]
    )

    train_indices = train_set.indices
    clean_subset = torch.utils.data.Subset(dataset, train_indices)
    aug_subsets = [
        AugmentSubset(clean_subset, train_transform) for _ in range(args.aug_copies)
    ]
    combined_train = torch.utils.data.ConcatDataset([clean_subset] + aug_subsets)
    print(f"📈 Train expanded: {len(train_indices)} → {len(combined_train)} samples")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(combined_train, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5
    )
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
        device
    )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    print("🚀 Training...")
    start = timer()

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n🧠 Epoch {epoch+1}/{args.epochs}  lr={current_lr:.2e}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, acc_fn, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, acc_fn.clone(), device
        )
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(
            f"  Loss → Train: {train_loss:.4f} Val: {val_loss:.4f} | Acc → Train: {train_acc:.4f} Val: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = (
                Path(args.model_dir) / f"{Path(args.dataset_dir).name}_best_model.pth"
            )
            _save_custom_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                checkpoint_path=best_path,
            )
            print(f"  ⭐ Best model updated (val_loss={val_loss:.4f})")

    print(f"\n⏱  Done in {timer() - start:.1f}s")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        dataset_name=dataset_name,
        save_dir=args.model_dir,
        show=False,
    )

    test_loss, test_acc = validate_one_epoch(
        model, test_loader, criterion, acc_fn.clone(), device
    )
    print(f"\n🏁 Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    out_dir = Path("outputs") / "train_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_clips(
        model, test_set, dataset.class_names, device, out_dir, num_samples=8
    )


if __name__ == "__main__":
    main()
