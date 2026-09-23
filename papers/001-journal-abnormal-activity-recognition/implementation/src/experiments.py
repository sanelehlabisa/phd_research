"""
experiments.py

Exploratory comparison runner for ConvLSTM and video-model baselines.
Saves all results to JSON.

Author: Sanele Hlabisa

python -m src.experiments \
    --dataset_dir "datasets/processed/frames_abnormal_activities" \
    --epochs 24 \
    --batch_size 16 \
    --sequence_length 16 \
    --height 32 \
    --width 32 \
    --aug_copies 1 \
    --num_workers 2
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
import torchvision.models.video as video_models

from .dataset import AHARDataset, AugmentSubset
from .model import CustomConvLSTM, PaperConvLSTM, count_trainable_parameters
from .utils import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--results_dir", type=str, default="experiments/grid_search")
parser.add_argument("--epochs", type=int, default=24)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--sequence_length", type=int, default=16)
parser.add_argument("--height", type=int, default=32)
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--aug_copies", type=int, default=1)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument(
    "--list-models",
    action="store_true",
    help="List approved baseline roles without loading data or starting training",
)


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


def model_registry() -> list[dict[str, str]]:
    """Describe the approved comparison models without allocating them.

    Parameters:
        None.

    Returns:
        Model names, families, classes, and comparison roles.
    """
    return [
        {
            "name": "paper_convlstm_published",
            "family": "ConvLSTM",
            "model_class": "PaperConvLSTM",
            "role": "source-paper ConvLSTM topology baseline",
        },
        {
            "name": "custom_convlstm_reference_8_k3",
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
) -> nn.Module:
    """Build one approved model by its registry name.

    Parameters:
        model_name: Exact name returned by `model_registry`.
        num_classes: Number of dataset classes.
        input_shape: Frame shape `(channels, height, width)`.
        sequence_length: Frames supplied to the model.

    Returns:
        The requested untrained model.
    """
    if model_name == "paper_convlstm_published":
        return PaperConvLSTM(
            num_classes,
            input_shape=input_shape,
            sequence_length=sequence_length,
        )
    if model_name == "custom_convlstm_reference_8_k3":
        return CustomConvLSTM(num_classes, layers=[(8, (3, 3))])
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


def print_model_registry() -> None:
    """Print approved model names and roles without allocating models.

    Parameters:
        None.

    Returns:
        None.
    """
    print("Approved comparison models")
    for entry in model_registry():
        print(
            f"- {entry['name']}: {entry['model_class']} | "
            f"{entry['family']} | {entry['role']}"
        )


def _train(model, loader, criterion, optimizer, acc_fn, device):
    """
    Runs a single training epoch and calculates the average loss and accuracy.

    Parameters:
        model (torch.nn.Module): The neural network model being trained.
        loader (DataLoader): The data loader providing batches of training data.
        criterion (torch.nn.Module): The loss function used to calculate the error.
        optimizer (torch.optim.Optimizer): The optimizer updating the model weights.
        acc_fn (torchmetrics.Metric): The function used to calculate accuracy.
        device (torch.device): The hardware device (CPU or GPU) running the calculations.

    Returns:
        metrics (tuple): A tuple containing the average loss and average accuracy for the epoch.
    """
    model.train()
    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


@torch.inference_mode()
def _validate(model, loader, criterion, acc_fn, device):
    """
    Evaluates the model on a validation or test dataset without updating weights.

    Parameters:
        model (torch.nn.Module): The neural network model being evaluated.
        loader (DataLoader): The data loader providing batches of evaluation data.
        criterion (torch.nn.Module): The loss function used to calculate the error.
        acc_fn (torchmetrics.Metric): The function used to calculate accuracy.
        device (torch.device): The hardware device (CPU or GPU) running the calculations.

    Returns:
        metrics (tuple): A tuple containing the average loss and average accuracy.
    """
    model.eval()
    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += acc_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


def _overfit_score(train_accs: list[float], val_accs: list[float]) -> float:
    """
    Calculates an overfitting score by analyzing the gap between training and validation accuracy.
    """
    gaps = [t - v for t, v in zip(train_accs, val_accs)]
    gradients = [gaps[i + 1] - gaps[i] for i in range(len(gaps) - 1)]
    return sum(gradients) / len(gradients) if gradients else (gaps[-1] if gaps else 0.0)


def main() -> None:
    args = parser.parse_args()
    registry = model_registry()
    if args.list_models:
        print_model_registry()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    num_classes = dataset.num_classes
    print(
        f"Loaded {len(dataset)} samples | {num_classes} classes: {dataset.class_names}"
    )

    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # ---- Augmentation (same pipeline as train.py) ----
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

    base_subset = torch.utils.data.Subset(dataset, train_set.indices)
    aug_subsets = [
        AugmentSubset(base_subset, train_transform) for _ in range(args.aug_copies)
    ]
    combined = torch.utils.data.ConcatDataset([base_subset] + aug_subsets)

    loader_kw = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )
    train_loader = DataLoader(combined, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)

    # ---- Model configs ----
    input_shape = (3, args.height, args.width)
    print(f"\nRunning {len(registry)} configurations...\n")
    all_results = []

    # Extra metrics tracker for final evaluation on test_set
    test_metrics = {
        "accuracy": torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(device),
        "precision": torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
        "recall": torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
        "f1": torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
    }

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
            entry["name"],
            num_classes,
            input_shape,
            args.sequence_length,
        )
        model = model.to(device)
        num_params = count_trainable_parameters(model)
        print(f"[{i+1}/{len(registry)}] {name} | params={num_params:,}")

        opt = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
            device
        )

        train_accs, val_accs, val_losses = [], [], []
        t0 = timer()

        for epoch in tqdm(range(args.epochs), leave=False, desc=name):
            _, tr_acc = _train(model, train_loader, criterion, opt, acc_fn, device)
            vl_loss, vl_acc = _validate(
                model, val_loader, criterion, acc_fn.clone(), device
            )
            train_accs.append(tr_acc)
            val_accs.append(vl_acc)
            val_losses.append(vl_loss)

        elapsed = timer() - t0
        best_val_loss = min(val_losses)
        best_val_acc = max(val_accs)
        overfit = _overfit_score(train_accs, val_accs)

        # Full test evaluation loop (collecting extra metrics and labels for the confusion matrix)
        all_true, all_pred = [], []
        model.eval()
        for m in test_metrics.values():
            m.reset()

        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(X)
                preds = logits.argmax(dim=1)
                for m in test_metrics.values():
                    m(preds, y)
                all_pred.extend(preds.cpu().tolist())
                all_true.extend(y.cpu().tolist())

        t_res = {k: m.compute().item() for k, m in test_metrics.items()}

        # Generates confusion matrix per architecture variant!
        dataset_name_clean = args.dataset_dir.strip("/").split("/")[-1]
        cm_path = str(results_dir / f"cm_{dataset_name_clean}_{name}.png")
        plot_confusion_matrix(
            all_true,
            all_pred,
            dataset.class_names,
            dataset_name=name,
            save_path=cm_path,
        )

        result = {
            "name": name,
            "family": entry["family"],
            "model_class": entry["model_class"],
            "role": role,
            "num_params": num_params,
            "best_val_loss": round(best_val_loss, 6),
            "best_val_acc": round(best_val_acc, 4),
            "test_acc": round(t_res["accuracy"], 4),
            "test_precision": round(t_res["precision"], 4),
            "test_recall": round(t_res["recall"], 4),
            "test_f1": round(t_res["f1"], 4),
            "overfit_score": round(overfit, 4),
            "train_time_s": round(elapsed, 1),
            "cm_path": cm_path,
        }
        all_results.append(result)
        print(
            f"  val_acc={best_val_acc:.4f}  test_acc={t_res['accuracy']:.4f}  "
            f"overfit={overfit:+.4f}  time={elapsed:.0f}s"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Rank and save ----
    stable = [r for r in all_results if r["overfit_score"] < 0.05]
    ranked = sorted(
        stable if stable else all_results,
        key=lambda r: (-r["best_val_acc"], r["num_params"]),
    )

    print(f"\nTop 5 (stable, best val acc, fewest params):")
    print("-" * 70)
    for r in ranked[:5]:
        print(
            f"  {r['name']:<30} val_acc={r['best_val_acc']:.4f}  "
            f"test_acc={r['test_acc']:.4f}  overfit={r['overfit_score']:+.4f}  "
            f"params={r['num_params']:,}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_clean = args.dataset_dir.strip("/").split("/")[-1]
    out_path = results_dir / f"grid_search_{dataset_name_clean}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"best": ranked[:5], "all": all_results}, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
