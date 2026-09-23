"""Shared classification metrics and validation-selection helpers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader


def metric_protocol() -> dict[str, str]:
    """Describe the loss and classification metric aggregation rules.

    Parameters:
        None.

    Returns:
        JSON-friendly aggregation and class-averaging settings.
    """
    return {
        "loss_aggregation": "sample_weighted_mean",
        "classification_aggregation": "full_partition",
        "precision_recall_f1_average": "macro",
    }


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute full-partition classification metrics with fresh state.

    Parameters:
        predictions: Predicted class indices.
        targets: Ground-truth class indices.
        num_classes: Number of classes in the dataset.

    Returns:
        Accuracy and macro precision, recall, and F1.
    """
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("predictions and targets must be one-dimensional")
    if predictions.numel() == 0 or predictions.numel() != targets.numel():
        raise ValueError("predictions and targets must have the same non-zero length")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")

    metric_objects = {
        "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "macro_precision": torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ),
        "macro_recall": torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ),
        "macro_f1": torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ),
    }
    return {
        name: float(metric(predictions, targets).item())
        for name, metric in metric_objects.items()
    }


def _complete_partition_metrics(
    total_loss: float,
    total_samples: int,
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    num_classes: int,
) -> dict[str, float]:
    """Complete sample-weighted loss and full-partition metrics.

    Parameters:
        total_loss: Sum of each batch mean loss multiplied by its batch size.
        total_samples: Number of samples seen.
        predictions: Per-batch predicted class indices.
        targets: Per-batch ground-truth class indices.
        num_classes: Number of dataset classes.

    Returns:
        Loss, accuracy, and macro precision, recall, and F1.
    """
    if total_samples <= 0:
        raise ValueError("cannot calculate metrics for an empty partition")
    results = classification_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        num_classes,
    )
    return {"loss": total_loss / total_samples, **results}


def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """Train one epoch and return sample-weighted partition metrics.

    Parameters:
        model: Classification model to train.
        loader: Training data loader.
        criterion: Per-batch mean loss function.
        optimizer: Optimizer used to update the model.
        device: Device used for the forward and backward passes.
        num_classes: Number of dataset classes.

    Returns:
        Training loss and full-partition classification metrics.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        predictions.append(logits.detach().argmax(dim=1).cpu())
        targets.append(labels.detach().cpu())
    return _complete_partition_metrics(
        total_loss,
        total_samples,
        predictions,
        targets,
        num_classes,
    )


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """Evaluate one complete partition with fresh metric state.

    Parameters:
        model: Classification model to evaluate.
        loader: Validation or test data loader.
        criterion: Per-batch mean loss function.
        device: Device used for inference.
        num_classes: Number of dataset classes.

    Returns:
        Sample-weighted loss and full-partition classification metrics.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    return _complete_partition_metrics(
        total_loss,
        total_samples,
        predictions,
        targets,
        num_classes,
    )


class ValidationLossSelector:
    """Track the lowest validation loss and early-stopping state."""

    def __init__(self, patience: int = 10) -> None:
        """Initialize validation-loss selection.

        Parameters:
            patience: Consecutive non-improving epochs allowed before stopping.

        Returns:
            None.
        """
        if not isinstance(patience, int) or isinstance(patience, bool) or patience <= 0:
            raise ValueError("patience must be a positive integer")
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def update(self, validation_loss: float, epoch: int) -> bool:
        """Update selection state from one completed validation epoch.

        Parameters:
            validation_loss: Full validation-partition loss.
            epoch: One-based epoch number.

        Returns:
            Whether this epoch became the selected checkpoint.
        """
        if not math.isfinite(validation_loss):
            raise ValueError("validation_loss must be finite")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self.should_stop = False
            return True
        self.epochs_without_improvement += 1
        self.should_stop = self.epochs_without_improvement >= self.patience
        return False

    def state(self, actual_epochs: int) -> dict[str, object]:
        """Return JSON-friendly selection and stopping state.

        Parameters:
            actual_epochs: Number of epochs that completed.

        Returns:
            Selection rule, selected epoch, and stopping state.
        """
        return {
            "selection_partition": "validation",
            "selection_metric": "loss",
            "selection_mode": "min",
            "best_validation_loss": self.best_loss,
            "selected_epoch": self.best_epoch,
            "patience": self.patience,
            "epochs_without_improvement": self.epochs_without_improvement,
            "actual_epochs": actual_epochs,
            "stopped_early": self.should_stop,
            "stop_reason": (
                "early_stopping_patience_reached"
                if self.should_stop
                else "maximum_epochs_completed"
            ),
        }


def validate_selected_checkpoint(
    checkpoint: dict[str, object],
    dataset_name: str,
    split_manifest_hash: str,
) -> dict[str, object]:
    """Validate validation-selected checkpoint provenance.

    Parameters:
        checkpoint: Loaded checkpoint object.
        dataset_name: Dataset required by the current command.
        split_manifest_hash: Manifest hash required by the current command.

    Returns:
        Validated checkpoint-selection provenance.
    """
    if checkpoint.get("checkpoint_role") != "validation_selected_lowest_loss":
        raise ValueError("checkpoint was not selected by lowest validation loss")
    if checkpoint.get("selection_partition") != "validation":
        raise ValueError("checkpoint selection partition must be validation")
    if checkpoint.get("selection_metric") != "loss":
        raise ValueError("checkpoint selection metric must be loss")
    if checkpoint.get("metric_protocol") != metric_protocol():
        raise ValueError("checkpoint metric protocol is missing or unsupported")
    if checkpoint.get("dataset_name") != dataset_name:
        raise ValueError("checkpoint dataset does not match the requested dataset")
    if checkpoint.get("split_manifest_hash") != split_manifest_hash:
        raise ValueError("checkpoint split manifest does not match this run")

    selected_epoch = checkpoint.get("selected_epoch")
    selection_value = checkpoint.get("selection_value")
    seed = checkpoint.get("seed")
    validation_metrics = checkpoint.get("validation_metrics")
    early_stopping = checkpoint.get("early_stopping")
    if (
        not isinstance(selected_epoch, int)
        or isinstance(selected_epoch, bool)
        or selected_epoch <= 0
    ):
        raise ValueError("checkpoint selected_epoch must be a positive integer")
    if (
        not isinstance(selection_value, (int, float))
        or isinstance(selection_value, bool)
        or not math.isfinite(float(selection_value))
    ):
        raise ValueError("checkpoint selection_value must be finite")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("checkpoint seed must be a non-negative integer")
    if not isinstance(validation_metrics, dict):
        raise ValueError("checkpoint validation_metrics are missing")
    for metric_name in (
        "loss",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        metric_value = validation_metrics.get(metric_name)
        if (
            not isinstance(metric_value, (int, float))
            or isinstance(metric_value, bool)
            or not math.isfinite(float(metric_value))
        ):
            raise ValueError(f"checkpoint validation {metric_name} must be finite")
    validation_loss = validation_metrics["loss"]
    if not isinstance(validation_loss, (int, float)) or not math.isclose(
        float(validation_loss), float(selection_value), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("checkpoint validation loss does not match selection_value")
    if not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("checkpoint model_config is missing")
    if not isinstance(early_stopping, dict):
        raise ValueError("checkpoint early_stopping state is missing")
    if early_stopping.get("selected_epoch") != selected_epoch:
        raise ValueError("checkpoint early-stopping epoch does not match selection")
    return {
        "checkpoint_role": checkpoint["checkpoint_role"],
        "selection_partition": checkpoint["selection_partition"],
        "selection_metric": checkpoint["selection_metric"],
        "selection_value": float(selection_value),
        "selected_epoch": selected_epoch,
        "dataset_name": checkpoint["dataset_name"],
        "split_manifest_hash": checkpoint["split_manifest_hash"],
        "seed": seed,
        "metric_protocol": checkpoint["metric_protocol"],
        "early_stopping": early_stopping,
    }


def rank_validation_results(
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rank candidates using validation metrics and parameter count only.

    Parameters:
        results: Candidate records containing validation metrics and parameters.

    Returns:
        Candidates ordered by F1, accuracy, parameters, and name.
    """

    def ranking_key(result: dict[str, object]) -> tuple[float, float, int, str]:
        metrics = result.get("validation_metrics")
        if not isinstance(metrics, dict):
            raise ValueError("candidate validation_metrics are missing")
        f1 = metrics.get("macro_f1")
        accuracy = metrics.get("accuracy")
        parameters = result.get("num_params")
        name = result.get("name")
        if (
            not isinstance(f1, (int, float))
            or isinstance(f1, bool)
            or not isinstance(accuracy, (int, float))
            or isinstance(accuracy, bool)
        ):
            raise ValueError("candidate validation F1 or accuracy is missing")
        if not isinstance(parameters, int) or not isinstance(name, str):
            raise ValueError("candidate parameter count or name is missing")
        return (-float(f1), -float(accuracy), parameters, name)

    return sorted(results, key=ranking_key)
