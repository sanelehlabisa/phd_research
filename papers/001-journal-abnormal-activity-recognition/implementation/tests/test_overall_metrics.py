"""Overall metric aggregation, historical provenance and validation-only ranking."""

from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import (
    ValidationLossSelector,
    classification_metrics,
    evaluate_classifier,
    metric_protocol,
    rank_validation_results,
    train_classifier_epoch,
    validate_selected_checkpoint,
)


def test_micro_metrics_match_hand_computed_imbalanced_classes():
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    predicted = torch.tensor([0, 0, 0, 1, 0, 1, 2])
    # TP=5, FP=2, FN=2 across all classes; each micro score is 5/7.
    result = classification_metrics(predicted, targets, 3)
    assert result == pytest.approx(
        dict(accuracy=5 / 7, precision=5 / 7, recall=5 / 7, f1=5 / 7)
    )
    assert metric_protocol()["precision_recall_f1_average"] == "micro"


def test_full_partition_metrics_and_weighted_loss_ignore_batch_boundaries():
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    predicted = torch.tensor([0, 0, 0, 1, 0, 1, 2])
    inputs = torch.nn.functional.one_hot(predicted, 3).float() * 3
    criterion = torch.nn.CrossEntropyLoss()
    expected = dict(
        loss=criterion(inputs, targets).item(),
        **classification_metrics(predicted, targets, 3)
    )
    model = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
    for batch_size in (2, 3, 7):
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
        assert evaluate_classifier(
            model, loader, criterion, torch.device("cpu"), 3
        ) == pytest.approx(expected)
        assert train_classifier_epoch(
            model,
            loader,
            criterion,
            torch.optim.SGD(model.parameters(), lr=0),
            torch.device("cpu"),
            3,
        ) == pytest.approx(expected)


def test_ranking_uses_accuracy_then_loss_then_size_not_test_or_f1():
    def candidate(name, accuracy, loss, parameters):
        return dict(
            name=name,
            num_params=parameters,
            partition="validation",
            validation_metrics=dict(accuracy=accuracy, loss=loss, f1=0),
            test_metrics=dict(accuracy=1),
        )

    rows = [
        candidate("a", 0.8, 0.5, 10),
        candidate("b", 0.8, 0.2, 200),
        candidate("c", 0.8, 0.2, 100),
        candidate("d", 0.7, 0.01, 1),
    ]
    rows[0]["validation_metrics"]["f1"] = 1  # Must not drive selection.
    assert [row["name"] for row in rank_validation_results(rows)] == [
        "c",
        "b",
        "a",
        "d",
    ]
    rows[0]["partition"] = "test"
    with pytest.raises(ValueError, match="never test"):
        rank_validation_results(rows)
    selector = ValidationLossSelector(3)
    assert selector.update(0.5, 1)
    assert not selector.update(0.8, 2)
    assert selector.update(0.2, 3)
    assert selector.best_epoch == 3


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_ranking_rejects_invalid_scores(value):
    with pytest.raises(ValueError):
        rank_validation_results(
            [
                dict(
                    name="bad",
                    num_params=1,
                    validation_metrics=dict(accuracy=value, loss=0.5),
                )
            ]
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_protocol_is_validated_without_rewriting(legacy):
    protocol = metric_protocol()
    prefix = ""
    if legacy:
        protocol = dict(protocol, precision_recall_f1_average="macro")
        prefix = "macro_"
    checkpoint = dict(
        checkpoint_role="validation_selected_lowest_loss",
        selection_partition="validation",
        selection_metric="loss",
        selected_epoch=1,
        selection_value=0.4,
        seed=42,
        dataset_name="fixture",
        split_manifest_hash="split",
        metric_protocol=protocol,
        model_config={},
        early_stopping=dict(selected_epoch=1),
        validation_metrics={
            "loss": 0.4,
            "accuracy": 0.5,
            prefix + "precision": 0.5,
            prefix + "recall": 0.5,
            prefix + "f1": 0.5,
        },
    )
    original = deepcopy(checkpoint)
    result = validate_selected_checkpoint(checkpoint, "fixture", "split")
    assert result["metric_protocol"] == protocol and checkpoint == original
    checkpoint["metric_protocol"] = dict(
        protocol, precision_recall_f1_average="weighted"
    )
    with pytest.raises(ValueError, match="unsupported"):
        validate_selected_checkpoint(checkpoint, "fixture", "split")
