"""Tests for bounded smoke-training updates."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import train_classifier_steps


def test_train_classifier_steps_runs_exact_updates() -> None:
    """Train across loader boundaries and return one record per update."""
    torch.manual_seed(7)
    inputs = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
    model = nn.Linear(4, 2)
    initial_weight = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    records = train_classifier_steps(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        max_steps=5,
    )

    assert [record["step"] for record in records] == [1, 2, 3, 4, 5]
    assert all(record["batch_size"] == 2 for record in records)
    assert all(torch.isfinite(torch.tensor(record["loss"])) for record in records)
    assert not torch.equal(initial_weight, model.weight)


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5])
def test_train_classifier_steps_rejects_invalid_limits(max_steps: object) -> None:
    """Reject non-positive and non-integer update limits."""
    loader = DataLoader(
        TensorDataset(torch.randn(2, 4), torch.tensor([0, 1])),
        batch_size=2,
    )
    model = nn.Linear(4, 2)

    with pytest.raises(ValueError, match="max_steps"):
        train_classifier_steps(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            torch.device("cpu"),
            max_steps=max_steps,
        )
