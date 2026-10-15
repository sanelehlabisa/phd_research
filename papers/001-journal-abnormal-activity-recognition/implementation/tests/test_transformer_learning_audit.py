"""Transformer wiring and train-only diagnostic checks without expensive models."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import experiments
from src.metrics import evaluate_classifier, train_classifier_steps
from notebooks.utils import diagnostics


@pytest.fixture(autouse=True, scope="module")
def small_cpu_runtime():
    """Keep tiny synthetic checks fast without changing later test modules."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


class TinyAttention(nn.Module):
    """Provide a small trainable stand-in for Swin's attention projection."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.qkv(inputs).reshape(inputs.shape[0], 3, 4).mean(dim=1)


class TinySwin(nn.Module):
    """Expose Swin's important parameter paths using a tiny CPU-only network."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(3, 4, kernel_size=1)
        block = nn.Module()
        block.attn = TinyAttention()
        self.features = nn.Sequential(nn.Sequential(block))
        self.head = nn.Linear(4, 400)
        self.received: list[torch.Tensor] = []
        self.modes: list[bool] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.received.append(inputs.detach().clone())
        self.modes.append(self.training)
        features = self.patch_embed.proj(inputs).mean(dim=(2, 3, 4))
        features = self.features[0][0].attn(features)
        return self.head(features)


@pytest.mark.parametrize("name", ["swin3d_t", "swin3d_s"])
def test_transformer_builder_preserves_pixels_and_replaces_class_head(
    name: str, monkeypatch
) -> None:
    """Both builders use scratch weights and exactly transpose the input axes."""
    base = TinySwin()
    requested = []

    def factory(*, weights):
        requested.append(weights)
        return base

    monkeypatch.setattr(experiments.video_models, name, factory)
    model = experiments.build_registered_model(name, 11, (3, 16, 12), 8)
    inputs = torch.linspace(0, 1, 2 * 8 * 3 * 16 * 12).reshape(2, 8, 3, 16, 12)

    logits = model(inputs)

    assert requested == [None]
    assert base.head.out_features == 11
    assert logits.shape == (2, 11)
    assert torch.equal(base.received[-1], inputs.permute(0, 2, 1, 3, 4))
    assert base.received[-1].shape == (2, 3, 8, 16, 12)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", ["swin3d_t", "swin3d_s"])
def test_transformer_shared_training_updates_features_attention_and_head(
    name: str, monkeypatch
) -> None:
    """The existing optimizer/training path updates the head and backbone."""
    torch.manual_seed(42)
    base = TinySwin()
    monkeypatch.setattr(experiments.video_models, name, lambda **kwargs: base)
    model = experiments.build_registered_model(name, 11, (3, 4, 4), 8)
    clips = torch.rand(4, 8, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(clips, labels), batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert actual == expected

    parameters = {
        "patch": base.patch_embed.proj.weight,
        "attention": base.features[0][0].attn.qkv.weight,
        "head": base.head.weight,
    }
    before = {
        name: parameter.detach().clone() for name, parameter in parameters.items()
    }
    gradients = {name: [] for name in parameters}
    handles = [
        parameter.register_hook(
            lambda gradient, name=name: gradients[name].append(
                gradient.detach().clone()
            )
        )
        for name, parameter in parameters.items()
    ]
    try:
        history = train_classifier_steps(
            model,
            loader,
            nn.CrossEntropyLoss(),
            optimizer,
            torch.device("cpu"),
            max_steps=4,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert [row["step"] for row in history] == [1, 2, 3, 4]
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)
    assert all(base.modes)
    for name, parameter in parameters.items():
        assert len(gradients[name]) == 4
        assert all(torch.isfinite(gradient).all() for gradient in gradients[name])
        assert all(float(gradient.norm()) > 0 for gradient in gradients[name])
        assert not torch.equal(parameter.detach(), before[name])

    result = evaluate_classifier(
        model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), num_classes=11
    )
    assert not model.training and not base.modes[-1]
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert 0 <= result["accuracy"] <= 1


@pytest.mark.parametrize("name", ["swin3d_t", "swin3d_s"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_registered_check_records_real_shared_training_hooks(
    name: str, batch_size: int, monkeypatch
) -> None:
    """The diagnostic traces actual optimizer hooks rather than inferred updates."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(experiments.video_models, name, lambda **kwargs: TinySwin())
    clips = torch.linspace(0, 1, 4 * 8 * 3 * 4 * 4).reshape(4, 8, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1])

    result = diagnostics.registered_learning_check(
        name, clips, labels, num_classes=11, steps=9, batch_size=batch_size
    )

    assert result["optimizer_covers_all_parameters"]
    assert result["device"] == "cpu" and result["peak_allocated_bytes"] is None
    assert result["recipe"]["learning_rate"] == 0.001
    assert result["recipe"]["weight_decay"] == 0
    assert result["recipe"]["weights"] is None
    assert result["recipe"]["seed"] == 42
    assert result["recipe"]["determinism"]
    assert result["input_shape_btchw"] == [4, 8, 3, 4, 4]
    assert result["backbone_input_bcthw"]["shape"] == [batch_size, 3, 8, 4, 4]
    assert result["num_classes"] == 11 and result["labels"] == [0, 1, 0, 1]
    assert [row["step"] for row in result["learning"]] == [0, 8, 9]
    assert [row["step"] for row in result["step_metrics"]] == list(range(1, 10))
    assert len(result["gradient_updates"]) == 9
    for row in result["learning"]:
        assert sum(row["prediction_counts"].values()) == 4
        assert torch.isfinite(torch.tensor(row["loss"]))
    for row in result["gradient_updates"]:
        assert row["training"] and row["learning_rate"] == 0.001
        assert row["all_gradients_present_and_finite"]
        assert len(row["parameters"]) == 3
        for parameter in row["parameters"].values():
            assert parameter["gradient_finite"]
            assert parameter["gradient_norm"] > 0
            assert parameter["update_norm"] > 0


@pytest.mark.parametrize("steps", [0, -1, 129, True, 1.5])
def test_diagnostic_rejects_unbounded_steps_before_data_access(steps, tmp_path) -> None:
    """Bad update budgets fail before opening files or building a model."""
    with pytest.raises(ValueError, match="steps"):
        diagnostics.audit_transformer_learning(tmp_path, steps=steps)
    with pytest.raises(ValueError, match="steps"):
        diagnostics.registered_learning_check(
            "swin3d_t", torch.empty(0), torch.empty(0), 11, steps=steps
        )


@pytest.mark.parametrize("batch_size", [0, -1, 3, True, 1.5])
def test_diagnostic_rejects_unsafe_batch_before_data_access(
    batch_size, tmp_path
) -> None:
    """The bounded diagnostic accepts only one or two clips per update."""
    with pytest.raises(ValueError, match="batch_size"):
        diagnostics.audit_transformer_learning(tmp_path, batch_size=batch_size)
    with pytest.raises(ValueError, match="batch_size"):
        diagnostics.registered_learning_check(
            "swin3d_t", torch.empty(0), torch.empty(0), 11, batch_size=batch_size
        )


@pytest.mark.parametrize(
    "clips",
    [
        torch.zeros(4, 3, 4, 4),
        torch.zeros(4, 8, 1, 4, 4),
        torch.zeros(0, 8, 3, 4, 4),
        torch.zeros(4, 8, 3, 4, 4, dtype=torch.uint8),
        torch.full((4, 8, 3, 4, 4), float("nan")),
        torch.full((4, 8, 3, 4, 4), -0.1),
        torch.full((4, 8, 3, 4, 4), 1.1),
    ],
)
def test_diagnostic_rejects_invalid_clip_data(clips) -> None:
    """Bad channels, axes, dtype, range and finite values cannot reach training."""
    with pytest.raises(ValueError, match="clips"):
        diagnostics.registered_learning_check(
            "swin3d_t", clips, torch.tensor([0, 1, 0, 1]), 11, steps=1
        )


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([-1, 1, 0, 1]),
        torch.tensor([0, 11, 0, 1]),
    ],
)
def test_diagnostic_rejects_invalid_original_class_targets(labels) -> None:
    """Keep original class indices and the full dataset classifier head."""
    with pytest.raises(ValueError, match="labels"):
        diagnostics.registered_learning_check(
            "swin3d_t", torch.zeros(4, 8, 3, 4, 4), labels, 11, steps=1
        )


def test_diagnostic_rejects_unregistered_models() -> None:
    """This ticket cannot silently expand the studied model families."""
    with pytest.raises(ValueError, match="two registered Swin"):
        diagnostics.registered_learning_check(
            "r3d_18", torch.zeros(4, 8, 3, 4, 4), torch.tensor([0, 1, 0, 1]), 11
        )


@pytest.mark.parametrize(
    "learning_rate", [0, -0.001, float("nan"), float("inf"), True, "0.001"]
)
def test_diagnostic_rejects_invalid_learning_rates(learning_rate) -> None:
    """A diagnostic recipe must record a finite positive numeric rate."""
    with pytest.raises(ValueError, match="learning_rate"):
        diagnostics.registered_learning_check(
            "swin3d_t",
            torch.zeros(4, 8, 3, 4, 4),
            torch.tensor([0, 1, 0, 1]),
            11,
            steps=1,
            learning_rate=learning_rate,
        )


def test_diagnostic_applies_and_records_explicit_lower_learning_rate(
    monkeypatch,
) -> None:
    """An explicit rate changes actual Adam steps, not only saved metadata."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        experiments.video_models, "swin3d_t", lambda **kwargs: TinySwin()
    )
    result = diagnostics.registered_learning_check(
        "swin3d_t",
        torch.rand(4, 8, 3, 4, 4),
        torch.tensor([0, 1, 0, 1]),
        11,
        steps=2,
        learning_rate=0.0001,
    )
    assert result["recipe"]["learning_rate"] == 0.0001
    assert all(row["learning_rate"] == 0.0001 for row in result["gradient_updates"])
    assert all(
        row["all_gradients_present_and_finite"] for row in result["gradient_updates"]
    )


@pytest.fixture
def existing_train_only_data(tmp_path, monkeypatch):
    """Provide an existing manifest with guarded synthetic held-out clips."""
    from src import dataset as dataset_module, dataset_source

    config = tmp_path / "configs/experiments/aad_shortlist_comparison_colab.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"dataset": {"path": "datasets/local-aad"}}))
    for relative in (
        "notebooks/utils/diagnostics.py",
        "src/experiments.py",
        "src/metrics.py",
        "src/dataset.py",
        "src/temporal_sampling.py",
    ):
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# Synthetic source identity for this isolated test.\n")
    data_path = tmp_path / "datasets/local-aad"
    data_path.mkdir(parents=True)
    split_path = tmp_path / "splits/abnormal-activities-dataset_seed42.json"
    split_path.parent.mkdir()
    split_path.write_bytes(b"original split evidence, never replace")
    original_split = split_path.read_bytes()
    decoded = []
    construction = []
    train_indices = [0, 1, 3, 4]
    samples = []
    for label in (0, 1):
        directory = data_path / f"class-{label}"
        directory.mkdir()
        for number in range(3):
            path = directory / f"{number}.mp4"
            path.write_bytes(f"synthetic {label} {number}".encode())
            samples.append((path, label))

    class GuardedDataset:
        class_names = [f"class-{label}" for label in range(11)]
        num_classes = 11

        def __init__(self, dataset_dir, **kwargs):
            assert dataset_dir == data_path
            construction.append(kwargs)
            self.samples = samples
            self.sampling_plans = {}

        def __getitem__(self, index):
            assert index in train_indices, "held-out clip must never be decoded"
            decoded.append(index)
            path = self.samples[index][0]
            self.sampling_plans[str(path.resolve())] = {
                "sampling_version": "timestamps_v1",
                "selected_timestamps": [0.0],
            }
            return torch.full((8, 3, 64, 64), index / 10), self.samples[index][1]

    def load_existing(dataset, manifest_path):
        assert manifest_path == split_path
        assert split_path.read_bytes() == original_split
        return (
            SimpleNamespace(indices=train_indices),
            SimpleNamespace(indices=[2]),
            SimpleNamespace(indices=[5]),
            {"manifest_path": str(split_path), "manifest_hash": "synthetic-existing"},
        )

    def forbid_download(*args, **kwargs):
        raise AssertionError("diagnosis must not download or resolve another dataset")

    monkeypatch.setattr(dataset_module, "AHARDataset", GuardedDataset)
    monkeypatch.setattr(dataset_module, "load_split_subsets", load_existing)
    monkeypatch.setattr(dataset_source, "resolve_dataset", forbid_download)
    return SimpleNamespace(
        root=tmp_path,
        split_path=split_path,
        original_split=original_split,
        decoded=decoded,
        construction=construction,
    )


def test_audit_decodes_only_fixed_training_clips_and_preserves_split(
    existing_train_only_data, monkeypatch
) -> None:
    """The saved audit uses balanced interleaved training clips, never holdouts."""
    data = existing_train_only_data
    requested = []

    def fake_check(name, clips, labels, num_classes, steps, batch_size):
        requested.append(name)
        assert clips.shape == (4, 8, 3, 64, 64)
        assert labels.tolist() == [0, 1, 0, 1]
        assert num_classes == 11 and steps == 8 and batch_size == 1
        return {"model": name, "synthetic_test_only": True}

    monkeypatch.setattr(diagnostics, "registered_learning_check", fake_check)
    result = diagnostics.audit_transformer_learning(data.root, steps=8, batch_size=1)

    assert result["status"] == "complete"
    assert requested == ["swin3d_t", "swin3d_s"]
    assert data.decoded == [0, 3, 1, 4]
    assert data.construction == [
        {
            "sequence_length": 8,
            "frame_size": (64, 64),
            "target_fps": 16,
            "sampling_version": "timestamps_v1",
        }
    ]
    assert result["validation_clips_opened"] == result["test_clips_opened"] == 0
    assert all(source["partition"] == "train" for source in result["sources"])
    assert [source["index"] for source in result["sources"]] == data.decoded
    assert all(len(source["sha256"]) == 64 for source in result["sources"])
    assert all(
        source["sampling"]["sampling_version"] == "timestamps_v1"
        for source in result["sources"]
    )
    assert len(result["source_sha256"]) == 5
    assert all(len(value) == 64 for value in result["source_sha256"].values())
    assert data.split_path.read_bytes() == data.original_split
    saved = json.loads((Path(result["run_dir"]) / "diagnosis.json").read_text())
    assert saved == result


def test_audit_requires_existing_split_without_creation(
    existing_train_only_data,
) -> None:
    """Missing split evidence stops before dataset construction or decoding."""
    data = existing_train_only_data
    data.split_path.unlink()
    with pytest.raises(FileNotFoundError, match="no download/resplit"):
        diagnostics.audit_transformer_learning(data.root, steps=1)
    assert not data.split_path.exists()
    assert data.decoded == data.construction == []


def test_audit_preserves_partial_evidence_if_second_model_fails(
    existing_train_only_data, monkeypatch
) -> None:
    """A failed bounded check retains earlier traces without opening test data."""
    data = existing_train_only_data

    def fail_second(name, *args):
        if name == "swin3d_s":
            raise RuntimeError("synthetic check failure")
        return {"model": name, "synthetic_test_only": True}

    monkeypatch.setattr(diagnostics, "registered_learning_check", fail_second)
    with pytest.raises(RuntimeError, match="synthetic check failure"):
        diagnostics.audit_transformer_learning(data.root, steps=1)

    outputs = list((data.root / "runs/experiments").glob("*/diagnosis.json"))
    assert len(outputs) == 1
    saved = json.loads(outputs[0].read_text())
    assert saved["status"] == "failed"
    assert saved["error"] == "synthetic check failure"
    assert list(saved["models"]) == ["swin3d_t"]
    assert saved["validation_clips_opened"] == saved["test_clips_opened"] == 0
    assert data.decoded == [0, 3, 1, 4]
    assert data.split_path.read_bytes() == data.original_split
