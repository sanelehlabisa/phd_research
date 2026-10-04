"""Bounded VDD pipeline-learnability diagnostic.

This command is deliberately separate from the controlled AAD experiment plan.
It may inspect file metadata for the full VDD inventory, but it decodes and trains
on the training and validation partitions only. The test partition stays locked.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from statistics import median

import av
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .dataset import AHARDataset, create_split_manifest, load_split_subsets
from .metrics import evaluate_classifier, train_classifier_epoch
from .model import CustomConvLSTM, count_trainable_parameters
from .utils import (
    RunContext,
    data_loader_generator,
    plot_training_curves,
    runtime_environment,
    seed_data_loader_worker,
    seed_everything,
    write_json,
)

EVIDENCE_ROLE = "pipeline_learnability_diagnostic"
VIDEO_EXTENSIONS = AHARDataset.SUPPORTED_EXTS


def resolve_vdd_root(download_root: str | Path) -> Path:
    """Find the shallowest directory containing at least two video classes."""
    root = Path(download_root).resolve()
    candidates: list[Path] = []
    for candidate in (root, *sorted(path for path in root.rglob("*") if path.is_dir())):
        class_dirs = [path for path in candidate.iterdir() if path.is_dir()]
        populated = [
            path
            for path in class_dirs
            if any(item.suffix.lower() in VIDEO_EXTENSIONS for item in path.iterdir())
        ]
        if len(populated) >= 2:
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"could not find a directory with at least two video class folders under {root}"
        )
    return min(candidates, key=lambda path: (len(path.parts), path.as_posix()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_vdd(dataset: AHARDataset) -> dict[str, object]:
    """Record formats, container metadata, corruption, and exact duplicates."""
    records: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = {}
    class_counts = Counter()
    for path, label in dataset.samples:
        relative_path = path.resolve().relative_to(dataset.dataset_dir.resolve()).as_posix()
        record: dict[str, object] = {
            "path": relative_path,
            "class_name": dataset.class_names[label],
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "readable": False,
        }
        class_counts[dataset.class_names[label]] += 1
        try:
            with av.open(str(path)) as container:
                stream = next(stream for stream in container.streams if stream.type == "video")
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else None
                )
                record.update(
                    {
                        "readable": True,
                        "duration_seconds": duration,
                        "fps": float(stream.average_rate) if stream.average_rate else None,
                        "width": stream.width,
                        "height": stream.height,
                        "codec": stream.codec_context.name,
                    }
                )
        except Exception as error:  # inventory must retain failures, not hide them
            record["error"] = f"{type(error).__name__}: {error}"
        content_hash = _sha256(path)
        record["sha256"] = content_hash
        hashes.setdefault(content_hash, []).append(relative_path)
        records.append(record)

    durations = [
        float(record["duration_seconds"])
        for record in records
        if record.get("duration_seconds") is not None
    ]
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "evidence_role": EVIDENCE_ROLE,
        "dataset_root": str(dataset.dataset_dir.resolve()),
        "classes": dataset.class_names,
        "class_counts": dict(sorted(class_counts.items())),
        "clip_count": len(records),
        "formats": dict(sorted(Counter(record["extension"] for record in records).items())),
        "corrupt_or_unreadable": [record for record in records if not record["readable"]],
        "duration_seconds": {
            "count": len(durations),
            "minimum": min(durations) if durations else None,
            "median": median(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
        "exact_duplicate_groups": duplicate_groups,
        "duplicate_risk": (
            "Exact byte duplicates are listed; visually equivalent re-encodes are not detected."
        ),
        "records": records,
    }


def balanced_tiny_indices(
    dataset: AHARDataset, train_indices: list[int], per_class: int
) -> list[int]:
    """Select the first stable, balanced samples from the training partition."""
    selected: list[int] = []
    counts = Counter()
    for index in sorted(train_indices, key=lambda item: dataset.samples[item][0].as_posix()):
        label = dataset.samples[index][1]
        if counts[label] < per_class:
            selected.append(index)
            counts[label] += 1
    missing = [name for index, name in enumerate(dataset.class_names) if counts[index] < per_class]
    if missing:
        raise ValueError(f"not enough training clips for tiny subset classes: {missing}")
    return selected


def _loader(dataset: object, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_data_loader_worker,
        generator=data_loader_generator(seed),
    )


def _gradient_diagnostics(model: nn.Module) -> dict[str, object]:
    norms = {
        name: float(parameter.grad.detach().norm().item())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


@torch.inference_mode()
def _batch_diagnostics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    """Capture one reproducible batch's labels, predictions, and tensor ranges."""
    inputs, labels = next(iter(loader))
    logits = model(inputs.to(device)).cpu()
    return {
        "input_shape": list(inputs.shape),
        "input_minimum": float(inputs.min().item()),
        "input_maximum": float(inputs.max().item()),
        "labels": labels.tolist(),
        "label_counts": dict(sorted(Counter(labels.tolist()).items())),
        "predictions": logits.argmax(dim=1).tolist(),
        "logits": logits.tolist(),
    }
    return {
        "parameter_count_with_gradients": len(norms),
        "zero_gradient_parameters": sorted(name for name, value in norms.items() if value == 0.0),
        "gradient_norms": norms,
    }


def _train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    stop_accuracy: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}
    for epoch in range(1, epochs + 1):
        training = train_classifier_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        validation = evaluate_classifier(
            model, validation_loader, criterion, device, num_classes
        )
        history.append(
            {"epoch": epoch, "training_metrics": training, "validation_metrics": validation}
        )
        diagnostics = _gradient_diagnostics(model)
        print(
            f"epoch={epoch:02d} train_loss={training['loss']:.4f} "
            f"train_acc={training['accuracy']:.4f} val_acc={validation['accuracy']:.4f}"
        )
        if stop_accuracy is not None and training["accuracy"] >= stop_accuracy:
            break
    return history, diagnostics


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--accepted-class",
        action="append",
        dest="accepted_classes",
        help="Exact class directory to include; repeat for each accepted class.",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--manifest", default="splits/vdd_diagnostic_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tiny-per-class", type=int, default=4)
    parser.add_argument("--tiny-epochs", type=int, default=30)
    parser.add_argument("--bounded-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    if args.tiny_per_class <= 0 or args.tiny_epochs <= 0 or args.bounded_epochs <= 0:
        raise ValueError("subset size and epoch budgets must be positive")
    deterministic = seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = resolve_vdd_root(args.dataset_dir)
    dataset = AHARDataset(
        dataset_root,
        sequence_length=args.sequence_length,
        frame_size=(args.frame_size, args.frame_size),
        accepted_classes=args.accepted_classes,
    )
    if dataset.num_classes < 2:
        raise ValueError(f"diagnostic requires at least two classes, found {dataset.class_names}")

    run = RunContext(
        args.runs_dir,
        purpose="train",
        dataset_path=dataset_root,
        label="vdd-pipeline-learnability-diagnostic",
        arguments=vars(args),
        metadata={"evidence_role": EVIDENCE_ROLE, "test_access": "locked"},
    )
    inventory = inspect_vdd(dataset)
    write_json(run.run_dir / "dataset_inventory.json", inventory)
    if inventory["corrupt_or_unreadable"]:
        raise RuntimeError("VDD inventory contains unreadable clips; see dataset_inventory.json")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        create_split_manifest(dataset, manifest_path, seed=args.seed)
    train_set, validation_set, test_set, split = load_split_subsets(
        dataset, manifest_path, seed=args.seed
    )
    paths_by_split = {
        name: {dataset.samples[index][0].resolve() for index in subset.indices}
        for name, subset in (
            ("train", train_set),
            ("validation", validation_set),
            ("test", test_set),
        )
    }
    overlap = {
        "train_validation": sorted(str(path) for path in paths_by_split["train"] & paths_by_split["validation"]),
        "train_test": sorted(str(path) for path in paths_by_split["train"] & paths_by_split["test"]),
        "validation_test": sorted(str(path) for path in paths_by_split["validation"] & paths_by_split["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"split path overlap detected: {overlap}")
    split_for_path = {
        path: split_name for split_name, paths in paths_by_split.items() for path in paths
    }
    duplicate_split_risks = []
    for paths in inventory["exact_duplicate_groups"]:
        memberships = {
            split_for_path[dataset_root / relative_path] for relative_path in paths
        }
        if len(memberships) > 1:
            duplicate_split_risks.append({"paths": paths, "splits": sorted(memberships)})
    if duplicate_split_risks:
        raise RuntimeError(
            "exact duplicate content crosses split boundaries; see dataset_inventory.json"
        )

    tiny_indices = balanced_tiny_indices(dataset, list(train_set.indices), args.tiny_per_class)
    tiny_set = Subset(dataset, tiny_indices)
    model_config = {
        "layers": [(8, (3, 3)), (16, (3, 3))],
        "num_classes": dataset.num_classes,
    }
    tiny_model = CustomConvLSTM(**model_config).to(device)
    tiny_loader = _loader(tiny_set, min(args.batch_size, len(tiny_set)), True, args.seed)
    tiny_eval_loader = _loader(tiny_set, min(args.batch_size, len(tiny_set)), False, args.seed + 1)
    tiny_history, tiny_diagnostics = _train_phase(
        tiny_model,
        tiny_loader,
        tiny_eval_loader,
        device,
        dataset.num_classes,
        args.tiny_epochs,
        args.learning_rate,
        args.weight_decay,
        stop_accuracy=0.95,
    )
    tiny_loss_decreased = (
        tiny_history[-1]["training_metrics"]["loss"]
        < tiny_history[0]["training_metrics"]["loss"]
    )
    tiny_passed = (
        tiny_history[-1]["training_metrics"]["accuracy"] >= 0.95
        and tiny_loss_decreased
    )
    result: dict[str, object] = {
        "evidence_role": EVIDENCE_ROLE,
        "test_access": "locked",
        "dataset_inventory": inventory,
        "split": split,
        "split_path_overlap": overlap,
        "exact_duplicate_split_risks": duplicate_split_risks,
        "tiny_subset_indices": tiny_indices,
        "tiny_history": tiny_history,
        "tiny_diagnostics": tiny_diagnostics,
        "tiny_batch_diagnostics": _batch_diagnostics(
            tiny_model, tiny_eval_loader, device
        ),
        "tiny_loss_decreased": tiny_loss_decreased,
        "tiny_passed": tiny_passed,
        "environment": runtime_environment(),
        "deterministic_settings": deterministic,
        "model": tiny_model.configuration(),
        "trainable_parameters": count_trainable_parameters(tiny_model),
    }
    write_json(run.run_dir / "metrics" / "tiny_subset.json", result)
    plot_training_curves(
        [item["training_metrics"]["loss"] for item in tiny_history],
        [item["validation_metrics"]["loss"] for item in tiny_history],
        [item["training_metrics"]["accuracy"] for item in tiny_history],
        [item["validation_metrics"]["accuracy"] for item in tiny_history],
        run.run_dir / "plots" / "tiny_subset_curves.png",
        show=False,
    )
    if not tiny_passed:
        result["outcome"] = "reproducible_tiny_subset_failure"
        write_json(run.run_dir / "result.json", result)
        run.complete(
            artifacts={
                "inventory": str(run.run_dir / "dataset_inventory.json"),
                "result": str(run.run_dir / "result.json"),
                "tiny_metrics": str(run.run_dir / "metrics" / "tiny_subset.json"),
                "tiny_curves": str(run.run_dir / "plots" / "tiny_subset_curves.png"),
            },
            results=result,
        )
        raise SystemExit("Tiny-subset accuracy did not reach 95%; bounded VDD run was not started")

    bounded_model = CustomConvLSTM(**model_config).to(device)
    bounded_history, bounded_diagnostics = _train_phase(
        bounded_model,
        _loader(train_set, args.batch_size, True, args.seed),
        _loader(validation_set, args.batch_size, False, args.seed + 1),
        device,
        dataset.num_classes,
        args.bounded_epochs,
        args.learning_rate,
        args.weight_decay,
    )
    checkpoint_path = run.run_dir / "checkpoints" / "bounded_last_epoch.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "evidence_role": EVIDENCE_ROLE,
            "test_access": "locked",
            "model_config": bounded_model.configuration(),
            "model_state_dict": bounded_model.state_dict(),
            "seed": args.seed,
            "split_manifest_hash": split["manifest_hash"],
        },
        checkpoint_path,
    )
    result.update(
        {
            "bounded_history": bounded_history,
            "bounded_diagnostics": bounded_diagnostics,
            "bounded_training_exceeded_70_percent": max(
                item["training_metrics"]["accuracy"] for item in bounded_history
            ) >= 0.70,
            "outcome": "bounded_vdd_diagnostic_completed",
        }
    )
    write_json(run.run_dir / "metrics" / "bounded_history.json", bounded_history)
    write_json(run.run_dir / "result.json", result)
    plot_training_curves(
        [item["training_metrics"]["loss"] for item in bounded_history],
        [item["validation_metrics"]["loss"] for item in bounded_history],
        [item["training_metrics"]["accuracy"] for item in bounded_history],
        [item["validation_metrics"]["accuracy"] for item in bounded_history],
        run.run_dir / "plots" / "bounded_curves.png",
        show=False,
    )
    run.complete(
        artifacts={
            "inventory": str(run.run_dir / "dataset_inventory.json"),
            "result": str(run.run_dir / "result.json"),
            "tiny_metrics": str(run.run_dir / "metrics" / "tiny_subset.json"),
            "tiny_curves": str(run.run_dir / "plots" / "tiny_subset_curves.png"),
            "bounded_history": str(run.run_dir / "metrics" / "bounded_history.json"),
            "bounded_curves": str(run.run_dir / "plots" / "bounded_curves.png"),
            "bounded_checkpoint": str(checkpoint_path),
        },
        results=result,
    )
    print(f"Diagnostic artifacts: {run.run_dir}")


if __name__ == "__main__":
    main()
