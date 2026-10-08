"""Training-only learnability and train/validation source-video audits."""

import hashlib
import time
from collections import Counter

import av
import numpy as np
import torch

from . import config as settings
from src.model import CustomConvLSTM
from src.utils import seed_everything, write_json


def balanced_tiny_indices(dataset, train_indices, per_class):
    """Choose a stable, balanced subset using training indices only."""
    selected = []
    counts = Counter()
    for index in sorted(
        train_indices, key=lambda item: dataset.samples[item][0].as_posix()
    ):
        label = dataset.samples[index][1]
        if counts[label] < per_class:
            selected.append(index)
            counts[label] += 1
    missing = [
        name
        for index, name in enumerate(dataset.class_names)
        if counts[index] < per_class
    ]
    if missing:
        raise ValueError(f"not enough training clips for tiny subset classes: {missing}")
    return selected


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Eight-hour suite budget exhausted; no final winner/test")


def audit_sources(prepared, destination, deadline):
    """Decode allowed partitions only; flag timing/repetition, never fix originals."""
    dataset = prepared["dataset"]
    rows = []
    for partition in ("train", "validation"):
        for index in prepared[partition].indices:
            check_deadline(deadline)
            if index not in dataset.allowed_indices:
                raise PermissionError("Audit attempted to open a locked clip")
            path, label = dataset.samples[index]
            times, repeated, previous = [], 0, None
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or 0)
                for frame in container.decode(stream):
                    check_deadline(deadline)
                    if frame.time is None:
                        raise ValueError(f"Missing source timestamps: {path}")
                    times.append(float(frame.time))
                    thumbnail = frame.to_ndarray(format="gray", width=16, height=16)
                    repeated += int(
                        previous is not None and np.array_equal(previous, thumbnail)
                    )
                    previous = thumbnail
            gaps = np.diff(times)
            if not len(gaps) or np.any(gaps <= 0) or fps <= 0:
                raise ValueError(
                    f"Invalid source timing: {path}; inspect before training"
                )
            measured = float(1 / np.median(gaps))
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            row = dict(
                partition=partition,
                source=str(path),
                sha256=checksum,
                label=dataset.class_names[label],
                native_fps=fps,
                timestamp_fps=measured,
                duration=times[-1] - times[0] + 1 / measured,
                frames=len(times),
                repeated_thumbnail_fraction=repeated / len(gaps),
                variable_timing=bool(np.max(gaps) - np.min(gaps) > 0.002),
                fps_disagreement=abs(measured - fps) / fps > 0.05,
            )
            rows.append(row)
            write_json(destination, rows)
            print(
                f"Audit {len(rows)}: {path.name} | {fps:g} FPS | {row['duration']:.2f}s",
                flush=True,
            )
    return rows


def tiny_learnability(prepared, destination, deadline):
    """Memorize fixed, balanced training clips; no validation or test selection."""
    seed_everything(settings.SEED)
    dataset = prepared["dataset"]
    dataset.training_windows = False
    indices = balanced_tiny_indices(
        dataset, list(prepared["train"].indices), settings.TINY_PER_CLASS
    )
    check_deadline(deadline)
    clips, labels = zip(*(dataset[i] for i in indices))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.stack(clips).to(device)
    targets = torch.tensor(labels, device=device)
    model = CustomConvLSTM(
        dataset.num_classes, [(16, (3, 3)), (32, (3, 3))], dropout=0
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    initial_loss = None
    try:
        for step in range(1, settings.TINY_STEPS + 1):
            check_deadline(deadline)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if initial_loss is None:
                initial_loss = float(loss.item())
            loss.backward()
            gradient = float(
                torch.stack(
                    [p.grad.norm() for p in model.parameters() if p.grad is not None]
                )
                .norm()
                .item()
            )
            if not np.isfinite(gradient) or gradient == 0:
                raise ValueError("Tiny check found zero/non-finite gradients")
            parameter = next(model.parameters())
            before = parameter.detach().clone()
            optimizer.step()
            update = float((parameter.detach() - before).norm().item())
            if step % 16 == 0 or step == settings.TINY_STEPS:
                model.eval()
                with torch.no_grad():
                    logits = model(inputs)
                    accuracy = float(
                        (logits.argmax(1) == targets).float().mean().item()
                    )
                    clean_loss = float(criterion(logits, targets).item())
                row = dict(
                    step=step,
                    loss=clean_loss,
                    accuracy=accuracy,
                    gradient_norm=gradient,
                    parameter_update_norm=update,
                    predictions=logits.argmax(1).tolist(),
                )
                history.append(row)
                passed = accuracy >= 0.95 and clean_loss < initial_loss and update > 0
                result = dict(
                    passed=passed,
                    partition="train-only",
                    indices=indices,
                    labels=list(labels),
                    shape=list(inputs.shape),
                    input_range=[float(inputs.min()), float(inputs.max())],
                    initial_loss=initial_loss,
                    history=history,
                    model=model.configuration(),
                    seed=settings.SEED,
                    learning_rate=0.001,
                    weight_decay=0,
                    scheduler="none",
                )
                write_json(destination, result)
                print(
                    f"Tiny check {step}/{settings.TINY_STEPS}: loss={clean_loss:.4f}, acc={accuracy:.1%}",
                    flush=True,
                )
                if passed:
                    return result
        raise ValueError(
            "Tiny training-only check did not reach 95%; inspect tiny.json before expensive screening"
        )
    finally:
        del model, optimizer, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
