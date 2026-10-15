"""Training-only learnability and train/validation source-video audits."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

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


def registered_learning_check(
    model_name: str,
    clips: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    steps: int = 64,
    batch_size: int = 2,
    learning_rate: float = 0.001,
    *,
    campaign_batch: bool = False,
) -> dict:
    """Trace a bounded transformer check through the existing modular trainer.

    Parameters:
        model_name: Swin registry name, using scratch weights.
        clips: Fixed training-only BTCHW clips in [0, 1].
        labels: Original dataset class indices, with at least two classes.
        num_classes: Full dataset head size, not the subset's class count.
        steps: Number of optimizer updates, limited to 128.
        batch_size: One or two clips for the local 6-GB diagnostic.
        learning_rate: Positive Adam rate for a separately labelled diagnostic.
        campaign_batch: Allow the frozen 089 batch (1-32), using full batches only.

    Returns:
        Recipe, input, learning, representative gradient and update records.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from src.experiments import build_registered_model
    from src.metrics import evaluate_classifier, train_classifier_steps

    if type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError("steps must be an integer from 1 to 128")
    allowed_batches = range(1, 33) if campaign_batch else (1, 2)
    if type(batch_size) is not int or batch_size not in allowed_batches:
        raise ValueError("batch_size must be 1/2 locally or 1-32 for campaign checks")
    if campaign_batch and (len(clips) < batch_size or len(clips) % batch_size):
        raise ValueError("Campaign check requires full batches at the frozen batch size")
    if type(learning_rate) not in (int, float) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if model_name not in ("swin3d_t", "swin3d_s"):
        raise ValueError("only the two registered Swin transformers are audited")
    if (clips.ndim != 5 or clips.shape[2] != 3 or not clips.numel()
            or not clips.is_floating_point() or not torch.isfinite(clips).all()
            or clips.min() < 0 or clips.max() > 1):
        raise ValueError("clips must be finite RGB BTCHW floats in [0, 1]")
    if (labels.dtype != torch.long or labels.ndim != 1
            or len(labels) != len(clips) or len(labels.unique()) < 2
            or labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels must contain valid dataset indices for two classes")
    determinism = seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(
        TensorDataset(clips.cpu(), labels.cpu()), batch_size=batch_size, shuffle=False
    )
    model = build_registered_model(
        model_name, num_classes, tuple(clips.shape[2:]), clips.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    parameters = dict(model.named_parameters())
    anchors = {
        suffix: next((name, value) for name, value in parameters.items()
                     if name.endswith(suffix))
        for suffix in ("patch_embed.proj.weight", "attn.qkv.weight", "head.weight")
    }
    optimizer_complete = {id(p) for p in model.parameters() if p.requires_grad} == {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    if not optimizer_complete:
        raise ValueError("optimizer omits trainable parameters")
    snapshots, gradient_updates, learning, backbone_input = {}, [], [], {}

    def observe_input(module, inputs):
        if not backbone_input:
            tensor = inputs[0]
            backbone_input.update(shape=list(tensor.shape), minimum=float(tensor.min()),
                                  maximum=float(tensor.max()))

    def clean_metrics(step):
        training = model.training
        try:
            metrics = evaluate_classifier(model, loader, criterion, device, num_classes)
            with torch.inference_mode():
                logits = torch.cat([model(x.to(device)).cpu() for x, _ in loader])
            learning.append(dict(step=step, **metrics,
                                 prediction_counts=dict(Counter(logits.argmax(1).tolist())),
                                 sample_logit_std=float(logits.std(dim=0).mean())))
            print(f"{model_name} step {step}/{steps}: train-only loss={metrics['loss']:.4f} "
                  f"acc={metrics['accuracy']:.1%}", flush=True)
        finally:
            model.train(training)

    def before_update(optimizer, args, kwargs):
        snapshots.update({key: parameter.detach().clone()
                          for key, (_, parameter) in anchors.items()})

    def after_update(optimizer, args, kwargs):
        step = len(gradient_updates) + 1
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        row = {"step": step, "training": model.training,
               "learning_rate": optimizer.param_groups[0]["lr"], "parameters": {},
               "all_gradients_present_and_finite": all(
                   gradient is not None and bool(torch.isfinite(gradient).all())
                   for gradient in gradients
               )}
        for key, (name, parameter) in anchors.items():
            gradient = parameter.grad
            row["parameters"][name] = {
                "gradient_finite": gradient is not None and bool(torch.isfinite(gradient).all()),
                "gradient_norm": None if gradient is None else float(gradient.norm()),
                "update_norm": float((parameter.detach() - snapshots[key]).norm()),
            }
        gradient_updates.append(row)
        if step % 8 == 0 or step == steps:
            clean_metrics(step)

    handles = [model.model.register_forward_pre_hook(observe_input),
               optimizer.register_step_pre_hook(before_update),
               optimizer.register_step_post_hook(after_update)]
    started = time.monotonic()
    try:
        clean_metrics(0)
        history = train_classifier_steps(model, loader, criterion, optimizer, device, steps)
        return {
            "model": model_name, "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device), "duration_seconds": time.monotonic() - started,
            "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "recipe": {"seed": 42, "steps": steps, "batch_size": batch_size,
                       "weights": None, "optimizer": "Adam", "learning_rate": learning_rate,
                       "weight_decay": 0, "augmentation": False, "scheduler": "none",
                       "determinism": determinism},
            "input_shape_btchw": list(clips.shape), "input_dtype": str(clips.dtype),
            "input_range": [float(clips.min()), float(clips.max())],
            "backbone_input_bcthw": backbone_input, "num_classes": num_classes,
            "labels": labels.tolist(), "optimizer_covers_all_parameters": optimizer_complete,
            "learning": learning, "step_metrics": history, "gradient_updates": gradient_updates,
            "limitations": "Train-only tiny check, not a full-recipe reproduction or held-out score; "
                            "fixed LR and repeated training clips; no plateau schedule or warm start.",
        }
    finally:
        for handle in handles:
            handle.remove()
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


def audit_transformer_learning(root: str | Path, steps: int = 64, batch_size: int = 2) -> dict:
    """Audit two transformers on four existing training clips without held-out access.

    Parameters:
        root: Implementation directory containing local AAD and its existing split.
        steps: Optimizer updates per model, limited to 128.
        batch_size: One or two clips per update.

    Returns:
        The saved diagnostic run and its complete trace; no model weights.
    """
    from src.dataset import AHARDataset, load_split_subsets
    from src.utils import RunContext

    if type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError("steps must be an integer from 1 to 128")
    if type(batch_size) is not int or batch_size not in (1, 2):
        raise ValueError("batch_size must be 1 or 2")
    root = Path(root).resolve()
    config_path = root / "configs/experiments/aad_shortlist_comparison_colab.json"
    config = json.loads(config_path.read_text())
    dataset_path = root / config["dataset"]["path"]
    split_path = root / "splits/abnormal-activities-dataset_seed42.json"
    if not dataset_path.is_dir() or not split_path.is_file():
        raise FileNotFoundError("Existing local AAD and seed-42 split are required; no download/resplit")
    dataset = AHARDataset(dataset_path, sequence_length=8, frame_size=(64, 64),
                          target_fps=16, sampling_version="timestamps_v1")
    train, _, _, split_metadata = load_split_subsets(dataset, split_path)
    classes = sorted({dataset.samples[index][1] for index in train.indices})[:2]
    if len(classes) != 2:
        raise ValueError("two training classes are required")
    grouped = [[index for index in sorted(train.indices, key=lambda i: str(dataset.samples[i][0]))
                if dataset.samples[index][1] == label][:2] for label in classes]
    if any(len(indices) != 2 for indices in grouped):
        raise ValueError("two training clips per class are required")
    indices = [grouped[class_index][offset] for offset in range(2) for class_index in range(2)]
    clips, labels = zip(*(dataset[index] for index in indices))
    sources = []
    for index in indices:
        path, label = dataset.samples[index]
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        sources.append(dict(index=index, path=str(path.relative_to(dataset_path)),
                            label=label, class_name=dataset.class_names[label],
                            partition="train", sha256=checksum,
                            sampling=dataset.sampling_plans[str(path.resolve())]))
    source_hashes = {}
    for relative in ("notebooks/utils/diagnostics.py", "src/experiments.py", "src/metrics.py",
                     "src/dataset.py", "src/temporal_sampling.py"):
        source_hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    run = RunContext(root / "runs", "experiments", dataset_path,
                     "transformer-train-only-audit", dict(steps=steps, batch_size=batch_size, seed=42),
                     dict(partition="train-only", split=split_metadata, source_sha256=source_hashes))
    result = dict(run_dir=str(run.run_dir), status="running", sources=sources,
                  class_names=dataset.class_names, split=split_metadata, source_sha256=source_hashes,
                  validation_clips_opened=0, test_clips_opened=0, models={})
    destination = run.run_dir / "diagnosis.json"
    write_json(destination, result)
    try:
        for name in ("swin3d_t", "swin3d_s"):
            result["models"][name] = registered_learning_check(
                name, torch.stack(clips), torch.tensor(labels), dataset.num_classes, steps, batch_size
            )
            write_json(destination, result)
        result["status"] = "complete"
        write_json(destination, result)
        run.complete({"diagnosis": "diagnosis.json"})
        return result
    except Exception as error:
        result.update(status="failed", error=str(error))
        write_json(destination, result)
        run.update(dict(status="failed", error=str(error)))
        raise
