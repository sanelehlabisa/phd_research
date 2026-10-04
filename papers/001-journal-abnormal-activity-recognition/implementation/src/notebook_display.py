"""Small, shared notebook displays; all inference runs in evaluation mode."""

import html
import random
from fractions import Fraction
from pathlib import Path

import av
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import pandas as pd
import torch
from IPython.display import HTML, Video, display

from .dataset import VideoAugmentation
from .utils import write_json, write_video_torchvision


def video_card(path, title, correct=None):
    color = "#26734d" if correct else "#bd3434"
    if correct is None:
        color = "#567"
    video = Video(filename=str(path), embed=True, width=420)._repr_html_()
    display(
        HTML(
            f'<div style="border:4px solid {color};padding:8px;display:inline-block">'
            f"<p>{html.escape(title)}</p>{video}</div>"
        )
    )


def native_preview(source, destination, seconds):
    """Encode the matching first window at source resolution and native FPS."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as container, av.open(str(destination), "w") as output:
        source_stream = container.streams.video[0]
        rate = source_stream.average_rate
        if not rate or rate <= 0:
            raise ValueError(f"Native frame rate unavailable: {source}")
        stream = output.add_stream("libx264", rate=rate)
        stream.width = source_stream.width
        stream.height = source_stream.height
        stream.pix_fmt = "yuv420p"
        for index, frame in enumerate(container.decode(source_stream)):
            if index / float(rate) >= seconds:
                break
            # Decoded frames retain the input container's time base. Reset both
            # fields together so the encoder keeps the declared native FPS.
            frame.pts = index
            frame.time_base = Fraction(rate.denominator, rate.numerator)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return float(rate)


def show_dataset(prepared):
    dataset = prepared["dataset"]
    display(pd.DataFrame(prepared["split"]["class_counts"]).T)
    print(
        "Clip-level split; source/subject independence is not established. Test remains locked."
    )
    index = random.Random(42).choice(prepared["train"].indices)
    clip, label = dataset[index]
    source = dataset.samples[index][0]
    output = prepared["root"] / "runs" / "previews" / prepared["specification"].key
    native = output / "native.mp4"
    fps = native_preview(source, native, dataset.sequence_length / dataset.target_fps)
    sampled, augmented = output / "sampled.mp4", output / "augmented.mp4"
    with torch.random.fork_rng():
        torch.manual_seed(43)
        augmented_clip = VideoAugmentation()(clip.clone())
    write_video_torchvision(clip, sampled, dataset.target_fps)
    write_video_torchvision(augmented_clip, augmented, dataset.target_fps)
    record = dict(
        source=str(source),
        partition="train",
        label=dataset.class_names[label],
        native_fps=fps,
        target_fps=dataset.target_fps,
        shape=list(clip.shape),
        native=str(native),
        sampled=str(sampled),
        augmented=str(augmented),
    )
    write_json(output / "preview.json", record)
    print(record)
    video_card(native, f"Native window | {fps:g} FPS | {source.name}")
    video_card(
        sampled,
        f"Same clip, model input | {dataset.target_fps} FPS | {dataset.frame_size}",
    )
    video_card(augmented, "Same model input + one clip-consistent augmentation")
    return record


class LiveCurves:
    """Update one display per epoch, preserving progress logs and saving a PNG."""

    def __init__(self):
        self.handle = None

    def __call__(self, history, run_dir):
        figure, axes = plt.subplots(1, 2, figsize=(10, 3))
        epochs = [row["epoch"] for row in history]
        for axis, metric in zip(axes, ("loss", "accuracy")):
            for partition in ("train", "validation"):
                axis.plot(
                    epochs, [row[partition][metric] for row in history], label=partition
                )
            axis.set(xlabel="Epoch", ylabel=metric, title=metric.title())
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            if metric == "accuracy":
                axis.set_ylim(0, 1)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.legend()
        figure.tight_layout()
        figure.savefig(Path(run_dir) / "learning_curves.png")
        if self.handle is None:
            self.handle = display(figure, display_id=True)
        else:
            self.handle.update(figure)
        plt.close(figure)


@torch.inference_mode()
def prediction_examples(model, prepared, partition, output, count=5):
    """Save and show the exact input clip, probabilities, label and correctness."""
    model.eval()
    dataset = prepared["dataset"]
    device = next(model.parameters()).device
    indices = list(prepared[partition].indices)
    indices = random.Random(43).sample(indices, min(count, len(indices)))
    records = []
    for index in indices:
        clip, label = dataset[index]  # dataset enforces the partition lock
        probabilities = model(clip.unsqueeze(0).to(device)).softmax(1)[0].cpu()
        predicted = int(probabilities.argmax())
        destination = Path(output) / f"{partition}_{index}.mp4"
        write_video_torchvision(clip, destination, dataset.target_fps)
        records.append(
            dict(
                path=str(destination),
                source=str(dataset.samples[index][0]),
                partition=partition,
                expected=dataset.class_names[label],
                predicted=dataset.class_names[predicted],
                correct=predicted == label,
                probabilities=probabilities.tolist(),
                classes=dataset.class_names,
            )
        )
    write_json(Path(output) / "predictions.json", records)
    show_predictions(records)
    return records


def show_predictions(records):
    """Display saved examples without reopening raw test data."""
    for record in records:
        status = "CORRECT" if record["correct"] else "WRONG"
        video_card(
            record["path"],
            f"{record['partition']} | {status} | true: {record['expected']} | "
            f"predicted: {record['predicted']}",
            record["correct"],
        )
        figure, axis = plt.subplots(figsize=(6, max(2, len(record["classes"]) * 0.25)))
        axis.barh(record["classes"], record["probabilities"])
        axis.set(xlim=(0, 1), xlabel="Softmax probability")
        figure.tight_layout()
        display(figure)
        plt.close(figure)
