"""Local Gradio demo using bundled weights; random UI tests require --random-weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The local demo does not need analytics or a public tunnel.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from src.demo_inference import (
    BUNDLED_CHECKPOINT,
    BUNDLED_METADATA,
    DemoPredictor,
    WINDOW_NOTE,
)

ROOT = Path(__file__).resolve().parent


def bundled_examples(root: Path = ROOT) -> list[str]:
    """Return up to three bundled MP4 files in a stable display order."""
    return [
        str(path.resolve())
        for path in sorted((root / "demo_assets/examples").glob("*.mp4"))
        if path.is_file()
    ][:3]


def validation_previews(root: Path = ROOT, limit: int = 3) -> list[str]:
    """Find local historical validation previews for random-weight UI tests only."""
    examples: list[str] = []
    manifests = sorted(
        root.glob("runs/experiments/*/models/*/predictions/predictions.json")
    )
    for path in manifests:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if metadata.get("partition") != "validation":
            continue
        for category in ("correct", "incorrect"):
            for video in sorted((path.parent / category).glob("*.mp4")):
                examples.append(str(video.resolve()))
                if len(examples) >= limit:
                    return examples
    return examples


def build_app(
    predictor: DemoPredictor, examples: list[str] | None = None, *, predict_fn=None
):
    """Build the UI; an optional callback can allocate a GPU per prediction."""
    import gradio as gr

    run_prediction = predictor.predict if predict_fn is None else predict_fn

    def predict(video):
        if not video:
            raise gr.Error("Upload a video or select an example first.")
        try:
            result = run_prediction(video)
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(str(error)) from error
        return result["probabilities"], result["sampling"], result["status"]

    with gr.Blocks(
        title="Video Activity Recognition",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        gr.Markdown("# 🎬 Video Activity Recognition\nBy Sanele Hlabisa")
        gr.Markdown("Upload a video, then click **Predict**.")
        if predictor.mode == "random":
            gr.Markdown("⚠️ UNTRAINED — UI test only. Predictions are random.")
        with gr.Row():
            video = gr.Video(label="📹 Video", sources=["upload"])
            probabilities = gr.Label(
                label="📊 Predictions",
                num_top_classes=len(predictor.class_names),
            )
        button = gr.Button("🔍 Predict", variant="primary")
        if examples:
            gr.Examples(
                examples=[[p] for p in examples],
                inputs=[video],
                cache_examples=False,
                label="🎞️ Try an example",
            )
        with gr.Accordion("ℹ️ Model details", open=False):
            gr.Markdown(
                "ConvLSTM [64, 64, 128] · 1,336,459 parameters  \n"
                "8 RGB frames · 16 FPS · 64 × 64 pixels  \n" + WINDOW_NOTE
            )
            status = gr.Textbox(
                label="Checkpoint", value=predictor.status, interactive=False
            )
            sampling = gr.JSON(label="Sampling details")
            if examples:
                gr.Markdown(
                    "Examples are illustrations, not an accuracy evaluation. "
                    "Use original videos; historical previews may already be processed."
                )
        button.click(
            predict,
            inputs=[video],
            outputs=[probabilities, sampling, status],
            api_name="predict",
            concurrency_limit=1,
        )
        gr.Markdown(
            "First 0.44 seconds only · Research demo · Scores are not calibrated."
        )
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--random-weights",
        action="store_true",
        help="Explicit untrained UI test; probabilities are not meaningful results.",
    )
    mode.add_argument(
        "--checkpoint",
        type=Path,
        help="Compatible winner checkpoint; defaults to bundled Git LFS weights.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Matching run config.json; bundled by default, otherwise inferred.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--example",
        type=Path,
        action="append",
        default=[],
        help="Local video; repeatable.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.random_weights and args.metadata:
        parser.error("--metadata applies only to a trained checkpoint")
    if args.checkpoint is None and not args.random_weights:
        args.checkpoint = BUNDLED_CHECKPOINT
        if args.metadata is None:
            args.metadata = BUNDLED_METADATA
    return args


def main(argv=None):
    args = parse_args(argv)
    predictor = DemoPredictor(
        checkpoint=args.checkpoint,
        metadata=args.metadata,
        random_weights=args.random_weights,
        device=args.device,
    )
    examples = []
    for path in args.example:
        if not path.is_file():
            raise SystemExit(f"Example video not found: {path}")
        examples.append(str(path.resolve()))
    if not examples:
        examples = validation_previews() if args.random_weights else bundled_examples()
    print(predictor.status, flush=True)
    print(WINDOW_NOTE, flush=True)
    app = build_app(predictor, examples)
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        max_file_size="200mb",
        allowed_paths=examples,
    )


if __name__ == "__main__":
    main()
