"""ZeroGPU entry point; reuse the local demo and verified weights."""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# Spaces must patch CUDA before any model/PyTorch imports.
import spaces

from demo import build_app, bundled_examples
from src.demo_inference import BUNDLED_CHECKPOINT, BUNDLED_METADATA, DemoPredictor


# ZeroGPU emulates CUDA at startup; real GPU execution occurs inside predict.
predictor = DemoPredictor(
    checkpoint=BUNDLED_CHECKPOINT, metadata=BUNDLED_METADATA, device="cuda"
)


@spaces.GPU(duration=30)
def predict(video):
    return predictor.predict(video)


# A module-level Blocks instance is also discoverable by Spaces hot reload.
demo = build_app(predictor, examples=bundled_examples(), predict_fn=predict)


def main():
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        max_file_size="200mb",
    )


if __name__ == "__main__":
    main()
