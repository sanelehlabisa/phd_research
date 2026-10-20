---
title: Video Activity Recognition
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
python_version: "3.10"
app_file: app.py
---

# Video Activity Recognition

By Sanele Hlabisa.

Upload a video and click **Predict**. ConvLSTM [64,64,128] uses the initial
eight frames sampled at 16 FPS (0–0.4375 seconds), resized to 64×64.
Research demo; scores are not calibrated and predictions may be wrong.

Uses the October 20 Colab-retrained checkpoint, not the original paper weights.
Uploads are processed on the Hugging Face server; do not upload private footage.
Three validation examples are included in `demo_assets/examples/`; click one,
then **Predict**. They are illustrations, not an accuracy evaluation.
No full datasets or research reports are included.

Dependencies are installed from the single root `requirements.txt` during build.
Select **ZeroGPU** hardware in Space settings. Do not add a CPU wheel index or
`+cpu` package pins: inference uses `@spaces.GPU` with CUDA-enabled PyTorch.
For an older CPU deployment, replace all source/requirements files and factory
rebuild. Use the repository's `demo.py` for local CPU testing, not this `app.py`.
