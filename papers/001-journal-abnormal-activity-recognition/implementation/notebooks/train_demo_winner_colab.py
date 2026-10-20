# Paste this entire file into ONE Colab GPU cell and run.
# Authorize Google Drive when asked. This trains ONE model, not the search.
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid


CONTENT_ROOT = Path("/content")
REPO_ROOT = CONTENT_ROOT / "phd_research"
DRIVE_ROOT = CONTENT_ROOT / "drive"
REPO_URL = "https://github.com/sanelehlabisa/phd_research.git"
MODEL = "custom_convlstm_configured"
REQUIRED_FILES = (
    "src/experiments.py",
    "src/dataset_source.py",
    "notebooks/utils/colab_bootstrap.py",
    "notebooks/utils/helpers.py",
    "requirements.txt",
    "splits/abnormal-activities-dataset_seed42.json",
)


def winner_config(local_runs):
    """Embed the frozen recipe so no unpublished demo helper is needed."""
    return {
        "models": [MODEL],
        "dataset_name": "aad",  # Existing runner downloads public AAD automatically.
        "dataset_dir": "datasets/abnormal-activities-dataset",
        "runs_dir": str(local_runs),
        "split_manifest": "splits/abnormal-activities-dataset_seed42.json",
        "split_seed": 42,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42,
        "sequence_length": 8,
        "height": 64,
        "width": 64,
        "target_fps": 16,
        "sampling_version": "timestamps_v1",
        "convlstm_layers": [[64, [3, 3]], [64, [3, 3]], [128, [3, 3]]],
        "hidden_classifier_width": None,
        "epochs": 256,
        "early_stopping_patience": 32,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "augment": False,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "scheduler": "reduce_on_plateau",
        "cache_dataset": True,
        "num_workers": 0,
        "pin_memory": True,
        "prediction_samples": 5,
        "prediction_samples_per_category": 5,
    }


def prepare_storage():
    """Verify Drive before training; route only experiment outputs there."""
    from google.colab import drive

    drive.mount(str(DRIVE_ROOT))
    my_drive = DRIVE_ROOT / "MyDrive"
    if not my_drive.is_dir():
        raise RuntimeError("Drive is not mounted. Authorize access and rerun.")
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    destination = my_drive / "phd_demo" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    local_runs = CONTENT_ROOT / f"phd_demo_{run_id}"
    local_runs.mkdir(exist_ok=False)
    # The shared runner stores its decoded cache under runs_dir/cache.
    # A local symlink sends only runs_dir/experiments to Drive, never that cache.
    target = destination / "experiments"
    target.mkdir()
    link = local_runs / "experiments"
    link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target.resolve():
        raise RuntimeError("Could not route experiment outputs to Drive.")
    probe = link / "drive_test.txt"
    probe.write_text(f"Drive save verified for {run_id}\n", encoding="utf-8")
    if probe.read_text(encoding="utf-8") != f"Drive save verified for {run_id}\n":
        raise RuntimeError("Drive write/read check failed; training not started.")
    print(f"Drive write/read verified: {probe.resolve()}", flush=True)
    print("Dataset and decoded cache stay on the Colab VM.", flush=True)
    return local_runs, destination


def train_command(config_path):
    """Use the existing validation-only trainer, with no study/search/test stage."""
    return [
        sys.executable,
        "-u",
        "-m",
        "src.experiments",
        "--config",
        str(config_path),
        "--minimum-epochs",
        "128",
        "--run-label",
        "demo-selected-colab",
        "--matched-training-evaluation",
        "--per-class-reporting",
    ]


def save_bundle(run_dir, local_runs, destination, root):
    """Keep the checkpoint beside its class/config metadata for the local app."""
    model_dir = run_dir / "models" / MODEL
    checkpoint = model_dir / "checkpoints/best_model.pth"
    required = [
        checkpoint,
        checkpoint.with_suffix(".json"),
        run_dir / "config.json",
        run_dir / "resolved_config.json",
        model_dir / "metrics/metrics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Training did not produce a complete demo bundle: {missing}"
        )
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(
            "Training is incomplete; retained files are not a final bundle."
        )
    shutil.copy2(
        root / "splits/abnormal-activities-dataset_seed42.json",
        run_dir / "split_manifest.json",
    )
    receipt = {
        "purpose": "Fresh demo retraining; not the archived paper checkpoint/score.",
        "test_access": "locked",
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metadata": "config.json",
        "git_revision": manifest.get("git_revision"),
        "local_launch_after_extracting_into_implementation_runs_demo": (
            f'python demo.py --checkpoint "runs/demo/{run_dir.name}/'
            f'models/{MODEL}/checkpoints/best_model.pth"'
        ),
    }
    (run_dir / "demo_bundle.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    # Only this one run: no dataset, cache, other studies or historical checkpoints.
    local_zip = Path(
        shutil.make_archive(
            str(local_runs / "demo_model"),
            "zip",
            root_dir=run_dir.parent,
            base_dir=run_dir.name,
        )
    )
    archive = destination / "demo_model.zip"
    shutil.copy2(local_zip, archive)
    if (
        hashlib.sha256(local_zip.read_bytes()).digest()
        != hashlib.sha256(archive.read_bytes()).digest()
    ):
        raise RuntimeError(
            "Drive ZIP verification failed; original run files are retained."
        )
    print(f"\nSaved checkpoint: {checkpoint.resolve()}", flush=True)
    print(
        f"Saved bundle: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)",
        flush=True,
    )
    print(
        receipt["local_launch_after_extracting_into_implementation_runs_demo"],
        flush=True,
    )
    return archive


def show_validation_preview(run_dir):
    """Display existing validation artifacts without importing ML into the kernel."""
    from IPython.display import Image, Video, display

    model_dir = run_dir / "models" / MODEL
    for name in ("training_curves.png", "validation_confusion_matrix.png"):
        path = model_dir / "metrics" / name
        if path.is_file():
            display(Image(filename=str(path)))
    examples = json.loads(
        (model_dir / "predictions/predictions.json").read_text(encoding="utf-8")
    )
    print(
        "Validation examples (not test). Sampled clips play at "
        f"{examples['playback_fps']} FPS; model input uses 16 FPS."
    )
    for row in examples["records"][:5]:
        print(
            f"True: {row['true']} | Predicted: {row['pred']} | Correct: {row['correct']} | Confidence: {row['confidence']:.1%}"
        )
        display(Video(row["path"], embed=True))


def train_demo_on_colab():
    """Mount/check Drive, install, train once on GPU, package and preview."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    local_runs, destination = prepare_storage()
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_ROOT / ".git").is_dir():
        if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
            raise RuntimeError(
                f"{REPO_ROOT} is not an empty Git checkout; files preserved."
            )
        subprocess.run(
            ["git", "clone", "--branch", "master", REPO_URL, str(REPO_ROOT)], check=True
        )
    else:
        subprocess.run(
            ["git", "pull", "--ff-only", "origin", "master"], cwd=REPO_ROOT, check=True
        )
    root = REPO_ROOT / "papers/001-journal-abnormal-activity-recognition/implementation"
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Checkout is missing required training files: {missing}")
    bootstrap = subprocess.run(
        [
            sys.executable,
            str(root / "notebooks/utils/colab_bootstrap.py"),
            str(root / "requirements.txt"),
        ],
        cwd=root,
    )
    if bootstrap.returncode not in (0, 75):
        raise RuntimeError(
            "Dependency setup failed; see output above. Training not started."
        )
    # 75 requests a restart for notebook imports. Here ALL ML work is in fresh
    # processes, so no stale torch/NumPy modules are reused by training.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from notebooks.utils.helpers import validate_runtime; "
            "print(validate_runtime(require_cuda=True), flush=True)",
        ],
        cwd=root,
        check=True,
    )
    config_path = destination / "training_config.json"
    config_path.write_text(
        json.dumps(winner_config(local_runs), indent=2) + "\n", encoding="utf-8"
    )
    print(
        "ONE fresh [64,64,128] model; 8 frames, 64x64, 16 FPS. "
        "128 minimum / 256 maximum epochs, validation-loss selection. "
        "Test stays locked. Rerunning starts a NEW training, not a resume.\n"
        f"Checkpoints/metrics are written to Drive during training: {destination}",
        flush=True,
    )
    subprocess.run(train_command(config_path), cwd=root, check=True)
    runs = sorted(
        path.parent for path in (destination / "experiments").glob("*/run.json")
    )
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly one training run, found {len(runs)}.")
    archive = save_bundle(runs[0], local_runs, destination, root)
    # Saving succeeds before optional browser/UI operations.
    try:
        show_validation_preview(runs[0])
    except Exception as error:
        print(
            f"Preview unavailable ({error}); all files are saved on Drive.", flush=True
        )
    try:
        from google.colab import files

        files.download(str(archive))
    except Exception as error:
        print(
            f"Browser download unavailable ({error}); get the ZIP from Drive.",
            flush=True,
        )
    return archive


if __name__ == "__main__":
    demo_archive = train_demo_on_colab()
