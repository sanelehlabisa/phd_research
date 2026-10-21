# Paste this entire file into ONE Colab GPU cell and run; authorize Drive.
# Exploratory single-label Kinetics training with the selected architecture.
from pathlib import Path
import json
import runpy
import subprocess
import sys


EPOCHS = 32
MINIMUM_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 8
REPO_ROOT = Path("/content/phd_research")
REPO_URL = "https://github.com/sanelehlabisa/phd_research.git"
HANDLE = "sanelehlabisa/kinetics-400-dataset"
SHARED_RUNNER = (
    "papers/001-journal-abnormal-activity-recognition/implementation/"
    "notebooks/scripts/09_train_demo_winner_colab.py"
)
PURPOSE = (
    "Exploratory single-label Kinetics task using the selected [64,64,128] "
    "architecture; not paper evidence, multi-label recognition, or proof of "
    "out-of-distribution AAD generalization. Separate weights and label head."
)

# Execute dataset resolution in a fresh interpreter after dependency setup.
# Keeping this source here also supports pasting the entire export into one cell.
DATASET_SETUP_CODE = r'''
from pathlib import Path
import json
import sys

import kagglehub
from src.dataset_source import resolve_downloaded_dataset_root, VIDEO_EXTENSIONS


def video_count(directory):
    return sum(
        path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        for path in directory.iterdir()
    )


def find_class_root(download_root):
    pending = [Path(download_root).resolve()]
    while pending:
        candidate = pending.pop(0)
        directories = sorted(path for path in candidate.iterdir() if path.is_dir())
        # Actual class labels such as "train" or "val" remain class labels.
        if sum(video_count(path) > 0 for path in directories) >= 2:
            split_names = {"train", "training", "val", "validation", "test", "testing"}
            if (
                all(path.name.lower() in split_names for path in directories)
                and any(path.suffix.lower() == ".csv" for path in candidate.iterdir())
            ):
                raise RuntimeError(
                    "Flat video/CSV split folders need an explicit label adapter; "
                    "train/test folder names are not activity labels."
                )
            return resolve_downloaded_dataset_root(candidate), None
        # Prefer upstream training data; never combine official split folders.
        for split_name in ("train", "training"):
            split_root = candidate / split_name
            if split_root.is_dir() and not video_count(split_root):
                try:
                    return resolve_downloaded_dataset_root(split_root), split_name
                except FileNotFoundError:
                    pass
        held_out = []
        for split_name in ("val", "validation", "test", "testing"):
            split_root = candidate / split_name
            if split_root.is_dir() and not video_count(split_root):
                try:
                    resolve_downloaded_dataset_root(split_root)
                    held_out.append(split_name)
                except FileNotFoundError:
                    pass
        if held_out:
            raise RuntimeError(
                "Official held-out split folders found without a usable train/training "
                "class-folder root; refusing to train on validation/test data."
            )
        pending.extend(path for path in directories if not video_count(path))
    raise RuntimeError(
        "Unsupported Kinetics layout: expected at least two class folders with "
        "videos directly inside each. Flat video/CSV layouts need an explicit "
        "label adapter; no classes or clips were silently selected."
    )


def download_dataset(handle, output_path):
    print(f"Downloading public Kaggle dataset: {handle}", flush=True)
    download_root = Path(kagglehub.dataset_download(handle))
    class_root, source_split = find_class_root(download_root)
    directories = sorted(path for path in class_root.iterdir() if path.is_dir())
    counts = {path.name: video_count(path) for path in directories}
    if len(counts) < 2 or any(count == 0 for count in counts.values()):
        raise RuntimeError(
            f"Unsupported Kinetics class layout at {class_root}: every class folder "
            "must contain supported videos directly. No folders were silently dropped."
        )
    inventory = {
        "source_handle": handle,
        "download_root": str(download_root.resolve()),
        "dataset_root": str(class_root),
        "source_split": source_split,
        "class_names": list(counts),
        "class_clip_counts": counts,
        "total_clips": sum(counts.values()),
        "note": "Observed Kaggle contents; completeness of the official release is not assumed.",
    }
    Path(output_path).write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Actual dataset root: {class_root}", flush=True)
    print(f"Actual classes: {len(counts)} | clips: {inventory['total_clips']}", flush=True)
    print(f"Class clip counts: {json.dumps(counts, sort_keys=True)}", flush=True)
    if source_split:
        print(f"Using upstream {source_split} only; generating a new 70/15/15 clip split.", flush=True)


if __name__ == "__main__":
    download_dataset(sys.argv[1], sys.argv[2])
'''


def kinetics_config(local_runs, winner_recipe):
    """Keep the architecture/input recipe; use a short, separate dataset task."""
    config = winner_recipe(local_runs)
    config.update({
        "dataset_name": "kinetics400-exploratory",
        "dataset_dir": "datasets/kinetics400",  # Replaced with the actual download root.
        "split_manifest": str(Path(local_runs) / "kinetics_split_seed42.json"),
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "cache_dataset": False,
        "num_workers": 2,
    })
    return config


def prepare_kinetics_dataset(root, destination, config):
    """Download/count in a fresh process; keep this run's split on Drive."""
    inventory_path = destination / "dataset_inventory.json"
    subprocess.run(
        [sys.executable, "-u", "-c", DATASET_SETUP_CODE, HANDLE, str(inventory_path)],
        cwd=root,
        check=True,
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    return {
        **config,
        "dataset_dir": inventory["dataset_root"],
        "split_manifest": str(destination / "kinetics_split_seed42.json"),
    }


def load_shared_runner():
    """Update the checkout and load its shared workflow without starting it."""
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_ROOT / ".git").is_dir():
        if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
            raise RuntimeError(f"{REPO_ROOT} is not an empty Git checkout; files preserved.")
        subprocess.run(
            ["git", "clone", "--branch", "master", REPO_URL, str(REPO_ROOT)], check=True
        )
    else:
        subprocess.run(
            ["git", "pull", "--ff-only", "origin", "master"], cwd=REPO_ROOT, check=True
        )
    path = REPO_ROOT / SHARED_RUNNER
    if not path.is_file():
        raise RuntimeError(
            "The shared 09 training helper is missing from the checkout. Publish "
            "the complete notebook/scripts update to master before using this "
            "one-cell Kinetics entry point in a fresh Colab runtime."
        )
    return runpy.run_path(str(path), run_name="kinetics_shared_runner")


def train_kinetics_on_colab():
    """Run one fresh training and save its own checkpoint, labels and examples."""
    if not 1 <= MINIMUM_EPOCHS <= EPOCHS or EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("Use 1 <= MINIMUM_EPOCHS <= EPOCHS and positive patience.")
    print(PURPOSE, flush=True)
    print(
        "All downloaded classes are used. Timestamp auditing decodes training/validation "
        "clips before epoch 1; the first pass can take time. Decoded caching is disabled.",
        flush=True,
    )
    shared = load_shared_runner()
    return shared["train_demo_on_colab"](
        config_factory=lambda runs: kinetics_config(runs, shared["winner_config"]),
        dataset_setup=prepare_kinetics_dataset,
        minimum_epochs=MINIMUM_EPOCHS,
        run_label="demo-kinetics-exploratory",
        storage_name="phd_demo_kinetics",
        purpose=PURPOSE,
        app_launch=False,
        update_checkout=False,
    )


if __name__ == "__main__":
    kinetics_archive = train_kinetics_on_colab()
