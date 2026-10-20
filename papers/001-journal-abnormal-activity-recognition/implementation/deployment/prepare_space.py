"""Build a small, upload-ready Spaces folder without publishing anything."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "__init__.py",
    "demo_inference.py",
    "model.py",
    "dataset.py",
    "dataset_source.py",
    "experiment_config.py",
    "utils.py",
    "temporal_sampling.py",
)
EXAMPLE_FILES = ("normal.mp4", "fight.mp4", "pollution.mp4", "README.md")


def sha256_file(path):
    """Hash in bounded chunks, including on Python 3.10."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_space(root=ROOT):
    root = Path(root)
    template = root / "deployment/huggingface_space"
    assets = root / "demo_assets/winner"
    checkpoint = assets / "best_model.pth"
    metadata = assets / "config.json"
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing demo checkpoint. Run git lfs pull first.")
    with checkpoint.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Checkpoint is an LFS pointer. Run git lfs pull first.")
    digest = sha256_file(checkpoint)
    details = json.loads(metadata.read_text(encoding="utf-8"))
    if details.get("checkpoint_sha256") != digest:
        raise ValueError("Checkpoint hash does not match the demo metadata.")
    required = [
        root / "demo.py",
        *(root / "demo_assets/examples" / name for name in EXAMPLE_FILES),
        *(root / "src" / name for name in SOURCE_FILES),
        *(
            template / name
            for name in ("app.py", "README.md", "requirements.txt", ".gitattributes")
        ),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required deployment input missing: {path}")
    exports = root / "runs/spaces_upload"
    exports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_")
    destination = Path(tempfile.mkdtemp(prefix=stamp, dir=exports))
    for name in ("app.py", "README.md", "requirements.txt", ".gitattributes"):
        shutil.copy2(template / name, destination / name)
    shutil.copy2(root / "demo.py", destination / "demo.py")
    # Copy code only; never traverse datasets, historical runs, .git or caches.
    for name in SOURCE_FILES:
        source = root / "src" / name
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    target_assets = destination / "demo_assets/winner"
    target_assets.mkdir(parents=True)
    shutil.copy2(checkpoint, target_assets / checkpoint.name)
    shutil.copy2(metadata, target_assets / metadata.name)
    target_examples = destination / "demo_assets/examples"
    target_examples.mkdir()
    for name in EXAMPLE_FILES:
        source = root / "demo_assets/examples" / name
        target = target_examples / name
        shutil.copy2(source, target)
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Exported example hash mismatch: {name}")
    if sha256_file(target_assets / checkpoint.name) != digest:
        raise RuntimeError("Exported checkpoint hash mismatch.")
    archive = shutil.make_archive(str(destination), "zip", root_dir=destination)
    print(f"Upload the CONTENTS of this folder to the Space root:\n{destination}")
    print(f"ZIP (extract before uploading):\n{archive}")
    print("Source + weights + three examples. Nothing has been uploaded or published.")
    return destination


if __name__ == "__main__":
    prepare_space()
