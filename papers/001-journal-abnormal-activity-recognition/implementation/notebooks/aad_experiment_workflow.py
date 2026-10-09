# In[ ]:
from pathlib import Path
import importlib
import os
import subprocess
import sys

WORKFLOW_STAGE = "all"  # Default runs search -> comparison -> frozen test -> examples -> ZIP.
SELECTED_CONFIG_PATH = None  # Optional manual comparison only; Run All passes this internally.
RESOLVED_PROFILE_PATH = None  # Saved resolved_*.json; only for explicit resume.
ARTIFACT_ZIP_PATH = None  # Saved artifacts.zip; only for download retry.
SAVED_STUDY_DIRECTORY = None  # Saved runs/notebook_studies/<id>; repackage only.


def run_workflow():
    # Artifact retry never bootstraps, checks GPUs, downloads data, trains or tests.
    if WORKFLOW_STAGE in {"download", "repackage"}:
        root = Path("/content/phd_research/papers/001-journal-abnormal-activity-recognition/implementation")
        sys.path.insert(0, str(root))
        from notebooks.utils.study_archive import retry_download, retry_package
        if WORKFLOW_STAGE == "repackage":
            if not SAVED_STUDY_DIRECTORY:
                raise ValueError("Set SAVED_STUDY_DIRECTORY to the existing study directory.")
            return retry_package(root, SAVED_STUDY_DIRECTORY)
        if not ARTIFACT_ZIP_PATH:
            raise ValueError("Set ARTIFACT_ZIP_PATH to the existing ZIP.")
        return retry_download(ARTIFACT_ZIP_PATH)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    REPO_ROOT = Path("/content/phd_research")
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not (REPO_ROOT / ".git").is_dir():
        if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
            raise RuntimeError(f"{REPO_ROOT} is not an empty folder or Git checkout; existing files were preserved.")
        subprocess.run(["git", "clone", "https://github.com/sanelehlabisa/phd_research.git", str(REPO_ROOT)], check=True)
    else:
        update = subprocess.run(["git", "pull", "--ff-only", "origin", "master"], cwd=REPO_ROOT, capture_output=True, text=True)
        print(update.stdout, end="", flush=True)
        if update.returncode:
            raise RuntimeError("Could not safely update the checkout. Resolve the Git message above and rerun; no local files were discarded.\n" + update.stderr)
    IMPLEMENTATION_ROOT = REPO_ROOT / "papers/001-journal-abnormal-activity-recognition/implementation"
    UTILS = IMPLEMENTATION_ROOT / "notebooks/utils"
    REQUIRED = [
        UTILS / "colab_bootstrap.py",
        UTILS / "helpers.py",
        UTILS / "aad_study.py",
        UTILS / "study_archive.py",
        IMPLEMENTATION_ROOT / "src/study_matrix.py",
        IMPLEMENTATION_ROOT / "src/study_cache.py",
        IMPLEMENTATION_ROOT / "src/capacity_search.py",
        IMPLEMENTATION_ROOT / "src/capacity_config.py",
        IMPLEMENTATION_ROOT / "src/temporal_sampling.py",
        IMPLEMENTATION_ROOT / "src/study_resources.py",
        IMPLEMENTATION_ROOT / "src/study_reporting.py",
        IMPLEMENTATION_ROOT / "configs/experiments/aad_capacity_search_colab.json",
        IMPLEMENTATION_ROOT / "configs/experiments/aad_shape_search_colab.json",
        IMPLEMENTATION_ROOT / "configs/experiments/aad_final_comparison_colab.json",
        IMPLEMENTATION_ROOT / "configs/experiments/aad_source_review.json",
        IMPLEMENTATION_ROOT / "configs/experiments/aad_local_smoke.json",
        IMPLEMENTATION_ROOT / "configs/experiments/aad_custom_search_colab.json",
        IMPLEMENTATION_ROOT / "configs/experiments/aad_model_comparison_colab.json",
    ]
    MISSING = [str(path.relative_to(IMPLEMENTATION_ROOT)) for path in REQUIRED if not path.is_file()]
    if MISSING:
        raise FileNotFoundError(f"Checkout is missing required committed files: {MISSING}")
    revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    print(f"Code checkout: {REPO_ROOT} @ {revision}", flush=True)
    print(f"Notebook Python: {sys.executable}", flush=True)
    bootstrap = subprocess.run([sys.executable, str(UTILS / "colab_bootstrap.py"), str(IMPLEMENTATION_ROOT / "requirements.txt")])

    if bootstrap.returncode == 75:
        print("Pinned packages were repaired. Restart the Colab runtime, reconnect the GPU, then rerun this cell.")
    elif bootstrap.returncode:
        raise RuntimeError("Colab runtime setup failed; see the bootstrap output above.")
    else:
        sys.path.insert(0, str(IMPLEMENTATION_ROOT))
        for module_name in [name for name in sys.modules if name == "src" or name.startswith("src.") or name == "notebooks" or name.startswith("notebooks.")]:
            del sys.modules[module_name]
        importlib.invalidate_caches()
        from notebooks.utils.helpers import validate_runtime
        print(validate_runtime(require_cuda=True), flush=True)
        from notebooks.utils.aad_study import run_aad_study, run_saved_comparison, resume_saved_study
        from notebooks.utils.aad_study import run_capacity_study, run_full_aad_study
        if WORKFLOW_STAGE == "all":
            artifact_zip = run_full_aad_study(IMPLEMENTATION_ROOT)
        elif WORKFLOW_STAGE == "capacity_search":
            artifact_zip = run_capacity_study(IMPLEMENTATION_ROOT)
        elif WORKFLOW_STAGE == "search":
            artifact_zip = run_aad_study(IMPLEMENTATION_ROOT)
        elif WORKFLOW_STAGE == "comparison":
            if not SELECTED_CONFIG_PATH:
                raise ValueError("Set SELECTED_CONFIG_PATH to the saved validation-selected config before comparison.")
            artifact_zip = run_saved_comparison(IMPLEMENTATION_ROOT, SELECTED_CONFIG_PATH)
        elif WORKFLOW_STAGE == "resume":
            artifact_zip = resume_saved_study(IMPLEMENTATION_ROOT, RESOLVED_PROFILE_PATH)
        elif WORKFLOW_STAGE == "smoke":
            artifact_zip = run_aad_study(IMPLEMENTATION_ROOT, run_full_study=False)
            run_capacity_study(IMPLEMENTATION_ROOT, run_full_study=False, profile_name="aad_shape_search_colab.json")
        else:
            raise ValueError('Choose all, capacity_search, search, comparison, smoke, resume, download, or repackage.')
        return artifact_zip


artifact_zip = run_workflow()
