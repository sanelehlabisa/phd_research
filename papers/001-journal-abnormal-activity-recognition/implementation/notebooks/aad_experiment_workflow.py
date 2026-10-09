# In[ ]:
from pathlib import Path
import importlib
import os
import subprocess
import sys

WORKFLOW_STAGE = "search"  # Choose: "search", "comparison", or "smoke".
SELECTED_CONFIG_PATH = None  # Required only when WORKFLOW_STAGE is "comparison".
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
    from notebooks.utils.aad_study import run_aad_study, run_saved_comparison
    if WORKFLOW_STAGE == "search":
        artifact_zip = run_aad_study(IMPLEMENTATION_ROOT)
    elif WORKFLOW_STAGE == "comparison":
        if not SELECTED_CONFIG_PATH:
            raise ValueError("Set SELECTED_CONFIG_PATH to the saved validation-selected config before comparison.")
        artifact_zip = run_saved_comparison(IMPLEMENTATION_ROOT, SELECTED_CONFIG_PATH)
    elif WORKFLOW_STAGE == "smoke":
        artifact_zip = run_aad_study(IMPLEMENTATION_ROOT, run_full_study=False)
    else:
        raise ValueError('WORKFLOW_STAGE must be "search", "comparison", or "smoke".')
