# In[ ]:
from pathlib import Path
import importlib
import os
import subprocess
import sys

RUN_FULL_STUDY = False  # Set True only after the smoke check passes.
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
    raise SystemExit("Pinned packages were repaired. Restart this Colab runtime, reconnect the GPU, and rerun this cell.")
if bootstrap.returncode:
    raise RuntimeError("Colab runtime setup failed; see the bootstrap output above.")
sys.path.insert(0, str(IMPLEMENTATION_ROOT))
for module_name in [name for name in sys.modules if name == "src" or name.startswith("src.") or name == "notebooks" or name.startswith("notebooks.")]:
    del sys.modules[module_name]
importlib.invalidate_caches()
from notebooks.utils.helpers import validate_runtime
print(validate_runtime(require_cuda=True), flush=True)
from notebooks.utils.aad_study import run_aad_study
print("Colab can terminate this runtime externally; a forced stop may precede the final download.", flush=True)
artifact_zip = run_aad_study(IMPLEMENTATION_ROOT, run_full_study=RUN_FULL_STUDY)
if artifact_zip is None:
    print("Smoke check passed. Set RUN_FULL_STUDY = True to run the two AAD stages.")
