from pathlib import Path
import importlib
import os
import subprocess
import sys

# Set this to runs/studies/<completed-study> relative to IMPLEMENTATION_ROOT,
# or an absolute path in this Colab runtime. Never choose a historical screen.
STUDY_RUN_DIR = os.environ.get("AAD_STUDY_RUN_DIR", "")
CONFIRMATION_EPOCHS = 64
REPO_ROOT = Path("/content/phd_research")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if not STUDY_RUN_DIR.strip():
    raise ValueError("Set STUDY_RUN_DIR (or AAD_STUDY_RUN_DIR) to the completed 052/053 study.")
REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
if not (REPO_ROOT / ".git").exists():
    if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
        raise RuntimeError("Checkout path is non-empty; existing files were preserved.")
    subprocess.run(["git", "clone", "https://github.com/sanelehlabisa/phd_research.git", str(REPO_ROOT)], check=True)
else:
    update = subprocess.run(["git", "pull", "--ff-only", "origin", "master"], cwd=REPO_ROOT, capture_output=True, text=True)
    print(update.stdout, flush=True)
    if update.returncode:
        raise RuntimeError("Cannot safely update checkout; preserve/resolve local changes first.\n" + update.stderr)
IMPLEMENTATION_ROOT = REPO_ROOT / "papers/001-journal-abnormal-activity-recognition/implementation"
NOTEBOOK_UTILS = IMPLEMENTATION_ROOT / "notebooks/utils"
for relative in ("notebooks/utils/aad_final.py", "notebooks/utils/colab_bootstrap.py", "src/study_config.py", "src/evaluate.py"):
    if not (IMPLEMENTATION_ROOT / relative).is_file():
        raise RuntimeError(f"Checkout is missing {relative}; push the complete code update first.")
revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
print(f"Checkout: {REPO_ROOT} @ {revision}; Python: {sys.executable}", flush=True)
bootstrap = subprocess.run([sys.executable, str(NOTEBOOK_UTILS / "colab_bootstrap.py"), str(IMPLEMENTATION_ROOT / "requirements.txt")])
if bootstrap.returncode == 75:
    raise SystemExit("Restart the Colab kernel, reconnect to GPU and rerun from the top.")
bootstrap.check_returncode()
sys.path[:] = [str(IMPLEMENTATION_ROOT)] + [p for p in sys.path if p != str(IMPLEMENTATION_ROOT)]
for name in [n for n in sys.modules if n in ("src", "notebooks") or n.startswith(("src.", "notebooks."))]:
    del sys.modules[name]
importlib.invalidate_caches()
os.chdir(IMPLEMENTATION_ROOT)
from notebooks.utils.helpers import validate_runtime
print(validate_runtime(require_cuda=True))
from notebooks.utils.aad_final import selection_plan, train_and_freeze, evaluate_frozen, show_results
study_dir = Path(STUDY_RUN_DIR).expanduser().resolve()


import json
plan = selection_plan(study_dir, epochs=CONFIRMATION_EPOCHS)
print(json.dumps(plan, indent=2))


final_dir = train_and_freeze(study_dir, epochs=CONFIRMATION_EPOCHS)
show_results(final_dir)


report_path = evaluate_frozen(final_dir)


show_results(final_dir, report_path)
