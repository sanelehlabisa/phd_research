# In[ ]:
from pathlib import Path
import importlib
import os
import subprocess
import sys


def run_frame_depth_workflow():
    """Run 128 fast search jobs, five resolution checks, then ten comparisons/tests."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    repo = Path("/content/phd_research")
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").is_dir():
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(
                f"{repo} is not an empty folder or Git checkout; existing files preserved."
            )
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/sanelehlabisa/phd_research.git",
                str(repo),
            ],
            check=True,
        )
    else:
        update = subprocess.run(
            ["git", "pull", "--ff-only", "origin", "master"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        print(update.stdout, end="", flush=True)
        if update.returncode:
            raise RuntimeError(
                "Could not safely update checkout; resolve this message without discarding evidence:\n"
                + update.stderr
            )
    root = repo / "papers/001-journal-abnormal-activity-recognition/implementation"
    required = [
        "src/focused_protocol.py",
        "src/capacity_search.py",
        "src/study_matrix.py",
        "src/campaign_checks.py",
        "notebooks/utils/diagnostics.py",
        "notebooks/utils/aad_study.py",
        "notebooks/utils/study_archive.py",
        "notebooks/utils/colab_bootstrap.py",
        "configs/experiments/aad_fast_frame_depth_search_colab.json",
        "configs/experiments/aad_frame_depth_comparison_colab.json",
    ]
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise RuntimeError(
            f"Checkout is missing focused workflow files: {missing}. Publish the complete verified update before running this export."
        )
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(
        f"Code checkout: {repo} @ {revision}\nNotebook Python: {sys.executable}",
        flush=True,
    )
    bootstrap = subprocess.run(
        [
            sys.executable,
            str(root / "notebooks/utils/colab_bootstrap.py"),
            str(root / "requirements.txt"),
        ]
    )
    if bootstrap.returncode == 75:
        print(
            "Pinned packages were repaired. Restart the Colab runtime once, reconnect the GPU, then rerun this cell.",
            flush=True,
        )
        return None
    if bootstrap.returncode:
        raise RuntimeError("Colab setup failed; see the bootstrap output above.")
    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if (
            name == "src"
            or name.startswith("src.")
            or name == "notebooks"
            or name.startswith("notebooks.")
        ):
            del sys.modules[name]
    importlib.invalidate_caches()
    from notebooks.utils.helpers import validate_runtime
    from notebooks.utils.aad_study import run_frame_depth_aad_study

    print(validate_runtime(require_cuda=True), flush=True)
    print(
        "143 trainings: 128 search (32px/128 epochs), five checks (64px/128 epochs), ten comparisons (64px/256 epochs). Authorize Drive and verify the printed test file before leaving; small results-only backups are automatic. Keep checkpoints separately.",
        flush=True,
    )
    return run_frame_depth_aad_study(root)


artifact_zip = run_frame_depth_workflow()
