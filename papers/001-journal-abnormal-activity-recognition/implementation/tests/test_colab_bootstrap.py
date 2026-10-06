"""Tests for the import-safe modular notebook bootstrap."""

import json
import subprocess
from pathlib import Path

import pytest

from notebook_files import NOTEBOOKS
from src import colab_bootstrap
from src.colab_bootstrap import exact_requirements, version_matches


def test_exact_requirements_parses_direct_pins(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.4.3\n# comment\ntorch==2.9.1\n", encoding="utf-8")
    assert exact_requirements(requirements) == {"numpy": "2.4.3", "torch": "2.9.1"}


def test_exact_requirements_rejects_ranges(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact pin"):
        exact_requirements(requirements)


@pytest.mark.parametrize(
    "installed,required,matches",
    [
        ("2.9.1+cu128", "2.9.1", True),
        ("2.9.1+cpu", "2.9.1", True),
        ("2.9.0+cu128", "2.9.1", False),
        (None, "2.9.1", False),
        ("2.9.1+cpu", "2.9.1+cu128", False),
    ],
)
def test_pins_accept_build_suffixes(installed, required, matches):
    assert version_matches(installed, required) is matches


@pytest.mark.parametrize("cpu_wheel", [False, True])
def test_gpu_bootstrap_preserves_or_repairs_cuda_pair(tmp_path, monkeypatch, cpu_wheel):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.9.1\ntorchvision==0.24.1\n")
    suffix = "+cpu" if cpu_wheel else "+cu128"
    before = {"torch": "2.9.1" + suffix, "torchvision": "0.24.1" + suffix}
    after = {"torch": "2.9.1+cu128", "torchvision": "0.24.1+cu128"}
    versions = iter([before, after])
    monkeypatch.setattr(
        colab_bootstrap, "installed_versions", lambda pins: next(versions)
    )
    monkeypatch.setattr(colab_bootstrap, "gpu_is_visible", lambda: True)
    monkeypatch.setattr(
        colab_bootstrap,
        "import_health_check",
        lambda *args: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="null" if cpu_wheel else '"12.8"', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    if cpu_wheel:
        with pytest.raises(SystemExit) as error:
            colab_bootstrap.main([str(requirements)])
        assert error.value.code == 75
        assert "torch==2.9.1+cu128" in commands[1]
        assert "torchvision==0.24.1+cu128" in commands[1]
        assert "https://download.pytorch.org/whl/cu128" in commands[1]
    else:
        colab_bootstrap.main([str(requirements)])
        assert len(commands) == 1  # CUDA probe only; healthy runtime skips pip


def test_matching_cpu_runtime_skips_requirement_install(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.4.3\n")
    monkeypatch.setattr(
        colab_bootstrap, "installed_versions", lambda pins: {"numpy": "2.4.3"}
    )
    monkeypatch.setattr(colab_bootstrap, "gpu_is_visible", lambda: False)
    monkeypatch.setattr(
        colab_bootstrap,
        "import_health_check",
        lambda *args: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    )
    commands = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    colab_bootstrap.main([str(requirements)])

    assert commands == []


def test_matching_but_broken_numpy_is_repaired_before_imports(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.4.3\n")
    monkeypatch.setattr(
        colab_bootstrap, "installed_versions", lambda pins: {"numpy": "2.4.3"}
    )
    monkeypatch.setattr(colab_bootstrap, "gpu_is_visible", lambda: False)
    checks = iter(
        [
            subprocess.CompletedProcess([], 1, stderr="AttributeError: _blas_supports_fpe"),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
    )
    monkeypatch.setattr(
        colab_bootstrap, "import_health_check", lambda *args: next(checks)
    )
    repaired = []
    monkeypatch.setattr(colab_bootstrap, "repair_numpy", repaired.append)

    with pytest.raises(SystemExit) as error:
        colab_bootstrap.main([str(requirements)])

    assert error.value.code == 75
    assert repaired == ["numpy==2.4.3"]


def test_unrepairable_import_failure_has_short_actionable_message(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.4.3\n")
    monkeypatch.setattr(
        colab_bootstrap, "installed_versions", lambda pins: {"numpy": "2.4.3"}
    )
    monkeypatch.setattr(colab_bootstrap, "gpu_is_visible", lambda: False)
    broken = subprocess.CompletedProcess(
        [], 1, stderr="long traceback\nAttributeError: _blas_supports_fpe"
    )
    monkeypatch.setattr(
        colab_bootstrap, "import_health_check", lambda *args: broken
    )
    monkeypatch.setattr(colab_bootstrap, "repair_numpy", lambda requirement: None)

    with pytest.raises(SystemExit, match="fresh Colab A100 runtime") as error:
        colab_bootstrap.main([str(requirements)])

    assert "AttributeError: _blas_supports_fpe" in str(error.value)
    assert "long traceback" not in str(error.value)


IMPLEMENTATION = Path("papers/001-journal-abnormal-activity-recognition/implementation")
HELPERS = (
    "notebook_data.py",
    "notebook_display.py",
    "notebook_workflows.py",
    "notebook_suite.py",
    "notebook_models.py",
    "notebook_diagnostics.py",
    "kinetics600_subset.py",
)


def checkout_setup(notebook, checkout):
    """Run the real notebook's checkout checks without installing or importing ML."""
    content = json.loads(notebook.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in content["cells"]
        if cell["cell_type"] == "code"
    )
    source = source.split("bootstrap = subprocess.run(", 1)[0]
    source = source.replace(
        "REPO_ROOT = Path('/content/phd_research')",
        f"REPO_ROOT = Path({str(checkout)!r})",
    )
    exec(compile(source, "<notebook setup>", "exec"), {})


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize("present", [False, True])
def test_checkout_checks_helpers_before_installation(
    tmp_path, monkeypatch, notebook, present
):
    (tmp_path / ".git").mkdir()
    if present:
        source = tmp_path / IMPLEMENTATION / "src"
        source.mkdir(parents=True)
        for name in HELPERS:
            (source / name).touch()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="test-revision\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    if present:
        checkout_setup(notebook, tmp_path)
    else:
        with pytest.raises(RuntimeError, match="NOT pip packages"):
            checkout_setup(notebook, tmp_path)
    assert commands[0] == ["git", "pull", "--ff-only", "origin", "master"]
    assert all(command[0] in ("git", "nvidia-smi") for command in commands)


def test_checkout_clones_into_empty_folder(tmp_path, monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "clone":
            source = tmp_path / IMPLEMENTATION / "src"
            source.mkdir(parents=True)
            for name in HELPERS:
                (source / name).touch()
        return subprocess.CompletedProcess(
            command, 0, stdout="test-revision", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    checkout_setup(NOTEBOOKS[-1], tmp_path)
    assert commands[0][:2] == ["git", "clone"]


def test_checkout_preserves_non_git_files(tmp_path):
    existing = tmp_path / "keep.txt"
    existing.write_text("my data")
    with pytest.raises(RuntimeError, match="not a Git checkout"):
        checkout_setup(NOTEBOOKS[-1], tmp_path)
    assert existing.read_text() == "my data"


@pytest.mark.parametrize("conflict", [False, True])
def test_real_fast_forward_preserves_local_edits(tmp_path, conflict):
    """A local Git remote reproduces Colab updates, with no network or credentials."""
    remote = tmp_path / "remote"
    remote.mkdir()

    def git(directory, *arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )

    git(remote, "init", "-b", "master")
    git(remote, "config", "user.name", "Notebook Test")
    git(remote, "config", "user.email", "notebook-test@example.invalid")
    (remote / "local-settings.txt").write_text("original\n")
    git(remote, "add", "local-settings.txt")
    git(remote, "commit", "-m", "initial")
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", str(remote), str(checkout))
    (checkout / "local-settings.txt").write_text("preserve my Colab edits\n")
    source = remote / IMPLEMENTATION / "src"
    source.mkdir(parents=True)
    for name in HELPERS:
        (source / name).write_text("# test checkout fixture\n")
    if conflict:
        (remote / "local-settings.txt").write_text("conflicting remote change\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "publish notebook helpers")
    if conflict:
        with pytest.raises(RuntimeError, match="Cannot safely update"):
            checkout_setup(NOTEBOOKS[-1], checkout)
    else:
        checkout_setup(NOTEBOOKS[-1], checkout)
        assert all(
            (checkout / IMPLEMENTATION / "src" / name).is_file() for name in HELPERS
        )
    assert (checkout / "local-settings.txt").read_text() == "preserve my Colab edits\n"
