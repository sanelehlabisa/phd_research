"""Tests for the import-safe modular notebook bootstrap."""

import json
import subprocess
from pathlib import Path

import pytest

from src.colab_bootstrap import exact_requirements


def test_exact_requirements_parses_direct_pins(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.4.3\n# comment\ntorch==2.9.1\n", encoding="utf-8")
    assert exact_requirements(requirements) == {"numpy": "2.4.3", "torch": "2.9.1"}


def test_exact_requirements_rejects_ranges(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact pin"):
        exact_requirements(requirements)


NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("0*.ipynb"))
IMPLEMENTATION = Path("papers/001-journal-abnormal-activity-recognition/implementation")
HELPERS = (
    "notebook_data.py",
    "notebook_display.py",
    "notebook_workflows.py",
    "notebook_suite.py",
    "notebook_models.py",
    "notebook_diagnostics.py",
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
    exec(compile(source, str(notebook), "exec"), {})


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize("present", [False, True])
def test_checkout_checks_helpers_before_installation(
    tmp_path, monkeypatch, notebook, present
):
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
    assert all(command[0] == "git" for command in commands)


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
