"""Tests for the import-safe modular notebook bootstrap."""

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
