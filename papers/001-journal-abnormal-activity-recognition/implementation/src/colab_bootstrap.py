"""Install exact requirements before notebook code imports binary packages."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")


def exact_requirements(path: str | Path) -> dict[str, str]:
    """Parse the repository's direct, exactly pinned runtime requirements."""
    pins: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise ValueError(f"requirement must use an exact pin: {line}")
        pins[match.group(1)] = match.group(2)
    if not pins:
        raise ValueError("requirements file contains no exact pins")
    return pins


def installed_versions(packages: dict[str, str]) -> dict[str, str | None]:
    """Return installed versions without importing any target package."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv: list[str] | None = None) -> None:
    """Install requirements and stop when the current kernel must restart."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit("usage: colab_bootstrap.py REQUIREMENTS.txt")
    requirements = Path(arguments[0]).resolve()
    pins = exact_requirements(requirements)
    before = installed_versions(pins)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
    after = installed_versions(pins)
    mismatches = {
        package: {"required": required, "installed": after[package]}
        for package, required in pins.items()
        if after[package] != required
    }
    if mismatches:
        raise SystemExit(f"installed versions do not match exact pins: {mismatches}")
    changed = {
        package: {"before": before[package], "after": after[package]}
        for package in pins
        if before[package] != after[package]
    }
    if changed:
        print(
            "Pinned packages changed. Restart the Colab runtime, reconnect to the "
            "A100, and rerun this notebook from the top. Changes: " + str(changed),
            flush=True,
        )
        raise SystemExit(75)
    print("Pinned requirements already match this runtime.")


if __name__ == "__main__":
    main()
