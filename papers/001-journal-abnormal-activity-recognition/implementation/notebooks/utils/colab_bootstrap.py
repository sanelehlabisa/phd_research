"""Install exact requirements before notebook code imports binary packages."""

from __future__ import annotations

import importlib.metadata
import json
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


def version_matches(installed: str | None, required: str) -> bool:
    """Accept a pinned public version with its CUDA/CPU wheel suffix."""
    if installed is None:
        return False
    return installed == required or (
        "+" not in required and installed.split("+", 1)[0] == required
    )


def gpu_is_visible() -> bool:
    """Check whether this process can see an NVIDIA GPU before importing torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def import_health_check(implementation_root: Path) -> subprocess.CompletedProcess[str]:
    """Check notebook imports, including compiled NumPy, in a fresh process."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import numpy, numpy.testing; from notebooks.utils.helpers import validate_runtime",
        ],
        cwd=implementation_root,
        capture_output=True,
        text=True,
        timeout=120,
    )


def concise_error(result: subprocess.CompletedProcess[str]) -> str:
    """Return the final exception line from a subprocess without its full traceback."""
    lines = (result.stderr or result.stdout).strip().splitlines()
    return lines[-1] if lines else f"import check exited with code {result.returncode}"


def has_numpy_integrity_error(result: subprocess.CompletedProcess[str]) -> bool:
    """Identify a broken NumPy Python/native-extension pairing from its traceback."""
    details = f"{result.stdout}\n{result.stderr}"
    return any(
        marker in details
        for marker in (
            "_blas_supports_fpe",
            "numpy._core",
            "numpy.core.multiarray failed to import",
            "numpy.dtype size changed",
            "compiled using NumPy 1.x",
        )
    )


def repair_numpy(requirement: str) -> None:
    """Reinstall only the pinned NumPy wheel when its Python/C extension files disagree."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            requirement,
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Install requirements and stop when the current kernel must restart."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit("usage: colab_bootstrap.py REQUIREMENTS.txt")
    requirements = Path(arguments[0]).resolve()
    pins = exact_requirements(requirements)
    before = installed_versions(pins)
    gpu_visible = gpu_is_visible()
    needs_cuda_wheel = False
    if gpu_visible:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, torch; print(json.dumps(torch.version.cuda))",
            ],
            capture_output=True,
            text=True,
        )
        try:
            needs_cuda_wheel = probe.returncode != 0 or json.loads(probe.stdout) is None
        except json.JSONDecodeError:
            needs_cuda_wheel = True
        needs_cuda_wheel = needs_cuda_wheel or any(
            not version_matches(before[name], pins[name])
            or (before[name] or "").endswith("+cpu")
            for name in ("torch", "torchvision")
        )

    pins_match = all(
        version_matches(before[name], required) for name, required in pins.items()
    )
    changed = False
    if pins_match and not needs_cuda_wheel:
        health = import_health_check(requirements.parent)
        if health.returncode == 0:
            print("Pinned runtime and NumPy imports verified; skipped pip install.")
            return
        if not has_numpy_integrity_error(health):
            raise SystemExit(
                "Notebook dependency import check failed before experiment imports: "
                f"{concise_error(health)}. Check the pinned dependencies and reconnect "
                "to a fresh Colab runtime if needed."
            )
        repair_numpy(f"numpy=={pins['numpy']}")
        changed = True
        health = import_health_check(requirements.parent)
        if health.returncode != 0:
            raise SystemExit(
                "NumPy/dependency import check still fails after repairing the pinned "
                f"NumPy wheel: {concise_error(health)}. Reconnect to a fresh Colab "
                "A100 runtime and rerun setup."
            )
        raise SystemExit(75)

    if gpu_visible and needs_cuda_wheel:
        # Install the official CUDA pair rather than a CPU wheel on a GPU VM.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"torch=={pins['torch']}+cu128",
                f"torchvision=={pins['torchvision']}+cu128",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
    changed = True
    after = installed_versions(pins)
    mismatches = {
        package: {"required": required, "installed": after[package]}
        for package, required in pins.items()
        if not version_matches(after[package], required)
    }
    if mismatches:
        raise SystemExit(f"installed versions do not match exact pins: {mismatches}")
    version_changes = {
        package: {"before": before[package], "after": after[package]}
        for package in pins
        if before[package] != after[package]
    }
    if version_changes:
        changed = True
    health = import_health_check(requirements.parent)
    if health.returncode != 0:
        if has_numpy_integrity_error(health):
            repair_numpy(f"numpy=={pins['numpy']}")
            changed = True
            health = import_health_check(requirements.parent)
        if health.returncode != 0:
            raise SystemExit(
                "NumPy/dependency import check failed after package setup: "
                f"{concise_error(health)}. Reconnect to a fresh Colab A100 runtime "
                "and rerun setup."
            )
    if changed:
        raise SystemExit(75)
    print("Pinned runtime and NumPy imports verified.")


if __name__ == "__main__":
    main()
