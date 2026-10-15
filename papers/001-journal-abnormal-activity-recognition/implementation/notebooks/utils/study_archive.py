"""ZIP-only evidence delivery; independent verification/retry never computes."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


def related_runs(root, directories):
    root = Path(root).resolve()
    found = {Path(p).resolve() for p in directories}
    pending = list(found)

    def links(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"run_dir", "source_group"} and isinstance(item, str):
                    yield item
                elif key in {"study_file", "selected_config"} and isinstance(item, str):
                    candidate = Path(item)
                    candidate = (
                        candidate if candidate.is_absolute() else root / candidate
                    ).resolve()
                    if candidate.is_relative_to(root / "runs"):
                        yield str(candidate.parent)
                elif key == "partial_runs" and isinstance(item, list):
                    yield from item
                else:
                    yield from links(item)
        elif isinstance(value, list):
            for item in value:
                yield from links(item)

    while pending:
        directory = pending.pop()
        for path in directory.rglob("*.json"):
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for value in links(values):
                target = Path(value)
                target = (target if target.is_absolute() else root / target).resolve()
                if not target.is_relative_to(root / "runs"):
                    raise ValueError(f"Evidence directory outside runs: {target}")
                if not target.is_dir():
                    raise FileNotFoundError(f"Referenced evidence missing: {target}")
                if target not in found:
                    found.add(target)
                    pending.append(target)
    return found


def _sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("archive_inventory.json"))
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("duplicate ZIP members")
        if set(archive.namelist()) != set(manifest) | {"archive_inventory.json"}:
            raise ValueError("ZIP inventory mismatch")
        for name, expected in manifest.items():
            if archive.getinfo(name).file_size != expected["bytes"]:
                raise ValueError(f"ZIP size mismatch: {name}")
            with archive.open(name) as stream:
                if _sha(stream) != expected["sha256"]:
                    raise ValueError(f"ZIP checksum mismatch: {name}")
    return manifest


def make_archive(
    root,
    archive_path,
    progress_path,
    run_directories,
    profiles,
    *,
    results_only=False,
    follow_links=True,
    receipts_dir=None,
):
    root = Path(root).resolve()
    receipts_dir = Path(receipts_dir or root / "runs/study_receipts").resolve()
    archive_path = Path(archive_path).resolve()
    partial = archive_path.with_suffix(".partial")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    initial = {*run_directories, Path(progress_path).parent}
    directories = related_runs(root, initial) if follow_links else initial
    files = {root / config for _, config in profiles}
    files.update(p for d in directories for p in d.rglob("*") if p.is_file())
    required_hashes = {}
    for directory in directories:
        for receipt in directory.rglob("receipt.json"):
            value = json.loads(receipt.read_text(encoding="utf-8"))
            if value.get("status") == "complete":
                required_hashes.update(value["files"])
        for receipt in (directory / "test_receipts").glob("*.json"):
            value = json.loads(receipt.read_text(encoding="utf-8"))
            if value.get("status") == "complete":
                required_hashes.update(value["files"])
    # Preserve restart guards when restoring the ZIP at the original Colab path.
    for directory in directories:
        lifecycle = directory / "run.json"
        claim_path = directory / "recovery_claim.json"
        if claim_path.is_file():
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            index = Path(claim["path"]).resolve()
            if not index.is_relative_to(receipts_dir) or not index.is_file():
                raise ValueError(
                    "Missing recovery test claim; do not discard restart guards"
                )
            with index.open("rb") as stream:
                if _sha(stream) != claim["sha256"]:
                    raise ValueError("Recovery test claim changed")
            files.add(index)
        if lifecycle.is_file():
            resume_index = json.loads(lifecycle.read_text(encoding="utf-8")).get(
                "resume_index"
            )
            if resume_index:
                index = Path(resume_index).resolve()
                if not index.is_relative_to(receipts_dir) or not index.is_file():
                    raise ValueError(f"Missing or invalid restart guard: {index}")
                files.add(index)
    files = sorted(
        p.resolve()
        for p in files
        if p.suffix not in {".zip", ".partial", ".tmp", ".lock"}
        and not {"cache", "caches", "datasets", "__pycache__"}.intersection(
            p.relative_to(root).parts
        )
    )
    if not all(p.is_file() and p.is_relative_to(root) for p in files):
        raise FileNotFoundError(
            "A required profile/artifact is missing or outside the implementation"
        )
    if not {Path(p).resolve() for p in required_hashes}.issubset(set(files)):
        raise ValueError(
            "Completed evidence is missing from ZIP inventory; restore the missing artifact before packaging."
        )
    omitted = {}
    if results_only:
        weights = {".pth", ".pt", ".ckpt", ".safetensors"}
        omitted = {
            p.relative_to(root).as_posix(): {
                "bytes": p.stat().st_size,
                "receipt_sha256": required_hashes.get(str(p)),
            }
            for p in files
            if p.suffix.lower() in weights
        }
        files = [p for p in files if p.suffix.lower() not in weights]
    total = sum(p.stat().st_size for p in files)
    required = int(total * 1.05) + 64 * 1024**2
    free = shutil.disk_usage(archive_path.parent).free
    if free < required:
        raise OSError(
            f"Not enough disk space for ZIP: need {required:,} free bytes, have {free:,}. "
            "Keep run artifacts; free unrelated space and retry packaging, not training."
        )
    manifest = {}
    try:
        with zipfile.ZipFile(
            partial, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for path in files:
                name = path.relative_to(root).as_posix()
                with path.open("rb") as stream:
                    digest = _sha(stream)
                if (
                    str(path) in required_hashes
                    and digest != required_hashes[str(path)]
                ):
                    raise ValueError(f"Completed evidence changed: {path}")
                manifest[name] = {"bytes": path.stat().st_size, "sha256": digest}
                archive.write(path, name)
            if results_only:
                payload = json.dumps(
                    {
                        "kind": "results_only",
                        "omitted_checkpoints": omitted,
                        "warning": "Metrics backup, not a standalone training/test resume archive. Retain checkpoints separately; receipt hashes are references, not verification of omitted payloads.",
                    },
                    indent=2,
                ).encode()
                name = "bundle_policy.json"
                archive.writestr(name, payload)
                manifest[name] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            archive.writestr("archive_inventory.json", json.dumps(manifest, indent=2))
        verify_archive(partial)
        partial.replace(
            archive_path
        )  # Preserve an older verified ZIP until replacement verifies.
    except Exception as error:
        raise RuntimeError(
            f"Archive failed: {error}. Artifacts remain at {progress_path.parent}; retry packaging only."
        ) from error
    print(
        f"Verified ZIP: {archive_path} ({archive_path.stat().st_size:,} bytes; {len(manifest)} files)",
        flush=True,
    )
    print(
        "Download this ZIP before deleting the Colab runtime. Browser download completion cannot be verified.",
        flush=True,
    )
    return archive_path


def download_archive(path):
    path = Path(path)
    try:
        from google.colab import files
    except ImportError:
        print(f"Artifacts saved locally: {path}")
        return "local_only"
    try:
        files.download(str(path))
    except Exception as error:
        raise RuntimeError(
            f"Download request failed; retry_download({str(path)!r}) or retrieve the ZIP manually: {error}"
        ) from error
    print(
        "Browser download requested, not confirmed. Check your local Downloads folder.",
        flush=True,
    )
    return "requested"


def compact_results(group, runs_dir, *, announce=False):
    """Keep the latest small metrics backup; checkpoint files remain untouched."""
    group = Path(group).resolve()
    root = Path(runs_dir).resolve().parent
    path = Path(runs_dir).resolve() / "focused_archives" / group.name / "results.zip"
    study_path = group / "study.json"
    standalone = study_path.is_file() and json.loads(
        study_path.read_text(encoding="utf-8")
    ).get("comparison_only")
    directories = {group}
    if standalone:
        # Package this attempt, not the immutable search/import tree. Receipts
        # and the original import hashes remain in the saved study/provenance.
        for receipt in [
            *group.rglob("receipt.json"),
            *(group / "test_receipts").glob("*.json"),
        ]:
            value = json.loads(receipt.read_text(encoding="utf-8"))
            references = value.get("partial_runs", []) + (
                [value["run_dir"]] if value.get("run_dir") else []
            )
            directories.update(Path(p).resolve() for p in references)
    result = make_archive(
        root,
        path,
        group / "progress.json",
        directories,
        (),
        results_only=True,
        follow_links=not standalone,
        receipts_dir=Path(runs_dir) / "study_receipts",
    )
    if announce:
        print(f"FOCUSED_RESULTS_ZIP: {result}", flush=True)
    if study_path.is_file():
        study = json.loads(study_path.read_text(encoding="utf-8"))
        if study.get("backup"):
            backup_results(result, group, study["backup"])
    return result


def backup_paths(backup):
    """Validate syntax without creating folders (safe for --list-plan)."""
    if not isinstance(backup, dict) or set(backup) != {"mount_root", "directory"}:
        raise ValueError("backup requires mount_root and directory")
    mount, directory = (Path(backup[k]) for k in ("mount_root", "directory"))
    # Colab POSIX profiles must also be inspectable on Windows.
    from pathlib import PurePosixPath

    posix_mount, posix_dir = (
        PurePosixPath(backup[k]) for k in ("mount_root", "directory")
    )
    if not (mount.is_absolute() and directory.is_absolute()):
        if not (posix_mount.is_absolute() and posix_dir.is_absolute()):
            raise ValueError("Drive backup paths must be absolute")
    if ".." in directory.parts or ".." in posix_dir.parts:
        raise ValueError("Drive backup must not traverse outside its folder")
    if not (
        directory.is_relative_to(mount / "MyDrive")
        or posix_dir.is_relative_to(posix_mount / "MyDrive")
    ):
        raise ValueError(
            "Backup directory must be within the authorized mount's MyDrive"
        )
    return mount, directory


def preflight_backup(backup):
    """No authentication side effects: the owner mounts Drive before launch."""
    mount, directory = backup_paths(backup)
    if not os.path.ismount(mount) or not (mount / "MyDrive").is_dir():
        raise RuntimeError(
            f"Drive is not mounted/authorized at {mount}. Mount it with owner authorization before training; no local substitute was created."
        )
    mount, directory = mount.resolve(), directory.resolve()
    if not directory.is_relative_to(mount / "MyDrive"):
        raise ValueError("Resolved Drive destination escapes MyDrive")
    directory.mkdir(parents=True, exist_ok=True)
    probe = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".phd-write-probe-", dir=directory, delete=False
        ) as out:
            probe = Path(out.name)
            out.write(b"phd-results-backup-probe")
            out.flush()
            os.fsync(out.fileno())
        if probe.read_bytes() != b"phd-results-backup-probe":
            raise OSError("Drive write/read verification failed")
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)
    return directory


def backup_results(archive, group, backup):
    """Verified campaign-scoped copy; failures keep the local ZIP and prior copy."""
    from src.study_matrix import atomic_json, file_hash, read_json, value_hash

    archive, group = Path(archive).resolve(), Path(group).resolve()
    temporary = None
    try:
        verify_archive(archive)
        base = preflight_backup(backup)
        target_dir = base / group.name
        owner = {
            "study_id": group.name,
            "study_sha256": value_hash(read_json(group / "study.json")),
        }
        owner_file = target_dir / "backup_owner.json"
        if target_dir.exists():
            if not owner_file.is_file() or read_json(owner_file) != owner:
                raise ValueError(
                    "Drive folder belongs to another/unverified campaign; nothing overwritten"
                )
        else:
            target_dir.mkdir()
            atomic_json(owner_file, owner)
        destination = target_dir / "results.zip"
        if destination.is_symlink() or target_dir.is_symlink():
            raise ValueError("Drive backup must not overwrite a symlink")
        size, digest = archive.stat().st_size, file_hash(archive)
        if shutil.disk_usage(target_dir).free < size + 64 * 1024**2:
            raise OSError(
                "Insufficient destination space for verified temporary ZIP copy"
            )
        with tempfile.NamedTemporaryFile(
            prefix=".results-", suffix=".partial", dir=target_dir, delete=False
        ) as out:
            temporary = Path(out.name)
            with archive.open("rb") as source:
                shutil.copyfileobj(source, out)
            out.flush()
            os.fsync(out.fileno())
        if temporary.stat().st_size != size or file_hash(temporary) != digest:
            raise ValueError("Drive temporary copy size/hash mismatch")
        verify_archive(temporary)
        os.replace(temporary, destination)
        if destination.stat().st_size != size or file_hash(destination) != digest:
            raise ValueError("Drive destination verification failed; keep local ZIP")
        receipt = {
            "status": "verified_copy",
            "local_zip": str(archive),
            "destination": str(destination),
            "bytes": size,
            "sha256": digest,
            "note": "Read-back verified now; not a guarantee against later provider/VM loss. Weights excluded.",
        }
        atomic_json(group / "drive_backup.json", receipt)
        print(
            f"Drive results verified: {destination} | {size:,} bytes | SHA256 {digest}",
            flush=True,
        )
        return receipt
    except Exception as error:
        print(
            f"DRIVE BACKUP FAILED: {error}. Local results retained: {archive}; retry backup only, not training.",
            flush=True,
        )
        raise RuntimeError(
            f"Drive backup failed; local ZIP preserved at {archive}: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checkpoint_archive(root, group):
    """Explicit optional full recovery export, independent of training/test."""
    root, group = Path(root).resolve(), Path(group).resolve()
    if not group.is_relative_to(root / "runs/studies"):
        raise ValueError("Specify an existing runs/studies directory")
    return make_archive(
        root,
        root / "runs/focused_archives" / group.name / "with_checkpoints.zip",
        group / "progress.json",
        {group},
        (),
    )


def retry_download(path):
    path = Path(path).expanduser().resolve()
    verify_archive(path)
    return download_archive(path)


def retry_package(root, study_directory):
    """Rebuild from persisted progress and group references only; never train/test."""
    from .aad_study import PROFILES

    root = Path(root).expanduser().resolve()
    directory = Path(study_directory).expanduser().resolve()
    if directory.is_relative_to(root / "runs/studies"):
        study = directory / "study.json"
        if (
            not study.is_file()
            or json.loads(study.read_text(encoding="utf-8")).get("mode")
            != "capacity_search"
        ):
            raise ValueError(
                "Direct study repackaging requires a capacity-search group"
            )
        return make_archive(
            root,
            root / "runs/capacity_archives" / directory.name / "artifacts.zip",
            directory / "progress.json",
            {directory},
            PROFILES,
        )
    if not directory.is_relative_to(root / "runs" / "notebook_studies"):
        raise ValueError("Specify the saved runs/notebook_studies directory")
    progress = directory / "progress.json"
    values = json.loads(progress.read_text(encoding="utf-8"))
    directories = {directory}
    for stage in values["stages"]:
        directories.update(root / p for p in stage.get("run_directories", []))
        if stage.get("selected_config"):
            directories.add(Path(stage["selected_config"]).parent)
    return make_archive(
        root, directory / "artifacts.zip", progress, directories, PROFILES
    )
