"""Five-class Kinetics-600 downloads, content inventory and source-grouped splits."""

import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import Request, urlopen

from sklearn.model_selection import StratifiedGroupKFold

from .notebook_config import KINETICS600_CLASSES, KINETICS600_SOURCE_URL
from .dataset import (
    _dataset_inventory,
    _inventory_hash,
    _split_counts,
    load_split_subsets,
)
from .utils import write_json

MIN_CLIPS = 2001
RELEASE = "kinetics-600"
GROUPING = "youtube-id-when-available-v1"
_verified_hashes = {}


def content_hash(value) -> str:
    """Hash a stable JSON value, without machine-specific paths."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    """Verify file content once per unchanged size, modification and change time."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _verified_hashes:
        with path.open("rb") as stream:
            _verified_hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    return _verified_hashes[key]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_target(path: Path, root: Path) -> None:
    """Reject links and paths outside this task's cache before any write."""
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Cache path escapes its root: {path}")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Symlink is not allowed in the cache: {component}")
        if component == root:
            break


def download_archive(url: str, target: Path) -> dict:
    """Reuse verified archives; retry interrupted downloads without a full-release fallback."""
    allowed = {
        f"{KINETICS600_SOURCE_URL}/{quote(name, safe='')}.tar.gz"
        for name in KINETICS600_CLASSES
    }
    if url not in allowed:
        raise ValueError(
            "Only the five approved Kinetics-600 training archives are allowed"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    record_path = target.with_name(target.name + ".json")
    partial = target.with_name(target.name + ".part")
    for path in (target, record_path, partial):
        _safe_target(path, target.parent)
    if target.is_file() and record_path.is_file():
        saved = _read(record_path)
        if (
            saved.get("url") == url
            and saved.get("archive_validated") is True
            and saved.get("bytes") == target.stat().st_size
            and saved.get("sha256") == file_hash(target)
        ):
            print(f"Cached archive: {target.name}", flush=True)
            return saved
    with urlopen(Request(url, method="HEAD"), timeout=30) as response:
        size = int(response.headers.get("Content-Length", 0))
        if response.status != 200 or size <= 0:
            raise ValueError(f"Archive size unavailable: {url}")
        identity = dict(
            url=url,
            bytes=size,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )
    headers = {"If-Match": identity["etag"]} if identity["etag"] else {}
    print(f"Downloading {target.name}: {size / 1_000_000:.1f} MB", flush=True)
    digest, received, next_progress = hashlib.sha256(), 0, 32 * 1024 * 1024
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            if (
                response.status != 200
                or int(response.headers.get("Content-Length", 0)) != size
            ):
                raise ValueError("Archive changed between inventory and download")
            if identity["etag"] and response.headers.get("ETag") != identity["etag"]:
                raise ValueError("Archive ETag changed during download")
            with partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > size:
                        raise ValueError("Archive exceeds its declared size")
                    output.write(block)
                    digest.update(block)
                    if received >= next_progress:
                        print(f"  {received / size:.0%} downloaded", flush=True)
                        next_progress += 32 * 1024 * 1024
        if received != size:
            raise ValueError(f"Incomplete archive: received {received}/{size} bytes")
    except Exception as error:
        raise RuntimeError(
            f"Download failed for {url}; rerun to retry: {error}"
        ) from error
    partial.replace(target)
    record = dict(identity, sha256=digest.hexdigest(), archive_validated=False)
    write_json(record_path, record)
    return record


def source_video_id(name: str) -> str | None:
    """Read the YouTube ID from a native Kinetics clip filename, when available."""
    match = re.fullmatch(r"([A-Za-z0-9_-]{11})(?:_[0-9]+_[0-9]+)?\.mp4", name)
    return match.group(1) if match else None


def _clip_record(path: Path, label: str) -> dict:
    return dict(
        path=f"{label}/{path.name}",
        class_name=label,
        bytes=path.stat().st_size,
        sha256=file_hash(path),
        source_video_id=source_video_id(path.name),
    )


def unique_clip_records(records: list[dict]) -> list[dict]:
    """Return only one unambiguous path for each exact video content hash."""
    return [
        row
        for row in records
        if "duplicate_of" not in row and "duplicate_label_conflict" not in row
    ]


def extract_archive(archive_path: Path, label: str, dataset_root: Path) -> list[dict]:
    """Stage safe MP4 members and verify the gzip trailer before publishing a class."""
    if label not in KINETICS600_CLASSES:
        raise ValueError("Unexpected Kinetics-600 activity")
    destination = dataset_root / label
    _safe_target(destination, dataset_root)
    record_path = archive_path.with_name(archive_path.name + ".extracted.json")
    _safe_target(record_path, archive_path.parent)
    archive_hash = file_hash(archive_path)
    if record_path.is_file():
        saved = _read(record_path)
        if saved.get("archive_sha256") == archive_hash and destination.is_dir():
            expected = {PurePosixPath(row["path"]).name for row in saved["files"]}
            actual = {p.name for p in destination.iterdir()}
            if actual - expected:
                raise ValueError(f"Unexpected files in cached class: {destination}")
            if actual == expected and all(
                not (destination / PurePosixPath(row["path"]).name).is_symlink()
                and _clip_record(destination / PurePosixPath(row["path"]).name, label)
                == row
                for row in saved["files"]
            ):
                print(f"Cached class: {label} ({len(expected)} clips)", flush=True)
                return saved["files"]
    dataset_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="extract-", dir=archive_path.parent) as temporary:
        staging = Path(temporary)
        records = []
        seen = set()
        try:
            with gzip.open(archive_path, "rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r|") as archive:
                    for member in archive:
                        path = PurePosixPath(member.name)
                        if (
                            path.is_absolute()
                            or ".." in path.parts
                            or any(c in member.name for c in ("\\", ":", "\0"))
                            or member.issym()
                            or member.islnk()
                        ):
                            raise ValueError(f"Unsafe archive member: {member.name}")
                        if member.isdir() and path.parts in ((), (label,)):
                            continue
                        if (
                            not member.isfile()
                            or member.size <= 0
                            or path.suffix.lower() != ".mp4"
                            or len(path.parts) not in (1, 2)
                            or (len(path.parts) == 2 and path.parts[0] != label)
                            or path.name in seen
                        ):
                            raise ValueError(
                                f"Unexpected or duplicate archive member: {member.name}"
                            )
                        seen.add(path.name)
                        target = staging / path.name
                        with archive.extractfile(member) as source, target.open(
                            "wb"
                        ) as output:
                            shutil.copyfileobj(source, output)
                        if target.stat().st_size != member.size:
                            raise ValueError(f"Truncated clip: {member.name}")
                        records.append(_clip_record(target, label))
                # tar ends at zero blocks; draining gzip also checks its CRC/trailer.
                while block := compressed.read(1024 * 1024):
                    if block.strip(b"\0"):
                        raise ValueError("Unexpected trailing archive data")
        except (OSError, EOFError, tarfile.TarError) as error:
            raise ValueError(
                f"Corrupt archive {archive_path}; no class was published"
            ) from error
        if not records:
            raise ValueError(f"Empty class archive: {label}")
        destination.mkdir(parents=True, exist_ok=True)
        if {p.name for p in destination.iterdir()} - seen:
            raise ValueError(f"Unexpected files in class directory: {destination}")
        for record in records:
            target = destination / PurePosixPath(record["path"]).name
            _safe_target(target, dataset_root)
            (staging / target.name).replace(target)
    records.sort(key=lambda row: row["path"])
    write_json(record_path, dict(archive_sha256=archive_hash, files=records))
    print(f"Extracted {label}: {len(records)} clips", flush=True)
    return records


def inventory_clips(dataset_root: Path) -> list[dict]:
    """Inventory files and flag exact copies without decoding any videos."""
    records, hashes = [], {}
    for label in KINETICS600_CLASSES:
        directory = dataset_root / label
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Missing or unsafe activity folder: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".mp4":
                raise ValueError(f"Unexpected file in activity folder: {path}")
            record = _clip_record(path, label)
            if record["bytes"] <= 0:
                raise ValueError(f"Empty clip: {path}")
            hashes.setdefault(record["sha256"], []).append(record)
            records.append(record)

    for copies in hashes.values():
        canonical = copies[0]
        labels = sorted({row["class_name"] for row in copies})
        for duplicate in copies[1:]:
            duplicate["duplicate_of"] = canonical["path"]
        if len(labels) > 1:
            for row in copies:
                row["duplicate_label_conflict"] = labels

    unique_records = unique_clip_records(records)
    for label in KINETICS600_CLASSES:
        count = sum(row["class_name"] == label for row in unique_records)
        if not count:
            raise ValueError(
                f"No usable unique clips remain for {label}; check duplicate-content "
                "label conflicts in the source inventory"
            )
        print(f"  {label}: {count} unique clips", flush=True)
    if len(unique_records) < MIN_CLIPS:
        raise ValueError(
            f"Kinetics-600 has {len(unique_records)} usable unique clips after exact "
            f"duplicate removal; at least {MIN_CLIPS} required before training"
        )
    conflicting = sum("duplicate_label_conflict" in row for row in records)
    excluded_copies = sum(
        "duplicate_of" in row and "duplicate_label_conflict" not in row
        for row in records
    )
    print(
        f"Kinetics-600 subset: {len(unique_records)} usable unique videos; "
        f"excluded {excluded_copies} exact copies and {conflicting} "
        "cross-label duplicate files.",
        flush=True,
    )
    return sorted(records, key=lambda row: row["path"])


def prepare_subset(root: Path, dataset_root: Path | None = None) -> tuple[Path, dict]:
    """Prepare only five upstream training classes, or inventory an existing local root."""
    archives = None
    if dataset_root is None:
        cache = Path(root) / "runs" / "datasets" / "kinetics600-five-activities"
        dataset_root = cache / "files" / "kinetics-600-train"
        _safe_target(dataset_root, cache)
        archives = []
        for label in KINETICS600_CLASSES:
            path = cache / "archives" / f"{label}.tar.gz"
            _safe_target(path, cache)
            url = f"{KINETICS600_SOURCE_URL}/{quote(label, safe='')}.tar.gz"
            record = download_archive(url, path)
            extract_archive(path, label, dataset_root)
            record["archive_validated"] = True
            write_json(path.with_name(path.name + ".json"), record)
            archives.append(dict(class_name=label, **record))
    dataset_root = Path(dataset_root)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError(
            f"Kinetics-600 local root must be a real directory: {dataset_root}"
        )
    files = inventory_clips(dataset_root)
    marker = dataset_root / ".kinetics600-source.json"
    inventory_hash = content_hash(files)
    if archives is None:
        if not marker.is_file():
            raise ValueError(
                "A supplied Kinetics-600 root needs its .kinetics600-source.json "
                "provenance file; use the selective downloader to create one"
            )
        saved = _read(marker)
        if (
            saved.get("release") != RELEASE
            or saved.get("upstream_split") != "train"
            or saved.get("source_url") != KINETICS600_SOURCE_URL
            or saved.get("inventory_hash") != inventory_hash
            or saved.get("origin") != "downloaded_archives"
            or not isinstance(saved.get("archives"), list)
            or len(saved["archives"]) != len(KINETICS600_CLASSES)
            or any(not isinstance(row, dict) for row in saved["archives"])
            or {row.get("class_name") for row in saved["archives"]}
            != set(KINETICS600_CLASSES)
            or any(
                row.get("archive_validated") is not True
                or row.get("url")
                != f"{KINETICS600_SOURCE_URL}/{quote(row['class_name'], safe='')}.tar.gz"
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
                for row in saved["archives"]
            )
        ):
            raise ValueError(
                "Local Kinetics-600 provenance does not match the verified release, "
                "five archives, or current clip inventory"
            )
        report = saved
    else:
        report = dict(
            release=RELEASE,
            upstream_split="train",
            source_url=KINETICS600_SOURCE_URL,
            origin=(
                "downloaded_archives" if archives is not None else "supplied_local_root"
            ),
            archives=archives,
            inventory_hash=inventory_hash,
            files=files,
        )
        if archives is not None:
            _safe_target(marker, dataset_root)
            write_json(marker, report)
    return dataset_root, report


def load_grouped_subsets(
    dataset, manifest_path: Path, source: dict, seed: int
) -> tuple:
    """Reuse generic manifest validation and additionally enforce source-ID isolation."""
    inventory = _dataset_inventory(dataset)
    source_by_path = {row["path"]: row for row in unique_clip_records(source["files"])}
    if {row["path"] for row in inventory} != set(source_by_path):
        raise ValueError("Dataset differs from the verified source inventory")
    groups = [
        source_by_path[row["path"]]["source_video_id"] or "clip:" + row["path"]
        for row in inventory
    ]
    labels = [row["class_index"] for row in inventory]
    if not manifest_path.is_file():
        group_counts = [
            len({g for g, label in zip(groups, labels) if label == i})
            for i in range(dataset.num_classes)
        ]
        folds = min(20, min(group_counts))
        if folds < 5:
            raise ValueError(
                "Source-grouped split needs at least five source groups per activity"
            )
        held_out_folds = max(1, round(folds * 0.15))
        membership = {}
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (_, indices) in enumerate(splitter.split(labels, labels, groups)):
            partition = (
                "validation"
                if fold < held_out_folds
                else "test" if fold < 2 * held_out_folds else "train"
            )
            membership.update({int(index): partition for index in indices})
        samples = [
            dict(row, source_group=groups[i], split=membership[i])
            for i, row in enumerate(inventory)
        ]
        counts, class_counts = _split_counts(samples, dataset.class_names)
        if any(
            class_counts[name][partition] == 0
            for name in dataset.class_names
            for partition in ("train", "validation", "test")
        ):
            raise ValueError(
                "Source-grouped split must represent every activity in every partition"
            )
        write_json(
            manifest_path,
            dict(
                schema_version=1,
                dataset_name=dataset.dataset_dir.resolve().name,
                split_level="clip",
                grouping=GROUPING,
                assignment_unit="source_video",
                seed=seed,
                requested_ratios=dict(train=0.7, validation=0.15, test=0.15),
                actual_ratios={
                    name: counts[name] / counts["total"]
                    for name in ("train", "validation", "test")
                },
                class_names=dataset.class_names,
                inventory_hash=_inventory_hash(dataset.class_names, inventory),
                source_inventory_hash=source["inventory_hash"],
                source_hash=content_hash(source),
                counts=counts,
                class_counts=class_counts,
                samples=samples,
            ),
        )
    saved = _read(manifest_path)
    if (
        saved.get("source_hash") != content_hash(source)
        or saved.get("source_inventory_hash") != source["inventory_hash"]
        or saved.get("grouping") != GROUPING
        or saved.get("assignment_unit") != "source_video"
    ):
        raise ValueError("Grouped manifest has incompatible source provenance")
    group_membership = {}
    expected_groups = {row["path"]: group for row, group in zip(inventory, groups)}
    for sample in saved["samples"]:
        group = expected_groups.get(sample["path"])
        if sample.get("source_group") != group or group is None:
            raise ValueError("Grouped manifest changed source-video IDs")
        partition = sample["split"]
        if group in group_membership and group_membership[group] != partition:
            raise ValueError("Source-video group crosses dataset partitions")
        group_membership[group] = partition
    train, validation, test, metadata = load_split_subsets(
        dataset, manifest_path, seed=seed
    )
    metadata.update(
        grouping=GROUPING,
        assignment_unit="source_video",
        source_inventory_hash=source["inventory_hash"],
        source_hash=content_hash(source),
        known_source_clips=sum(
            row["source_video_id"] is not None for row in source["files"]
        ),
        actual_ratios={
            name: metadata["counts"][name] / metadata["counts"]["total"]
            for name in ("train", "validation", "test")
        },
    )
    return train, validation, test, metadata
