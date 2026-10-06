"""Safety and provenance checks for the selectively downloaded Kinetics-600 subset."""

from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest
import torch

from src import kinetics600_subset as kinetics
from src import notebook_config as settings
from src.notebook_data import data_identity, prepare_data
from src.utils import write_json


def make_archive(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    return output.getvalue()


def add_activity_clips(root: Path, groups_per_class: int = 20) -> list[dict]:
    for class_index, label in enumerate(kinetics.KINETICS600_CLASSES):
        directory = root / label
        directory.mkdir(parents=True)
        for index in range(groups_per_class):
            video_id = f"{class_index:02d}{index:09d}"
            (directory / f"{video_id}.mp4").write_bytes(f"{label}-{index}".encode())
            if index == 0:
                (directory / f"{video_id}_000_010.mp4").write_bytes(
                    f"{label}-{index}-second-window".encode()
                )
    return kinetics.inventory_clips(root)


def write_trusted_source(root: Path, files: list[dict]) -> dict:
    report = dict(
        release=kinetics.RELEASE,
        upstream_split="train",
        source_url=settings.KINETICS600_SOURCE_URL,
        origin="downloaded_archives",
        archives=[
            dict(
                class_name=label,
                url=(
                    f"{settings.KINETICS600_SOURCE_URL}/"
                    f"{kinetics.quote(label, safe='')}.tar.gz"
                ),
                bytes=100,
                etag='"fixture"',
                last_modified="fixture",
                sha256="a" * 64,
                archive_validated=True,
            )
            for label in kinetics.KINETICS600_CLASSES
        ],
        inventory_hash=kinetics.content_hash(files),
        files=files,
    )
    write_json(root / ".kinetics600-source.json", report)
    return report


def test_archive_download_is_allowlisted_cached_and_retryable(
    tmp_path, monkeypatch, capsys
):
    url = (
        f"{settings.KINETICS600_SOURCE_URL}/"
        f"{kinetics.quote('headbutting', safe='')}.tar.gz"
    )
    payload = make_archive({"headbutting/abcdefghijk.mp4": b"clip bytes"})
    requests = []

    class Response(BytesIO):
        status = 200

        def __init__(self, body, headers):
            super().__init__(body)
            self.headers = headers

    def open_request(request, timeout):
        requests.append(request.get_method())
        headers = {"Content-Length": str(len(payload)), "ETag": '"fixture"'}
        if request.get_method() == "HEAD":
            return Response(b"", headers)
        assert request.get_header("If-match") == '"fixture"'
        body = payload[:12] if len(requests) == 2 else payload
        return Response(body, headers)

    monkeypatch.setattr(kinetics, "urlopen", open_request)
    target = tmp_path / "archives" / "headbutting.tar.gz"
    with pytest.raises(RuntimeError, match="rerun to retry"):
        kinetics.download_archive(url, target)
    assert not target.exists()
    assert target.with_name(target.name + ".part").is_file()

    record = kinetics.download_archive(url, target)
    assert record["sha256"] == kinetics.file_hash(target)
    extracted = kinetics.extract_archive(target, "headbutting", tmp_path / "dataset")
    assert len(extracted) == 1
    kinetics.extract_archive(
        target, "headbutting", tmp_path / "dataset", show_cache=False
    )
    assert "Cached class:" not in capsys.readouterr().out
    record["archive_validated"] = True
    write_json(target.with_name(target.name + ".json"), record)
    assert kinetics.download_archive(url, target, show_cache=False) == record
    assert "Cached archive:" not in capsys.readouterr().out
    assert requests == ["HEAD", "GET", "HEAD", "GET"]

    with pytest.raises(ValueError, match="Only the five approved"):
        kinetics.download_archive("https://example.com/unrelated.tar.gz", target)


@pytest.mark.parametrize(
    "member_name",
    ["../escape.mp4", "/absolute.mp4", "C:/drive.mp4", "link.mp4"],
)
def test_extractor_rejects_unsafe_members_without_publishing(tmp_path, member_name):
    content = b"untrusted"
    archive_bytes = BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        if member_name == "link.mp4":
            info.type = tarfile.SYMTYPE
            info.linkname = "../escape.mp4"
            archive.addfile(info)
        else:
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    archive_path = tmp_path / "source.tar.gz"
    archive_path.write_bytes(archive_bytes.getvalue())
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="Unsafe|Unexpected"):
        kinetics.extract_archive(archive_path, "headbutting", root)
    assert not (root / "headbutting").exists()
    assert not (tmp_path / "escape.mp4").exists()


def test_corrupt_gzip_trailer_is_rejected_before_class_publish(tmp_path):
    payload = make_archive({"abcdefghijk.mp4": b"clip"})
    damaged = payload[:-1] + bytes([payload[-1] ^ 0xFF])
    archive_path = tmp_path / "damaged.tar.gz"
    archive_path.write_bytes(damaged)
    with pytest.raises(ValueError, match="Corrupt archive"):
        kinetics.extract_archive(archive_path, "headbutting", tmp_path / "dataset")
    assert not (tmp_path / "dataset" / "headbutting").exists()


def test_inventory_requires_five_classes_and_over_two_thousand_unique_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(kinetics, "MIN_CLIPS", 6)
    root = tmp_path / "dataset"
    for index, label in enumerate(kinetics.KINETICS600_CLASSES):
        directory = root / label
        directory.mkdir(parents=True)
        (directory / f"{index:011d}.mp4").write_bytes(f"clip-{index}".encode())
    (root / kinetics.KINETICS600_CLASSES[0] / "00000000001.mp4").write_bytes(b"clip-0")
    with pytest.raises(ValueError, match="at least 6"):
        kinetics.inventory_clips(root)

    monkeypatch.setattr(kinetics, "MIN_CLIPS", 5)
    records = kinetics.inventory_clips(root)
    unique = kinetics.unique_clip_records(records)
    assert len(records) == 6
    assert len(unique) == 5
    assert {row["class_name"] for row in unique} == set(kinetics.KINETICS600_CLASSES)
    duplicate = next(row for row in records if "duplicate_of" in row)
    assert duplicate["duplicate_of"] == "headbutting/00000000000.mp4"


def test_inventory_excludes_identical_content_with_conflicting_labels(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(kinetics, "MIN_CLIPS", 8)
    root = tmp_path / "dataset"
    for class_index, label in enumerate(kinetics.KINETICS600_CLASSES):
        directory = root / label
        directory.mkdir(parents=True)
        (directory / f"{class_index:02d}000000000.mp4").write_bytes(
            f"clip-{class_index}".encode()
        )
        (directory / f"{class_index:02d}000000001.mp4").write_bytes(
            f"second-{class_index}".encode()
        )
    (root / kinetics.KINETICS600_CLASSES[1] / "01000000000.mp4").write_bytes(b"clip-0")

    records = kinetics.inventory_clips(root)
    unique = kinetics.unique_clip_records(records)

    assert len(records) == 10
    assert len(unique) == 8
    conflicts = [row for row in records if "duplicate_label_conflict" in row]
    assert len(conflicts) == 2
    assert all(
        row["duplicate_label_conflict"] == sorted(kinetics.KINETICS600_CLASSES[:2])
        for row in conflicts
    )


def test_default_inventory_gate_is_more_than_two_thousand():
    assert kinetics.MIN_CLIPS == 2001


def test_local_root_needs_verified_source_marker_and_matching_inventory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(kinetics, "MIN_CLIPS", 50)
    root = tmp_path / "local-source"
    files = add_activity_clips(root, groups_per_class=10)
    with pytest.raises(ValueError, match="provenance file"):
        kinetics.prepare_subset(tmp_path, root)

    report = write_trusted_source(root, files)
    prepared_root, verified = kinetics.prepare_subset(tmp_path, root)
    assert prepared_root == root
    assert verified == report
    assert not (tmp_path / "runs/datasets").exists()

    (root / kinetics.KINETICS600_CLASSES[0] / "00000000000.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        kinetics.prepare_subset(tmp_path, root)


def test_grouped_manifest_is_repeatable_and_keeps_video_ids_together(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(kinetics, "MIN_CLIPS", 50)
    monkeypatch.setattr(settings, "SELECTED_DIAGNOSTIC_DATASET", "kinetics600-subset")
    root = tmp_path / "kinetics-600-train"
    files = add_activity_clips(root)
    duplicate_path = root / kinetics.KINETICS600_CLASSES[0] / "99999999999.mp4"
    duplicate_path.write_bytes((root / files[0]["path"]).read_bytes())
    files = kinetics.inventory_clips(root)
    source = write_trusted_source(root, files)
    prepared = prepare_data(tmp_path, root)
    assert prepared["dataset"].class_names == sorted(kinetics.KINETICS600_CLASSES)
    assert len(prepared["dataset"]) == len(kinetics.unique_clip_records(files))
    assert all(path != duplicate_path for path, _ in prepared["dataset"].samples)
    assert prepared["split"]["assignment_unit"] == "source_video"
    assert data_identity(prepared)["source_inventory_hash"] == source["inventory_hash"]
    assert all(
        abs(prepared["split"]["actual_ratios"][name] - ratio) <= 0.05
        for name, ratio in (("train", 0.70), ("validation", 0.15), ("test", 0.15))
    )
    assert prepared["dataset"].allowed_indices.isdisjoint(prepared["test"].indices)

    manifest_bytes = prepared["manifest_path"].read_bytes()
    manifest = json.loads(manifest_bytes)
    membership = {row["path"]: row["split"] for row in manifest["samples"]}
    groups = {}
    for row in manifest["samples"]:
        previous = groups.setdefault(row["source_group"], row["split"])
        assert previous == row["split"]
    first_window = "headbutting/00000000000.mp4"
    second_window = "headbutting/00000000000_000_010.mp4"
    assert membership[first_window] == membership[second_window]

    repeated = prepare_data(tmp_path, root)
    assert repeated["manifest_path"] == prepared["manifest_path"]
    assert repeated["manifest_path"].read_bytes() == manifest_bytes

    from src import notebook_workflows as workflows

    checkpoint_path = tmp_path / "old-source.pth"
    torch.save({"data_identity": data_identity(prepared)}, checkpoint_path)
    monkeypatch.setattr(workflows, "validate_selected_checkpoint", lambda *a: None)
    source_record = source["archives"][0]
    source_record["sha256"] = "b" * 64
    write_json(root / ".kinetics600-source.json", source)
    changed_source = prepare_data(tmp_path, root)
    assert data_identity(changed_source) != data_identity(prepared)
    with pytest.raises(ValueError, match="incompatible"):
        workflows.load_checkpoint(changed_source, checkpoint_path)

    changed_manifest = json.loads(changed_source["manifest_path"].read_text())
    changed_manifest["samples"][0]["source_group"] = "tampered"
    changed_source["manifest_path"].write_text(
        json.dumps(changed_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source-video IDs"):
        prepare_data(tmp_path, root)
