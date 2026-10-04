"""Version-pinned, file-only Kinetics downloads and exact activity filtering."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import kagglehub

from .utils import write_json


def versioned_handle(handle):
    """Require an explicit version before querying or downloading Kinetics."""
    match = re.fullmatch(r"([\w-]+)/([\w-]+)/versions/([1-9][0-9]*)", handle)
    if not match:
        raise ValueError("Kinetics requires owner/dataset/versions/NUMBER")
    owner, slug, version = match.groups()
    return owner, slug, int(version)


def matched_interests(available, interests):
    """Preserve exact labels; absent interests are informational, not errors."""
    available, requested = set(available), set(interests)
    matched = sorted(available & requested)
    skipped = sorted(requested - available)
    print(f"Matched Kinetics classes: {matched}", flush=True)
    print(f"Skipped absent interests: {skipped}", flush=True)
    if len(matched) < 2:
        raise ValueError("Kinetics needs at least two matching class folders")
    return matched


def _get_page(handle, token):
    owner, slug, version = versioned_handle(handle)
    query = dict(datasetVersionNumber=version, pageSize=200)
    if token:
        query["pageToken"] = token
    url = (
        f"https://www.kaggle.com/api/v1/datasets/list/{owner}/{slug}?{urlencode(query)}"
    )
    request = Request(url, headers={"User-Agent": "phd-research-kinetics-subset"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _validated_files(files, slug):
    """Validate remote/cache paths before passing any of them to a downloader."""
    result = []
    seen = set()
    for item in files:
        name, size = item["name"], item["bytes"]
        # This copy includes unfinished .mp4.part downloads in other classes.
        # They are not videos and AHARDataset would not load them either.
        if PurePosixPath(name).suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
            continue
        parts = name.split("/")
        if (
            len(parts) != 3
            or parts[0] != slug
            or any(part in ("", ".", "..") for part in parts)
            or any(character in name for character in ("\\", ":", "\0"))
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or name in seen
        ):
            raise ValueError(f"Invalid or duplicate Kinetics inventory entry: {name}")
        seen.add(name)
        result.append(dict(name=name, bytes=size))
    if not result:
        raise ValueError("Kinetics filename inventory is empty")
    return sorted(result, key=lambda item: item["name"])


def load_inventory(handle, cache):
    """Cache complete versioned filename metadata, never a dataset archive."""
    _, slug, _ = versioned_handle(handle)
    path = Path(cache) / "inventory.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["handle"] != handle:
            raise ValueError("Cached Kinetics inventory belongs to another version")
        return _validated_files(saved["files"], slug)
    files, seen_tokens, token, page_number = [], set(), "", 0
    while True:
        page = _get_page(handle, token)
        if page.get("errorMessage"):
            raise ValueError(f"Kaggle inventory failed: {page['errorMessage']}")
        files.extend(
            dict(name=item["name"], bytes=item["totalBytes"])
            for item in page["datasetFiles"]
        )
        page_number += 1
        print(
            f"Kinetics inventory: page {page_number}, {len(files)} filenames",
            flush=True,
        )
        token = page.get("nextPageToken", "")
        if not token:
            break
        if token in seen_tokens:
            raise ValueError("Kaggle inventory returned a repeated pagination token")
        seen_tokens.add(token)
    files = _validated_files(files, slug)
    write_json(path, dict(handle=handle, files=files))
    return files


@contextmanager
def _file_downloads_only():
    names = ("DISABLE_COLAB_CACHE", "DISABLE_KAGGLE_CACHE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(dict.fromkeys(names, "true"))
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def download_subset(root, handle, interests):
    """Download only matching files; reruns reuse inventory and size-checked clips."""
    _, slug, _ = versioned_handle(handle)
    key = hashlib.sha256(handle.encode()).hexdigest()[:16]
    cache = Path(root) / "runs" / "datasets" / f"kinetics-{key}"
    files = load_inventory(handle, cache)
    matched = matched_interests((f["name"].split("/")[1] for f in files), interests)
    selected = [f for f in files if f["name"].split("/")[1] in matched]
    print(
        f"Kinetics subset: {len(selected)} videos, "
        f"{sum(f['bytes'] for f in selected) / 1_000_000:.1f} MB; no full archive.",
        flush=True,
    )
    output = (cache / "files").resolve()
    # KaggleHub's host mounts ignore output_dir even for a single-file request.
    # Disable only these mounts for this operation; preserve the user's settings.
    with _file_downloads_only():
        for number, item in enumerate(selected, 1):
            target = output.joinpath(*item["name"].split("/"))
            if not target.resolve().is_relative_to(output):
                raise ValueError("Kinetics cache path escapes its download directory")
            cached = target.is_file() and target.stat().st_size == item["bytes"]
            print(
                f"  {number}/{len(selected)} {'Cached' if cached else 'Downloading'}: {item['name']}",
                flush=True,
            )
            if not cached:
                if target.exists() and not target.is_file():
                    raise ValueError(f"Kinetics cache target is not a file: {target}")
                downloaded = kagglehub.dataset_download(
                    handle,
                    path=item["name"],
                    output_dir=str(output),
                    force_download=target.exists(),
                )
                if Path(downloaded).resolve() != target.resolve():
                    raise ValueError(
                        "KaggleHub did not return the requested local file"
                    )
                if not target.is_file() or target.stat().st_size != item["bytes"]:
                    raise ValueError(
                        f"Incomplete Kinetics download: {item['name']}; rerun to retry"
                    )
    return output / slug, matched
