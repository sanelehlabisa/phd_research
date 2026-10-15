"""Read-only, hash-verified relocation of focused Colab selection evidence.

An import manifest maps original absolute prefixes to local directories. No
checkpoint, source JSON or search ranking is rewritten. Search weights are not
needed for fresh training; their metadata and complete validation records are.
"""

from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath

from .study_matrix import read_json, file_hash, value_hash


def portable(path):
    return PurePosixPath(str(path).replace("\\", "/"))


class EvidenceImport:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.spec = read_json(self.path)
        if self.spec.get("protocol") != "comparison_import_v1":
            raise ValueError("Expected comparison_import_v1 manifest")
        self.mappings = []
        for item in self.spec["path_map"]:
            source = portable(item["source"])
            local = (self.path.parent / item["destination"]).resolve()
            if (
                not (source.is_absolute() or PureWindowsPath(str(source)).is_absolute())
                or ".." in source.parts
                or not local.is_dir()
            ):
                raise ValueError(
                    "Import map requires an original absolute prefix and existing directory"
                )
            self.mappings.append((source, local))
        if not self.mappings or len({s for s, _ in self.mappings}) != len(
            self.mappings
        ):
            raise ValueError("Missing or duplicate import prefixes")

    def relocate(self, path):
        original = portable(path)
        if ".." in original.parts:
            raise ValueError("Parent traversal in imported path")
        matches = [(s, d) for s, d in self.mappings if original.is_relative_to(s)]
        if not matches:
            raise FileNotFoundError(f"No explicit import mapping for {path}")
        source, destination = max(matches, key=lambda pair: len(pair[0].parts))
        result = destination.joinpath(*original.relative_to(source).parts).resolve()
        if not result.is_relative_to(destination):
            raise ValueError("Import path escapes its declared destination")
        return result

    def verify(self, path, digest):
        local = self.relocate(path)
        if not local.is_file():
            raise FileNotFoundError(
                f"Required imported evidence missing: {path} -> {local}"
            )
        if file_hash(local) != digest:
            raise ValueError(f"Imported evidence hash mismatch: {path}")
        return local

    def inventory(self, files, *, weights=False):
        if not files:
            raise ValueError("Missing original evidence inventory")
        for path, digest in files.items():
            if not weights and portable(path).suffix in {
                ".pth",
                ".pt",
                ".ckpt",
                ".safetensors",
            }:
                continue
            self.verify(path, digest)

    def checked(self, path, files):
        matches = [
            digest
            for source, digest in files.items()
            if portable(source) == portable(path)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Required evidence not uniquely covered by hashes: {path}"
            )
        return self.verify(path, matches[0])

    def json(self, path):
        return read_json(self.relocate(path))

    def result_records(self, result, split):
        """Reconcile the original validation totals without relabelling history."""
        import math
        from .study_reporting import pooled_metrics, confusion_counts

        cfg = result["experiment_config"]
        records = self.json(result["validation_predictions"])
        expected = {
            str(portable(cfg["dataset_dir"]) / r["path"]): r["class_index"]
            for r in split["samples"]
            if r["split"] == "validation"
        }
        paths = [str(portable(r["source"])) for r in records]
        if (
            len(paths) != len(expected)
            or set(paths) != set(expected)
            or any(
                r["partition"] != "validation" or r["target"] != expected[p]
                for r, p in zip(records, paths)
            )
        ):
            raise ValueError(
                "Imported predictions differ from the full validation split"
            )
        metrics = pooled_metrics(records, split["class_names"])
        if any(
            not math.isclose(
                result["validation_metrics"].get(k, float("nan")),
                v,
                abs_tol=1e-5,
                rel_tol=1e-5,
            )
            for k, v in metrics.items()
        ):
            raise ValueError("Imported validation metrics disagree with predictions")
        counts = confusion_counts(records, split["class_names"])
        if result["validation_confusion_matrix"]["matrix"] != counts["matrix"]:
            raise ValueError("Imported confusion counts disagree with predictions")
        return records


def load_imported_selection(path):
    """Verify all 64 original jobs and preserve the exact recorded decision."""
    from .focused_protocol import decisions, candidates, SLOTS, blocks
    from .experiment_config import ExperimentConfig

    evidence = EvidenceImport(path)
    ref = evidence.spec["selection"]
    selected = read_json(evidence.verify(ref["path"], ref["sha256"]))
    if (
        selected.get("protocol") != "focused_top5_v1"
        or selected.get("selection_partition") != "validation"
        or selected.get("test_access") != "locked"
        or selected.get("seed") != 42
    ):
        raise ValueError(
            "Only a completed focused validation selection can be imported"
        )
    group = portable(selected["source_group"])
    sources = {}
    for filename, field in (
        ("study.json", "source_study_sha256"),
        ("progress.json", "source_progress_sha256"),
        ("split_audit.json", "source_audit_sha256"),
    ):
        sources[filename] = read_json(
            evidence.verify(group / filename, selected[field])
        )
    state, audit = sources["progress.json"], sources["split_audit.json"]
    if state["status"] != "complete" or len(state["jobs"]) != 64:
        raise ValueError("Import requires all 64 completed search jobs")
    evidence.inventory(state["preparation_files"])
    if (
        selected["code_identity"] != state["code_identity"]
        or len(selected["top5"]) != 5
    ):
        raise ValueError("Imported selection code identity/finalist count mismatch")
    expected_configs = {
        (row["model"], row["config"]["height"], row["config"]["target_fps"]): row[
            "config"
        ]
        for _, rows in blocks(ExperimentConfig.from_mapping(state["effective_config"]))
        for row in rows
    }
    split_path = evidence.verify(
        selected["split"]["manifest_path"], selected["split"]["manifest_hash"]
    )
    split = read_json(split_path)
    expected_files = {
        str(portable(selected["dataset_dir"]) / r["path"]) for r in split["samples"]
    }
    if {str(portable(r["path"])) for r in audit["files"]} != expected_files:
        raise ValueError(
            "Imported dataset hash audit does not cover the complete split"
        )
    review = audit.get("source_review", {})
    if (
        audit["cross_split_file_hashes"]
        or audit["cross_split_source_ids"]
        or review.get("disposition") != "independent_recordings_user_confirmed"
        or review.get("split_content_sha256") != value_hash(split)
    ):
        raise ValueError(
            "Missing compatible source review or known cross-split duplication"
        )
    evidence.inventory({r["path"]: r["sha256"] for r in audit["files"]})
    specs = {c["name"]: c for c in candidates()}
    for job in state["jobs"]:
        if job["status"] != "complete":
            raise ValueError("Incomplete imported search job")
        receipt = evidence.json(job["receipt"])
        result = job["result"]
        if receipt["status"] != "complete" or receipt["result"] != result:
            raise ValueError("Imported receipt/result mismatch")
        evidence.inventory(receipt["files"])
        for required in (
            result["validation_predictions"],
            str(portable(result["selected_checkpoint"]).with_suffix(".json")),
        ):
            evidence.checked(required, receipt["files"])
        cfg = result["experiment_config"]
        if cfg != expected_configs.get(
            (result["name"], cfg["height"], cfg["target_fps"])
        ):
            raise ValueError(
                "Imported search configuration differs from the recorded protocol"
            )
        if (
            result["split"]["manifest_hash"] != selected["split"]["manifest_hash"]
            or cfg["convlstm_layers"] != specs[result["name"]]["convlstm_layers"]
            or cfg["hidden_classifier_width"]
            != specs[result["name"]]["hidden_classifier_width"]
        ):
            raise ValueError("Imported architecture/split mismatch")
        sidecar = evidence.json(
            portable(result["selected_checkpoint"]).with_suffix(".json")
        )
        if (
            sidecar["experiment_config"] != cfg
            or sidecar["selection_partition"] != "validation"
            or sidecar["checkpoint_role"] != "validation_selected_lowest_loss"
            or sidecar["selected_epoch"]
            != result["checkpoint_selection"]["selected_epoch"]
        ):
            raise ValueError("Imported checkpoint metadata mismatch")
        evidence.result_records(result, split)
    decision = decisions(state["jobs"])
    if any(selected.get(key) != value for key, value in decision.items()):
        raise ValueError(
            "Imported selection differs from verified original search decision"
        )
    for i, item in enumerate(selected["top5"]):
        name = decision["finalist_names"][i]
        if item["comparison_name"] != SLOTS[i] or item["candidate"] != specs[name]:
            raise ValueError("Imported finalist differs from the frozen selection")
        jobs = [j for j in state["jobs"] if j["result"]["name"] == name]
        expected = [
            dict(
                run_dir=j["run_dir"],
                receipt=j["receipt"],
                receipt_sha256=file_hash(evidence.relocate(j["receipt"])),
                config_sha256=value_hash(j["result"]["experiment_config"]),
            )
            for j in jobs
        ]
        if item["sources"] != expected:
            raise ValueError("Finalist sources do not match verified receipts")
    local = deepcopy(selected)
    local["dataset_dir"] = str(evidence.relocate(selected["dataset_dir"]))
    local["split"]["manifest_path"] = str(split_path)
    local["source_group"] = str(evidence.relocate(group))
    local["verified_import"] = {
        "manifest": str(evidence.path),
        "sha256": file_hash(evidence.path),
        "selection_sha256": ref["sha256"],
        "search_jobs": 64,
    }
    return local
