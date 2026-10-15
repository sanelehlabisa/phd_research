"""089's approved recipe and bounded train-only gate; no alternative trainer."""

from pathlib import Path

from .study_matrix import (
    atomic_json,
    file_hash,
    read_json,
    value_hash,
    verify_inventory,
)

APPROVED_RECIPE = {
    "approval": "owner_2026-10-15",
    "weights": None,
    "default_learning_rate": 0.001,
    "swin_learning_rate": 0.00001,
    "check_steps": 128,
    "check_rule": "finite_updates_lower_loss_and_perfect_tiny_training_accuracy",
}


def model_learning_rate(name):
    return 0.00001 if name in ("swin3d_t", "swin3d_s") else 0.001


def freeze_resolution_checks(group, jobs, selected):
    """Document 64px ranking separately; retain all five and never tune on test."""
    from .study_matrix import validation_accuracy

    names = [item["comparison_name"] for item in selected["top5"]]
    if len(jobs) != 5 or [j["result"]["name"] for j in jobs] != names:
        raise ValueError(
            "All five 64px validation checks must complete before final training"
        )
    ranked = sorted(
        jobs,
        key=lambda j: (
            -validation_accuracy(j["result"], True),
            j["result"]["validation_metrics"]["loss"],
            j["result"]["num_params"],
            j["result"]["name"],
        ),
    )
    report = dict(
        selection_partition="validation",
        test_access="locked",
        search_input=selected["final_input"],
        final_size=[64, 64],
        search_order=names,
        validation_64px_order=[j["result"]["name"] for j in ranked],
        policy="Retain all five; independent fresh final training. No extra FPS/recipe tuning or test selection.",
        checks=[
            dict(
                model=j["result"]["name"],
                receipt=j["receipt"],
                receipt_sha256=file_hash(j["receipt"]),
                validation_metrics=j["result"]["validation_metrics"],
            )
            for j in jobs
        ],
        note="Only shortlisted architectures checked at 64px; not an exhaustive high-resolution search.",
    )
    path = Path(group) / "resolution_checks.json"
    if path.exists() and read_json(path) != report:
        raise ValueError("Frozen 64px check evidence changed; final testing locked")
    atomic_json(path, report)


def check_passed(report):
    learning, updates = report["learning"], report["gradient_updates"]
    return (
        len(updates) == 128
        and report["optimizer_covers_all_parameters"]
        and learning[-1]["accuracy"] == 1.0
        and learning[-1]["loss"] < learning[0]["loss"]
        and all(u["all_gradients_present_and_finite"] for u in updates)
        and all(
            any(v["update_norm"] > 0 for v in u["parameters"].values()) for u in updates
        )
    )


def run_transformer_checks(config, destination):
    """Decode four fixed training clips only, repeat to the actual common batch."""
    import torch
    from .dataset import AHARDataset, load_split_subsets
    from notebooks.utils.diagnostics import registered_learning_check

    data = AHARDataset(
        config.dataset_dir,
        config.sequence_length,
        (config.width, config.height),
        target_fps=config.target_fps,
        sampling_version=config.sampling_version,
    )
    train, _, _, split = load_split_subsets(
        data,
        config.split_manifest,
        seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    labels = sorted({data.samples[i][1] for i in train.indices})[:2]
    groups = [
        [
            i
            for i in sorted(train.indices, key=lambda i: str(data.samples[i][0]))
            if data.samples[i][1] == label
        ][:2]
        for label in labels
    ]
    if len(groups) != 2 or any(len(g) != 2 for g in groups):
        raise ValueError("Transformer check needs two training clips from two classes")
    indices = [groups[c][offset] for offset in range(2) for c in range(2)]
    samples = [data[i] for i in indices]
    # Four unique clips, repeated only to retain the preflight-selected batch.
    repeats = max(1, config.batch_size // 4)
    clips = torch.stack([s[0] for s in samples] * repeats)
    targets = torch.tensor([s[1] for s in samples] * repeats, dtype=torch.long)
    report = dict(
        status="running",
        split=split,
        config=config.to_dict(),
        models={},
        validation_clips_opened=0,
        test_clips_opened=0,
        sources=[
            dict(
                path=str(data.samples[i][0]),
                partition="train",
                sha256=file_hash(data.samples[i][0]),
            )
            for i in indices
        ],
    )
    atomic_json(destination, report)
    try:
        for name in ("swin3d_t", "swin3d_s"):
            result = registered_learning_check(
                name,
                clips,
                targets,
                data.num_classes,
                steps=128,
                batch_size=config.batch_size,
                learning_rate=model_learning_rate(name),
                campaign_batch=True,
            )
            result["passed"] = check_passed(result)
            report["models"][name] = result
            atomic_json(destination, report)
            if not result["passed"]:
                raise RuntimeError(
                    f"{name} failed its train-only check; no comparison/test authorized"
                )
        report["status"] = "passed"
        atomic_json(destination, report)
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        atomic_json(destination, report)
        raise


def ensure_transformer_checks(group, study, config, state, selected):
    """Persist/revalidate diagnostics; never retry a failed check silently."""
    helper = Path(__file__).resolve().parents[1] / "notebooks/utils/diagnostics.py"
    identity = value_hash(
        dict(
            config=config.to_dict(),
            recipe=study["approved_recipe"],
            selection_sha256=file_hash(study["selected_config"]),
            diagnostic_code_sha256=file_hash(helper),
            split_sha256=selected["split"]["manifest_hash"],
        )
    )
    path = Path(group) / "transformer_checks.json"
    if "transformer_checks" in state:
        saved = state["transformer_checks"]
        verify_inventory({str(path): saved["sha256"]})
        if saved["identity"] != identity or read_json(path)["status"] != "passed":
            raise ValueError(
                "Transformer check recipe/evidence changed; comparison/test locked"
            )
        return
    if (
        path.exists()
        or state["jobs"]
        or (Path(group) / "validation_frozen.json").exists()
    ):
        raise ValueError(
            "Missing/failed check receipt; inspect diagnostics, no silent retry"
        )
    result = run_transformer_checks(config, path)
    if result["status"] != "passed" or set(result["models"]) != {
        "swin3d_t",
        "swin3d_s",
    }:
        raise ValueError("Both transformer checks must pass")
    state["transformer_checks"] = dict(identity=identity, sha256=file_hash(path))
    atomic_json(Path(group) / "progress.json", state)
    print(
        "Both train-only transformer checks passed. Discarded probe weights; ten fresh trainings follow.",
        flush=True,
    )
