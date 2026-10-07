"""Predeclared, budgeted diagnostic suite; separate from the controlled AAD CLI."""

import gc
import json
import statistics
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader

from . import config as settings
from src.experiment_config import ExperimentConfig
from src.metrics import evaluate_classifier, rank_validation_results
from src.model import count_trainable_parameters
from .data import data_identity, prepare_data
from .diagnostics import audit_sources, check_deadline, tiny_learnability
from .display import LiveCurves, prediction_examples, show_predictions
from .models import BASELINES, build_model
from .workflows import _read, _sha, configuration, load_checkpoint
from src.train import train_notebook_model
from src.utils import RunContext, write_json

EVIDENCE = "exploratory_diagnostic_not_independent_paper_evidence"


def prepare_experiment_data(root, dataset_root=None):
    """Use suite inputs without changing the preview/single-model defaults."""
    return prepare_data(
        root,
        dataset_root,
        settings.SUITE_FRAME_SIZE,
        sequence_length=settings.SUITE_SEQUENCE_LENGTH,
        target_fps=settings.SUITE_TARGET_FPS,
    )


def candidate_specs(root):
    manifest = _read(Path(root) / "configs/experiments/kinetics_diagnostic_candidates.json")
    specs = {
        c["name"]: dict(
            name="custom",
            layers=c["convlstm_layers"],
            dropout=settings.NOTEBOOK_DROPOUT,
        )
        for c in manifest["candidates"]
    }
    specs.update({name: dict(name=name) for name in BASELINES})
    return specs


def suite_plan(prepared):
    config = configuration(prepared, settings.SUITE_SCREEN_EPOCHS).to_dict()
    plan = dict(
        schema=1,
        evidence_role=EVIDENCE,
        candidates=candidate_specs(prepared["root"]),
        screen_config=config,
        data_identity=data_identity(prepared),
        scheduler=dict(
            factor=settings.LR_FACTOR,
            patience=settings.LR_PATIENCE,
            min_lr=settings.MIN_LR,
        ),
        coverage_fps=list(settings.TEMPORAL_FPS),
        dropout_trials=[0.0, 0.5],
        weight_decay_trials=[0.0, 0.001],
        spatial_sizes=list(settings.SUITE_SPATIAL_SIZES),
        seeds=list(settings.CONFIRMATION_SEEDS),
        final_epochs=settings.SUITE_FINAL_EPOCHS,
        final_size=settings.SUITE_FINAL_FRAME_SIZE,
        hours=settings.SUITE_HOURS,
        paper_minutes=settings.PAPER_MAX_MINUTES,
        selection="validation macro-F1, accuracy, parameter count, name; never test",
        confirmation="selected custom plus all three practical baselines, identical inputs/budgets, both seeds",
        paper="separate native 50-frame/50x50 topology; dataset-specific head; not a faithful training-protocol reproduction",
    )
    reference = dict(
        config=config,
        model_spec=dict(
            name="custom",
            layers=config["convlstm_layers"],
            dropout=settings.NOTEBOOK_DROPOUT,
        ),
    )
    counts = dict(
        screen=len(plan["candidates"]),
        one_factor=len(ablation_trials(plan, reference)),
        native_attempt=1,
        confirmation=(1 + len(BASELINES)) * len(plan["seeds"]),
    )
    plan["run_counts"] = dict(counts, total=sum(counts.values()), tiny_checks=1)
    return plan


def show_experiment_plan(prepared):
    plan = suite_plan(prepared)
    config = plan["screen_config"]
    rows = []
    for name, spec in plan["candidates"].items():
        with torch.device("meta"):
            model = build_model(
                spec,
                prepared["dataset"].num_classes,
                (3, config["height"], config["width"]),
                config["sequence_length"],
            )
        rows.append(
            dict(
                candidate=name,
                parameters=count_trainable_parameters(model),
                frames=config["sequence_length"],
                fps=plan["data_identity"]["target_fps"],
                resolution=config["height"],
                epochs=config["epochs"],
            )
        )
    display(pd.DataFrame(rows))
    reference = dict(
        config=config,
        model_spec=dict(
            name="custom",
            layers=config["convlstm_layers"],
            dropout=settings.NOTEBOOK_DROPOUT,
        ),
    )
    variants = ablation_trials(plan, reference)
    display(
        pd.DataFrame(
            [
                dict(
                    name=v["name"],
                    resolution=v["config"]["height"],
                    fps=v["fps"],
                    weight_decay=v["config"]["weight_decay"],
                    dropout=v["model"].get("dropout"),
                    epochs=v["config"]["epochs"],
                )
                for v in variants
            ]
        )
    )
    confirmation_count = (1 + len(BASELINES)) * len(plan["seeds"])
    print(
        f"Planned training runs: {len(rows)} screen + {len(variants)} one-factor "
        f"+ 1 separate native attempt + {confirmation_count} confirmation = "
        f"{len(rows) + len(variants) + 1 + confirmation_count}; plus the tiny check. "
        "The compute deadline may leave this plan incomplete."
    )
    print(
        f"Audit → tiny overfit → {len(rows)} candidates → one-factor comparisons → separate native paper model → two-seed confirmation → frozen exploratory test."
    )
    print(
        f"Compute cap: {plan['hours']} hours (checked between batches). No accuracy guarantee; exploratory evidence only."
    )
    return plan


def _data(prepared, config, fps):
    view = prepare_data(
        prepared["root"],
        prepared["dataset"].dataset_dir,
        config.height,
        sequence_length=config.sequence_length,
        target_fps=fps,
    )
    if prepared.get("suite_run_dir") is not None:
        view["suite_run_dir"] = prepared["suite_run_dir"]
        view["suite_train_run_dirs"] = prepared["suite_train_run_dirs"]
    expected = dict(
        data_identity(prepared),
        frame_size=[config.height, config.width],
        sequence_length=config.sequence_length,
        target_fps=fps,
    )
    if data_identity(view) != expected:
        raise ValueError("Input view changed source, labels, preprocessing or split")
    return view


def _trial(prepared, name, spec, config, deadline):
    check_deadline(deadline)
    result = train_notebook_model(
        prepared, config, name, LiveCurves(), model_spec=spec, deadline=deadline
    )
    result["checkpoint_sha256"] = _sha(result["selected_checkpoint"])
    return result


def _table(results, path):
    rows = [
        dict(
            name=r["name"],
            partition="validation",
            **r["validation_metrics"],
            parameters=r["num_params"],
            seconds=r["training_seconds"],
            seed=r["config"]["seed"],
            fps=r["data_identity"]["target_fps"],
            frames=r["config"]["sequence_length"],
            resolution=r["config"]["height"],
            weight_decay=r["config"]["weight_decay"],
            model=json.dumps(r["model_spec"], sort_keys=True),
        )
        for r in results
    ]
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    display(table)


def _verify_result(result):
    run = _read(Path(result["run_dir"]) / "run.json")
    original = {k: v for k, v in result.items() if k != "checkpoint_sha256"}
    if (
        run["status"] != "complete"
        or run["results"] != original
        or result["partition"] != "validation"
        or result["checkpoint_sha256"] != _sha(result["selected_checkpoint"])
    ):
        raise ValueError(
            "Changed/incomplete trial evidence; cannot select or open test"
        )


def ablation_trials(plan, custom):
    """Derive one-factor trials solely from the frozen plan and screen reference."""
    base = ExperimentConfig.from_mapping(custom["config"])
    base_fps = plan["data_identity"]["target_fps"]
    trials = []
    for fps in plan["coverage_fps"]:
        if fps != base_fps:
            trials.append((f"coverage-{fps}fps", custom["model_spec"], base, fps))
    for dropout in plan["dropout_trials"]:
        trials.append(
            (
                f"dropout-{dropout}",
                dict(custom["model_spec"], dropout=dropout),
                base,
                base_fps,
            )
        )
    for decay in plan["weight_decay_trials"]:
        trials.append(
            (
                f"weight-decay-{decay}",
                custom["model_spec"],
                replace(base, weight_decay=decay),
                base_fps,
            )
        )
    for size in plan["spatial_sizes"]:
        if size != base.height:
            trials.append(
                (
                    f"spatial-{size}",
                    custom["model_spec"],
                    replace(base, height=size, width=size),
                    base_fps,
                )
            )
    return [dict(name=n, model=s, config=c.to_dict(), fps=f) for n, s, c, f in trials]


def _native_trial(prepared, plan, directory, deadline):
    """Never shrink the native topology to fit. Resource limits are explicit."""
    config = replace(
        ExperimentConfig.from_mapping(plan["screen_config"]),
        sequence_length=50,
        height=50,
        width=50,
    )
    spec = dict(name="paper_convlstm_published")
    with torch.device("meta"):
        count = count_trainable_parameters(
            build_model(spec, prepared["dataset"].num_classes, (3, 50, 50), 50)
        )
    # Conservative Adam/gradient/activation allowance, not a promise of peak memory.
    estimated = count * 32 + config.batch_size * 50 * 64 * 50 * 50 * 4 * 12
    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    record = dict(
        status="resource_limited",
        parameters=count,
        estimated_bytes=estimated,
        free_cuda_bytes=free,
        comparable_to_screen=False,
        note=plan["paper"],
        config=config.to_dict(),
    )
    native_deadline = min(deadline - 3600, time.time() + plan["paper_minutes"] * 60)
    if free < estimated or native_deadline <= time.time():
        record["reason"] = (
            "Native topology fails memory/time preflight; not substituted"
        )
    else:
        try:
            result = _trial(
                _data(prepared, config, plan["data_identity"]["target_fps"]),
                "paper-native",
                spec,
                config,
                native_deadline,
            )
            record.update(status="complete", result=result)
        except (torch.cuda.OutOfMemoryError, TimeoutError) as error:
            record["reason"] = str(error)
            gc.collect()
            torch.cuda.empty_cache()
    write_json(directory / "native_paper.json", record)
    print(f"Native paper topology: {record['status']}", flush=True)
    return record


def run_screen(prepared):
    plan = suite_plan(prepared)
    if (
        plan["final_epochs"] <= plan["screen_config"]["epochs"]
        or plan["final_size"] <= plan["screen_config"]["height"]
    ):
        raise ValueError(
            "Confirmation must be longer and spatially finer than screening"
        )
    context = RunContext(
        prepared["root"] / "runs",
        "experiments",
        prepared["dataset"].dataset_dir,
        "learning-temporal-suite",
        plan,
        {"evidence_role": EVIDENCE},
    )
    directory = context.run_dir
    prepared["suite_run_dir"] = directory
    prepared["suite_train_run_dirs"] = []
    deadline = time.time() + plan["hours"] * 3600
    write_json(directory / "plan.json", plan)
    write_json(directory / "budget.json", dict(deadline=deadline, started=time.time()))
    print(f"Suite evidence: {directory}", flush=True)
    results = []
    try:
        audit_sources(prepared, directory / "source_audit.json", deadline)
        tiny_learnability(prepared, directory / "tiny.json", deadline)
        for name, spec in plan["candidates"].items():
            config = ExperimentConfig.from_mapping(plan["screen_config"])
            if spec["name"] == "custom":
                config = replace(
                    config,
                    convlstm_layers=tuple((n, tuple(k)) for n, k in spec["layers"]),
                )
            print(
                f"Screen {len(results)+1}/{len(plan['candidates'])}: {name}", flush=True
            )
            results.append(_trial(prepared, name, spec, config, deadline))
            write_json(directory / "screen_progress.json", results)
            _table(results, directory / "screen.csv")
        ranked = rank_validation_results(results)
        custom = next(r for r in ranked if r["model_spec"]["name"] == "custom")
        variants = ablation_trials(plan, custom)
        write_json(directory / "ablation_plan.json", variants)
        ablations = [custom]
        for variant in variants:
            config = ExperimentConfig.from_mapping(variant["config"])
            trial_data = _data(prepared, config, variant["fps"])
            ablations.append(
                _trial(trial_data, variant["name"], variant["model"], config, deadline)
            )
            write_json(directory / "ablations.json", ablations)
            _table(ablations, directory / "ablations.csv")
        reference = rank_validation_results(ablations)[0]
        native = _native_trial(prepared, plan, directory, deadline)
        write_json(
            directory / "screen.json",
            dict(
                ranked=ranked,
                reference=reference,
                ablations=ablations,
                plan_sha256=_sha(directory / "plan.json"),
                native_status=native["status"],
            ),
        )
        context.complete(
            {"screen": str(directory / "screen.json")}, {"reference": reference["name"]}
        )
        return directory
    except BaseException as error:
        context.update({"status": "partial", "error": str(error)})
        print(
            f"Partial evidence retained at {directory}; no frozen winner/test.",
            flush=True,
        )
        raise


def create_suite_artifact_archive(prepared, directory=None):
    """Bundle the current suite and its registered model-run artifacts."""
    root = Path(prepared["root"]).resolve()
    runs_root = (root / "runs").resolve()
    suite_dir = Path(
        directory or prepared.get("suite_run_dir") or ""
    ).resolve()
    experiments_root = (runs_root / "experiments").resolve()
    if not suite_dir.is_relative_to(experiments_root) or not suite_dir.is_dir():
        raise ValueError("A current experiment-suite directory is required")

    run_dirs = [suite_dir]
    train_root = (runs_root / "train").resolve()
    for run_dir in prepared.get("suite_train_run_dirs", []):
        candidate = Path(run_dir)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate.is_relative_to(train_root) and candidate.is_dir():
            run_dirs.append(candidate)

    archive_path = runs_root / "exports" / f"{suite_dir.name}_artifacts.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for run_dir in run_dirs:
            for artifact in sorted(run_dir.rglob("*")):
                if artifact.is_file():
                    archive.write(artifact, artifact.relative_to(root))
    return archive_path


def download_suite_artifacts(prepared, directory=None):
    """Create one suite ZIP and trigger its Colab browser download."""
    archive_path = create_suite_artifact_archive(prepared, directory)
    try:
        from google.colab import files
    except ImportError:
        print(f"Artifact archive saved at {archive_path}; Colab download unavailable.")
    else:
        files.download(str(archive_path))
    return archive_path


def checked_screen(prepared, directory):
    directory = Path(directory)
    plan, screen = _read(directory / "plan.json"), _read(directory / "screen.json")
    if (
        _read(directory / "run.json")["status"] != "complete"
        or screen["plan_sha256"] != _sha(directory / "plan.json")
        or data_identity(prepared) != plan["data_identity"]
    ):
        raise ValueError("Screen plan/data is changed or incomplete")
    if len(screen["ranked"]) != len(plan["candidates"]) or {
        r["name"] for r in screen["ranked"]
    } != set(plan["candidates"]):
        raise ValueError("Every declared screen candidate must complete")
    if screen["ranked"] != rank_validation_results(screen["ranked"]):
        raise ValueError("Screen ranking must use validation only")
    expected_ablations = {r["name"] for r in _read(directory / "ablation_plan.json")}
    custom = next(r for r in screen["ranked"] if r["model_spec"]["name"] == "custom")
    declared = ablation_trials(plan, custom)
    if _read(directory / "ablation_plan.json") != declared:
        raise ValueError("One-factor ablation plan changed")
    for result in screen["ranked"]:
        spec = plan["candidates"][result["name"]]
        config = ExperimentConfig.from_mapping(plan["screen_config"])
        if spec["name"] == "custom":
            config = replace(
                config, convlstm_layers=tuple((n, tuple(k)) for n, k in spec["layers"])
            )
        if (
            result["config"] != config.to_dict()
            or result["model_spec"] != spec
            or result["data_identity"] != plan["data_identity"]
            or result["actual_epochs"] != config.epochs
        ):
            raise ValueError("Candidate did not follow the declared matched protocol")
    if {r["name"] for r in screen["ablations"]} != expected_ablations | {
        custom["name"]
    }:
        raise ValueError("Every declared ablation must complete")
    for result in screen["ranked"] + screen["ablations"]:
        _verify_result(result)
    by_name = {r["name"]: r for r in screen["ablations"]}
    if by_name[custom["name"]] != custom:
        raise ValueError("Ablation reference differs from screen")
    for variant in declared:
        result = by_name[variant["name"]]
        identity = dict(
            plan["data_identity"],
            target_fps=variant["fps"],
            frame_size=[variant["config"]["height"], variant["config"]["width"]],
        )
        if (
            result["config"] != variant["config"]
            or result["model_spec"] != variant["model"]
            or result["data_identity"] != identity
            or result["actual_epochs"] != variant["config"]["epochs"]
        ):
            raise ValueError("Ablation changed more than its declared factor")
    if screen["reference"] != rank_validation_results(screen["ablations"])[0]:
        raise ValueError("Reference must come from validation-only ablation ranking")
    native = _read(directory / "native_paper.json")
    if (
        native["status"] not in ("complete", "resource_limited")
        or native["status"] != screen["native_status"]
    ):
        raise ValueError("Native topology status is missing or inconsistent")
    return plan, screen


def confirmation_ranking(results):
    groups = {}
    for result in results:
        group = result["name"].rsplit("-seed", 1)[0]
        groups.setdefault(group, []).append(result)
    rows = []
    for group, runs in groups.items():
        row = dict(name=group, parameters=runs[0]["num_params"])
        for key in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1"):
            values = [r["validation_metrics"][key] for r in runs]
            row[key] = statistics.mean(values)
            row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return sorted(
        rows, key=lambda r: (-r["macro_f1"], -r["accuracy"], r["parameters"], r["name"])
    )


def confirmation_configuration(plan, reference):
    """Declare longer matched confirmation without shrinking the selected input."""
    return replace(
        ExperimentConfig.from_mapping(reference["config"]),
        height=max(plan["final_size"], reference["config"]["height"]),
        width=max(plan["final_size"], reference["config"]["width"]),
        epochs=plan["final_epochs"],
        early_stopping_patience=plan["final_epochs"],
    )


def train_winner(prepared, directory):
    directory = Path(directory)
    if (directory / "frozen.json").exists():
        raise ValueError(
            "Already frozen; reuse final_evaluate, do not retrain after test"
        )
    plan, screen = checked_screen(prepared, directory)
    deadline = _read(directory / "budget.json")["deadline"]
    reference = screen["reference"]
    base = confirmation_configuration(plan, reference)
    fine = _data(prepared, base, reference["data_identity"]["target_fps"])
    specs = dict(
        custom_selected=reference["model_spec"], **{n: dict(name=n) for n in BASELINES}
    )
    expected = {
        f"{name}-seed{seed}": dict(
            model=spec, config=replace(base, seed=seed).to_dict()
        )
        for name, spec in specs.items()
        for seed in plan["seeds"]
    }
    write_json(
        directory / "confirmation_plan.json",
        dict(
            trials=expected,
            data_identity=data_identity(fine),
            screen_sha256=_sha(directory / "screen.json"),
        ),
    )
    results = []
    try:
        for name, trial in expected.items():
            results.append(
                _trial(
                    fine,
                    name,
                    trial["model"],
                    ExperimentConfig.from_mapping(trial["config"]),
                    deadline,
                )
            )
            write_json(directory / "confirmation_progress.json", results)
            _table(results, directory / "confirmation.csv")
        ranking = confirmation_ranking(results)
        pd.DataFrame(ranking).to_csv(
            directory / "confirmation_mean_std.csv", index=False
        )
        display(pd.DataFrame(ranking))
        write_json(
            directory / "frozen.json",
            dict(
                results=results,
                ranking=ranking,
                winner=ranking[0]["name"],
                plan_sha256=_sha(directory / "confirmation_plan.json"),
            ),
        )
        print(
            f"Frozen {ranking[0]['name']} using mean validation metrics across both seeds. Test not opened."
        )
        return fine
    except BaseException as error:
        write_json(
            directory / "confirmation_partial.json",
            dict(error=str(error), completed=len(results)),
        )
        raise


def _checked_confirmation(prepared, directory):
    plan = _read(directory / "plan.json")
    # Reconstruct the original screen's metadata without decoding any videos.
    base = _data(
        prepared,
        ExperimentConfig.from_mapping(plan["screen_config"]),
        plan["data_identity"]["target_fps"],
    )
    _, screen = checked_screen(base, directory)
    confirmation, frozen = _read(directory / "confirmation_plan.json"), _read(
        directory / "frozen.json"
    )
    if (
        confirmation["screen_sha256"] != _sha(directory / "screen.json")
        or frozen["plan_sha256"] != _sha(directory / "confirmation_plan.json")
        or data_identity(prepared) != confirmation["data_identity"]
    ):
        raise ValueError("Confirmation evidence/data changed")
    results = frozen["results"]
    reference = screen["reference"]
    config = confirmation_configuration(plan, reference)
    specs = dict(
        custom_selected=reference["model_spec"], **{n: dict(name=n) for n in BASELINES}
    )
    expected_plan = {
        f"{name}-seed{seed}": dict(
            model=spec, config=replace(config, seed=seed).to_dict()
        )
        for name, spec in specs.items()
        for seed in plan["seeds"]
    }
    if confirmation["trials"] != expected_plan:
        raise ValueError(
            "Confirmation differs from the declared models, seeds or budget"
        )
    if len(results) != len(confirmation["trials"]) or {
        r["name"] for r in results
    } != set(confirmation["trials"]):
        raise ValueError("Both seeds of every confirmation model must complete")
    for result in results:
        _verify_result(result)
        expected = confirmation["trials"][result["name"]]
        if (
            result["config"] != expected["config"]
            or result["model_spec"] != expected["model"]
            or result["actual_epochs"] != plan["final_epochs"]
            or result["data_identity"] != confirmation["data_identity"]
        ):
            raise ValueError(
                "Final checkpoint does not match complete longer/finer confirmation"
            )
    ranking = confirmation_ranking(results)
    if ranking != frozen["ranking"] or frozen["winner"] != ranking[0]["name"]:
        raise ValueError("Frozen winner must use mean validation evidence only")
    return frozen


def final_evaluate(prepared, directory):
    """Score all frozen confirmations, never rerank from test. Persist before display."""
    directory = Path(directory)
    frozen = _checked_confirmation(prepared, directory)
    fingerprint = _sha(directory / "frozen.json")
    report_path = directory / "final_test.json"
    if report_path.exists():
        report = _read(report_path)
        if report["frozen_sha256"] != fingerprint:
            raise ValueError("Saved report belongs to different frozen models")
        display(pd.DataFrame(report["rows"]))
        show_predictions(report["examples"])
        return report
    deadline = _read(directory / "budget.json")["deadline"]
    saved_path = directory / "final_test_metrics.json"
    saved = (
        _read(saved_path)
        if saved_path.exists()
        else dict(frozen_sha256=fingerprint, rows=[])
    )
    if saved["frozen_sha256"] != fingerprint:
        raise ValueError("Saved test metrics belong to different frozen models")
    dataset = prepared["dataset"]
    old_allowed = dataset.allowed_indices.copy()
    dataset.training_windows = False
    dataset.allowed_indices = set(prepared["test"].indices)
    rows = saved["rows"]
    try:
        for result in frozen["results"]:
            if any(r["name"] == result["name"] for r in rows):
                continue
            check_deadline(deadline)
            model, _ = load_checkpoint(prepared, result["selected_checkpoint"])
            metrics = evaluate_classifier(
                model,
                DataLoader(prepared["test"], batch_size=settings.BATCH_SIZE),
                torch.nn.CrossEntropyLoss(),
                next(model.parameters()).device,
                dataset.num_classes,
                lambda *args: check_deadline(deadline),
            )
            rows.append(
                dict(
                    name=result["name"],
                    partition="test",
                    evidence_role=EVIDENCE,
                    **metrics,
                )
            )
            write_json(saved_path, dict(frozen_sha256=fingerprint, rows=rows))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Seed 42 is predeclared for examples, never chosen by its test score.
        example_name = frozen["winner"] + "-seed42"
        selected = next(r for r in frozen["results"] if r["name"] == example_name)
        check_deadline(deadline)
        model, _ = load_checkpoint(prepared, selected["selected_checkpoint"])
        examples = prediction_examples(
            model,
            prepared,
            "test",
            directory / "test_predictions",
            render=False,
            before_sample=lambda: check_deadline(deadline),
        )
        report = dict(
            frozen_sha256=fingerprint,
            rows=rows,
            examples=examples,
            winner=frozen["winner"],
            evidence_role=EVIDENCE,
            limitation="Diagnostic split, not independent paper evidence; prior Kinetics-400/VDD test feedback must not be treated as a fresh holdout",
        )
        write_json(report_path, report)
        pd.DataFrame(rows).to_csv(directory / "test_exploratory.csv", index=False)
    finally:
        dataset.allowed_indices = old_allowed
    display(pd.DataFrame(rows))
    show_predictions(report["examples"])
    return report
