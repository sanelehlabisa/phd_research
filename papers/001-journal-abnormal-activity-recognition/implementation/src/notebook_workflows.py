"""Independent selected-dataset workflows; controlled AAD runners stay separate."""

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader

from . import notebook_config as settings
from .experiment_config import ExperimentConfig
from .metrics import (
    evaluate_classifier,
    rank_validation_results,
    validate_selected_checkpoint,
)
from .model import (
    CustomConvLSTM,
    count_trainable_parameters,
    custom_model_from_checkpoint,
)
from .notebook_data import data_identity, prepare_data
from .notebook_display import LiveCurves, prediction_examples, show_predictions
from .train import train_notebook_model
from .utils import RunContext, seed_everything, write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def configuration(prepared, epochs, layers=None):
    dataset = prepared["dataset"]
    return ExperimentConfig(
        dataset_dir=str(dataset.dataset_dir),
        runs_dir=str(prepared["root"] / "runs"),
        split_manifest=str(prepared["manifest_path"]),
        seed=settings.SEED,
        sequence_length=dataset.sequence_length,
        height=dataset.frame_size[0],
        width=dataset.frame_size[1],
        epochs=epochs,
        early_stopping_patience=epochs,
        batch_size=settings.BATCH_SIZE,
        convlstm_layers=layers or settings.DEFAULT_LAYERS,
    )


def inspect_model(prepared):
    seed_everything(settings.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        CustomConvLSTM(
            num_classes=prepared["dataset"].num_classes,
            layers=list(settings.DEFAULT_LAYERS),
        )
        .to(device)
        .eval()
    )
    print(model)
    display(pd.DataFrame(model.configuration()["layers"]))
    print(
        f"Input: (batch, {prepared['dataset'].sequence_length}, 3, "
        f"{prepared['dataset'].frame_size}) | parameters: {count_trainable_parameters(model):,}"
    )
    print("Random weights: this prediction checks wiring, not learned accuracy.")
    run = RunContext(
        prepared["root"] / "runs",
        "model",
        prepared["dataset"].dataset_dir,
        "random-inspection",
        model.configuration(),
        {
            "data_identity": data_identity(prepared),
            "evidence_role": "random_weight_check",
        },
    )
    records = prediction_examples(model, prepared, "train", run.run_dir, count=1)
    run.complete({"predictions": str(run.run_dir / "predictions.json")})
    return records


def load_checkpoint(prepared, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_selected_checkpoint(
        checkpoint,
        prepared["dataset"].dataset_dir.resolve().name,
        prepared["split"]["manifest_hash"],
    )
    if checkpoint.get("data_identity") != data_identity(prepared):
        raise ValueError(
            "Checkpoint dataset, classes or preprocessing are incompatible"
        )
    model = custom_model_from_checkpoint(checkpoint)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, checkpoint


def train_single(prepared):
    return train_notebook_model(
        prepared,
        configuration(prepared, settings.TRAIN_EPOCHS),
        "single-model",
        LiveCurves(),
    )


def show_validation_predictions(prepared, result):
    model, _ = load_checkpoint(prepared, result["selected_checkpoint"])
    return prediction_examples(
        model, prepared, "validation", Path(result["run_dir"]) / "predictions"
    )


def screen_configurations(prepared):
    """Declare every combination of the finite experiment lists before training."""
    reference = replace(
        configuration(prepared, settings.SCREEN_EPOCHS),
        batch_size=settings.SCREEN_BATCH_SIZE,
    )
    choices = {
        "models": settings.SCREEN_MODEL_NAMES,
        "frame sizes": settings.SCREEN_FRAME_SIZES,
        "frame counts": settings.SCREEN_SEQUENCE_LENGTHS,
        "weight decay": settings.SCREEN_WEIGHT_DECAYS,
        "augmentation": settings.SCREEN_AUGMENT_OPTIONS,
    }
    for label, values in choices.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Declare a nonempty list of distinct {label} values")
    if not settings.SCREEN_CANDIDATES:
        raise ValueError("Declare at least one model architecture")
    configs = {}
    unknown_models = set(settings.SCREEN_MODEL_NAMES) - set(settings.SCREEN_CANDIDATES)
    if unknown_models:
        raise ValueError(f"Unknown model names: {sorted(unknown_models)}")
    combinations = product(*choices.values())
    for model_name, size, frames, weight_decay, augment in combinations:
        config = replace(
            reference,
            convlstm_layers=settings.SCREEN_CANDIDATES[model_name],
            height=size,
            width=size,
            sequence_length=frames,
            weight_decay=weight_decay,
            augment=augment,
        )
        decay_label = str(weight_decay).replace(".", "p").replace("-", "m")
        name = f"{model_name}_s{size}_t{frames}_wd{decay_label}_aug{int(augment)}"
        if name in configs:
            raise ValueError(f"Duplicate experiment name: {name}")
        configs[name] = config
    return configs


def show_experiment_plan(prepared):
    rows = []
    for name, config in screen_configurations(prepared).items():
        rows.append(
            dict(
                candidate=name,
                layers=config.to_dict()["convlstm_layers"],
                epochs=config.epochs,
                resolution=f"{config.height}x{config.width}",
                frames=config.sequence_length,
                batch=config.batch_size,
                learning_rate=config.learning_rate,
                augment=config.augment,
                weight_decay=config.weight_decay,
                seed=config.seed,
            )
        )
    display(pd.DataFrame(rows))
    print(
        f"{len(settings.SCREEN_MODEL_NAMES)} architectures x "
        f"{len(settings.SCREEN_FRAME_SIZES)} frame sizes x "
        f"{len(settings.SCREEN_SEQUENCE_LENGTHS)} frame counts x "
        f"{len(settings.SCREEN_WEIGHT_DECAYS)} weight-decay values x "
        f"{len(settings.SCREEN_AUGMENT_OPTIONS)} augmentation options.",
        flush=True,
    )
    print(
        f"{len(rows)} runs x {settings.SCREEN_EPOCHS} epochs = "
        f"{len(rows) * settings.SCREEN_EPOCHS} screening epochs in total. "
        f"Every run uses all {len(prepared['train'])} training and "
        f"{len(prepared['validation'])} validation clips; no augmented copies.",
        flush=True,
    )
    print(
        f"Then retrain the validation winner from scratch: {settings.FINAL_EPOCHS} epochs, "
        "keeping its selected input size, frame count, augmentation and weight decay; "
        "freeze before final test."
    )
    print("Exploratory single-seed diagnostic, not the controlled AAD paper screen.")
    return rows


def run_screen(prepared):
    configs = screen_configurations(prepared)
    if not configs or settings.FINAL_EPOCHS <= settings.SCREEN_EPOCHS:
        raise ValueError("Declare candidates and a longer final training budget")
    plan = dict(
        candidates={name: config.to_dict() for name, config in configs.items()},
        candidate_data_identities={},
        final_epochs=settings.FINAL_EPOCHS,
        final_input_policy="selected",
        data_identity=data_identity(prepared),
    )
    views = {}
    for name, config in configs.items():
        if config.height != config.width:
            raise ValueError("Notebook experiments require square frames")
        key = (config.height, config.sequence_length)
        if key not in views:
            views[key] = prepare_data(
                prepared["root"],
                prepared["dataset"].dataset_dir,
                frame_size=config.height,
                sequence_length=config.sequence_length,
            )
        identity = data_identity(views[key])
        for field, value in plan["data_identity"].items():
            if (
                field not in ("frame_size", "sequence_length")
                and identity[field] != value
            ):
                raise ValueError(
                    "Experiments must reuse the same dataset, split and FPS"
                )
        plan["candidate_data_identities"][name] = identity
    run = RunContext(
        prepared["root"] / "runs",
        "experiments",
        prepared["dataset"].dataset_dir,
        "diagnostic-screen",
        {name: config.to_dict() for name, config in configs.items()},
        {
            "evidence_role": "exploratory_diagnostic",
            "data_identity": data_identity(prepared),
        },
    )
    write_json(run.run_dir / "plan.json", plan)
    results = []
    try:
        for number, (name, config) in enumerate(configs.items(), 1):
            print(f"Candidate {number}/{len(configs)}: {name}", flush=True)
            view = views[(config.height, config.sequence_length)]
            result = train_notebook_model(view, config, name, LiveCurves())
            result["checkpoint_sha256"] = _sha(result["selected_checkpoint"])
            results.append(result)
            write_json(run.run_dir / "screen_progress.json", results)
        ranked = rank_validation_results(results)
        write_json(
            run.run_dir / "screen.json",
            {"ranked": ranked, "plan_sha256": _sha(run.run_dir / "plan.json")},
        )
        run.complete(
            {"screen": str(run.run_dir / "screen.json")}, {"winner": ranked[0]["name"]}
        )
        display(
            pd.DataFrame(
                [
                    dict(
                        candidate=r["name"],
                        resolution=f"{r['config']['height']}x{r['config']['width']}",
                        frames=r["config"]["sequence_length"],
                        augment=r["config"]["augment"],
                        weight_decay=r["config"]["weight_decay"],
                        epochs=r["actual_epochs"],
                        **r["validation_metrics"],
                        parameters=r["num_params"],
                    )
                    for r in ranked
                ]
            )
        )
        return run.run_dir
    except BaseException as error:
        run.update({"status": "failed", "error": str(error)})
        raise


def checked_winner(screen_dir):
    """Reject partial, changed or mixed-data screens before confirmation or test."""
    screen_dir = Path(screen_dir)
    plan = _read(screen_dir / "plan.json")
    screen = _read(screen_dir / "screen.json")
    if _read(screen_dir / "run.json")["status"] != "complete" or screen[
        "plan_sha256"
    ] != _sha(screen_dir / "plan.json"):
        raise ValueError("Screen is incomplete or its declared plan changed")
    ranked = screen["ranked"]
    if len(ranked) != len(plan["candidates"]) or {r["name"] for r in ranked} != set(
        plan["candidates"]
    ):
        raise ValueError("Every declared candidate must complete")
    if ranked != rank_validation_results(ranked):
        raise ValueError("Winner must come from the validation ranking")
    for result in ranked:
        manifest = _read(Path(result["run_dir"]) / "run.json")
        expected_identity = plan.get("candidate_data_identities", {}).get(
            result["name"], plan["data_identity"]
        )
        if (
            result["partition"] != "validation"
            or manifest["status"] != "complete"
            or result["config"] != plan["candidates"][result["name"]]
            or result["data_identity"] != expected_identity
            or result["validation_metrics"] != manifest["results"]["validation_metrics"]
            or result["checkpoint_sha256"] != _sha(result["selected_checkpoint"])
        ):
            raise ValueError("Candidate provenance does not match the completed screen")
    return ranked[0], plan


def final_training_configuration(winner, plan):
    """Retain the grid winner's inputs, with compatibility for older saved plans."""
    config = ExperimentConfig.from_mapping(winner["config"])
    if plan.get("final_input_policy") == "selected":
        return replace(
            config,
            epochs=plan["final_epochs"],
            early_stopping_patience=plan["final_epochs"],
        )
    return replace(
        config,
        height=plan["final_frame_size"],
        width=plan["final_frame_size"],
        sequence_length=plan.get("final_sequence_length", config.sequence_length),
        epochs=plan["final_epochs"],
        early_stopping_patience=plan["final_epochs"],
    )


def train_winner(prepared, screen_dir):
    screen_dir = Path(screen_dir)
    if (screen_dir / "frozen.json").exists():
        raise ValueError(
            "Winner already frozen; reuse final_evaluate, do not retrain after test"
        )
    winner, plan = checked_winner(screen_dir)
    if data_identity(prepared) != plan["data_identity"]:
        raise ValueError("Selected dataset changed after screening")
    config = final_training_configuration(winner, plan)
    fine = prepare_data(
        prepared["root"],
        prepared["dataset"].dataset_dir,
        config.height,
        sequence_length=config.sequence_length,
    )
    result = train_notebook_model(
        fine, config, winner["name"] + "-longer", LiveCurves()
    )
    write_json(
        screen_dir / "frozen.json",
        dict(
            winner=winner["name"],
            screen_sha256=_sha(screen_dir / "screen.json"),
            checkpoint_sha256=_sha(result["selected_checkpoint"]),
            result=result,
        ),
    )
    print(
        f"Frozen {winner['name']}; checkpoint selected by validation loss. Test not opened yet."
    )
    return fine


def final_evaluate(prepared, screen_dir):
    """Only the frozen, completed longer-trained winner can open the test split."""
    screen_dir = Path(screen_dir)
    winner, plan = checked_winner(screen_dir)
    frozen = _read(screen_dir / "frozen.json")
    result = frozen["result"]
    run = _read(Path(result["run_dir"]) / "run.json")
    config = result["config"]
    expected = final_training_configuration(winner, plan).to_dict()
    if (
        frozen["winner"] != winner["name"]
        or config != expected
        or frozen["screen_sha256"] != _sha(screen_dir / "screen.json")
        or frozen["checkpoint_sha256"] != _sha(result["selected_checkpoint"])
        or run["status"] != "complete"
        or run["results"] != result
        or result["actual_epochs"] <= winner["actual_epochs"]
    ):
        raise ValueError(
            "Final test requires a compatible completed, longer-trained frozen winner"
        )
    model, checkpoint = load_checkpoint(prepared, result["selected_checkpoint"])
    if checkpoint["experiment_config"] != config or result[
        "data_identity"
    ] != data_identity(prepared):
        raise ValueError("Final checkpoint does not match the frozen configuration")
    report_path = screen_dir / "final_test.json"
    if report_path.exists():
        report = _read(report_path)
        if report["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
            raise ValueError("Saved test report belongs to a different checkpoint")
        print("Reusing saved final evaluation; no test clips reopened.")
        display(pd.DataFrame([report["metrics"]]))
        show_predictions(report["examples"])
        return report
    dataset = prepared["dataset"]
    allowed_before = dataset.allowed_indices.copy()
    dataset.allowed_indices = set(prepared["test"].indices)
    try:
        metrics_path = screen_dir / "final_test_metrics.json"
        if metrics_path.exists():
            saved_metrics = _read(metrics_path)
            if saved_metrics["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
                raise ValueError("Saved test metrics belong to a different checkpoint")
            metrics = saved_metrics["metrics"]
            print("Reusing saved test metrics; recovering prediction export only.")
        else:
            print(
                f"Final evaluation: {len(prepared['test'])} test clips; model is frozen.",
                flush=True,
            )
            metrics = evaluate_classifier(
                model,
                DataLoader(prepared["test"], batch_size=config["batch_size"]),
                torch.nn.CrossEntropyLoss(),
                next(model.parameters()).device,
                dataset.num_classes,
                lambda done, total, loss: print(f"  Test: {done}/{total}", flush=True),
            )
            write_json(
                metrics_path,
                dict(
                    partition="test",
                    checkpoint_sha256=frozen["checkpoint_sha256"],
                    metrics=metrics,
                ),
            )
        examples = prediction_examples(
            model, prepared, "test", screen_dir / "test_predictions", render=False
        )
        report = dict(
            partition="test",
            evidence_role="exploratory_diagnostic",
            checkpoint_sha256=frozen["checkpoint_sha256"],
            metrics=metrics,
            examples=examples,
        )
        write_json(report_path, report)
    finally:
        dataset.allowed_indices = allowed_before
    # Persist evidence and re-lock raw clips before invoking frontend display.
    display(pd.DataFrame([report["metrics"]]))
    show_predictions(report["examples"])
    return report
