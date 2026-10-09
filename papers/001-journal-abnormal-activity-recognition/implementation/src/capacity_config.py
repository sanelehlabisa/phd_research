"""Versioned validation-only capacity (069) and focused-shape (073) decisions."""

from copy import deepcopy
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import statistics

from .experiment_config import CandidateManifest, ExperimentConfig
from .study_matrix import rank_resolutions, value_hash, validation_accuracy

REFERENCES = ("r3d_18", "swin3d_t")
STAGE_LIMITS = {
    "calibration": 12,
    "flat": 63,
    "references": 6,
    "refinement": 18,
    "confirmation": 24,
    "weight_decay": 18,
    "temporal": 15,
}
SHAPE_LIMITS = {**STAGE_LIMITS, "flat": 48, "refinement": 54}


def focused(study):
    return study["schema_version"] == 3


def stage_limits(study):
    return SHAPE_LIMITS if focused(study) else STAGE_LIMITS


NOTE = (
    "Finite, adaptive validation search; best tested trade-off, not global optimum. "
    "Two seeds are limited uncertainty evidence. Temporal ablations use 48x48 only. "
    "From-scratch/low-resolution references are not best-possible pretrained baselines. "
    "No test decoding or automatic final comparison."
)


def protocol_note(study):
    if not focused(study):
        return NOTE
    return (
        "Evidence-guided finite 1-3-layer search, not unbiased/exhaustive or a global optimum. "
        "Two seeds give limited uncertainty evidence; temporal ablations use 48x48. "
        "From-scratch low-resolution references are not exhaustively tuned baselines. "
        "Search never decodes test; the automatic workflow hands verified selection to "
        "fresh comparison training and validation-frozen testing. Filename independence "
        "is user-attested, not proof of subject/scene independence."
    )


def candidate(filters):
    filters = list(filters)
    return {
        "name": "custom_" + "_".join(map(str, filters)),
        "research_question": "Capacity of ConvLSTM stack "
        + "-".join(map(str, filters)),
        "convlstm_layers": [[w, [3, 3]] for w in filters],
        "hidden_classifier_width": None,
    }


def flats(study):
    specs = [candidate([w] * d) for d in study["depths"] for w in study["widths"]]
    return specs + ([candidate([64, 64, 64])] if focused(study) else [])


def shape_candidates():
    """Every relative ordering, representative scales, and historical controls."""
    variants = []
    for a, b in ((8, 16), (16, 24), (16, 32)):
        variants.extend(([a, b], [b, a]))
    a, b = 16, 32
    variants.extend(([a, a, b], [a, b, a], [b, a, a], [a, b, b], [b, a, b], [b, b, a]))
    variants.extend(permutations((8, 16, 32)))
    return [candidate(v) for v in variants]


def load_capacity(study):
    from .study_config import _fields, _path

    study = deepcopy(study)
    _fields(
        study,
        {
            "schema_version",
            "mode",
            "dataset",
            "widths",
            "depths",
            "frame_sizes",
            "learning_rates",
            "seeds",
            "training",
            "runs_dir",
            "max_runs",
        },
        "capacity search",
    )
    fixed = {
        "schema_version": 3 if focused(study) else 2,
        "mode": "capacity_search",
        "widths": [4, 8, 16, 24, 32] if focused(study) else [4, 8, 16, 24, 32, 48, 64],
        "depths": [1, 2, 3],
        "frame_sizes": [32, 48, 64],
        "learning_rates": [0.001, 0.003, 0.01],
        "seeds": [42, 2026],
        "max_runs": 177 if focused(study) else 156,
    }
    for key, expected in fixed.items():
        if value_hash(study[key]) != value_hash(expected):
            raise ValueError(f"capacity protocol requires {key}={expected}")
    data = study["dataset"]
    _fields(data, {"name", "path", "split_manifest"}, "dataset")
    if data["name"] != "aad":
        raise ValueError("capacity study is the AAD protocol")
    training = study["training"]
    _fields(
        training,
        {
            "epochs",
            "minimum_epochs",
            "early_stopping_patience",
            "batch_size",
            "num_workers",
            "pin_memory",
            "cache_dataset",
            "prediction_samples_per_category",
        },
        "capacity training",
    )
    for key, expected in {
        "epochs": 128 if focused(study) else 200,
        "minimum_epochs": 64,
        "early_stopping_patience": 24,
    }.items():
        if type(training[key]) is not int or training[key] != expected:
            raise ValueError(f"capacity protocol requires {key}={expected}")
    if not training["cache_dataset"]:
        raise ValueError("capacity search requires bounded cache reuse")
    if training["prediction_samples_per_category"] != 0:
        raise ValueError(
            "search saves complete validation records, not per-job video examples"
        )
    config = ExperimentConfig.from_mapping(
        {
            **{k: v for k, v in training.items() if k != "minimum_epochs"},
            "dataset_name": "aad",
            "dataset_dir": _path(data["path"]),
            "split_manifest": (
                _path(data["split_manifest"])
                if data["split_manifest"] is not None
                else None
            ),
            "runs_dir": _path(study["runs_dir"]),
            "seed": 42,
            "sequence_length": 8,
            "height": 48,
            "width": 48,
            "learning_rate": 0.001,
            "weight_decay": 0,
            "augment": False,
            "sampling_version": "timestamps_v1",
            "target_fps": 16,
        }
    )
    return (
        study,
        config,
        CandidateManifest.from_mapping(
            {
                "screening_id": (
                    "focused_shapes_v1" if focused(study) else "capacity_search_v1"
                ),
                "candidates": flats(study),
            }
        ),
    )


def job(stage, model, config, spec=None, **overrides):
    values = {**config.to_dict(), **overrides}
    if spec:
        values.update(
            convlstm_layers=spec["convlstm_layers"], hidden_classifier_width=None
        )
    values = ExperimentConfig.from_mapping(values).to_dict()
    return {
        "stage": stage,
        "model": model,
        "seed": values["seed"],
        "candidate": spec,
        "config": values,
        "minimum_epochs": 64,
        "changed_factor": {
            "calibration": "learning_rate",
            "flat": "width_depth_resolution",
            "references": "model_family_resolution",
            "refinement": "architecture",
            "confirmation": "model_seed",
            "weight_decay": "weight_decay",
            "temporal": (
                "target_fps" if values["sequence_length"] == 8 else "sequence_length"
            ),
        }[stage],
    }


def calibration_rows(study, config):
    specs = [candidate([24] * 3), candidate([64] * 3)]
    return [
        job("calibration", name, config, spec, learning_rate=lr)
        for name, spec in [(c["name"], c) for c in specs]
        + [(n, None) for n in REFERENCES]
        for lr in study["learning_rates"]
    ]


def select_learning_rates(study, jobs):
    expected = {
        (name, lr)
        for name in (
            candidate([24] * 3)["name"],
            candidate([64] * 3)["name"],
            *REFERENCES,
        )
        for lr in study["learning_rates"]
    }
    actual = {
        (j["result"]["name"], j["result"]["experiment_config"]["learning_rate"])
        for j in jobs
    }
    if actual != expected or len(jobs) != len(expected):
        raise ValueError("calibration requires every model/LR result exactly once")

    def select(names):
        scores = []
        for lr in study["learning_rates"]:
            entries = [
                j["result"]
                for j in jobs
                if j["result"]["name"] in names
                and j["result"]["experiment_config"]["learning_rate"] == lr
            ]
            scores.append(
                (
                    -statistics.mean(
                        validation_accuracy(e, focused(study)) for e in entries
                    ),
                    statistics.mean(e["validation_metrics"]["loss"] for e in entries),
                    lr,
                )
            )
        return min(scores)[2]

    return {
        "custom": select({candidate([24] * 3)["name"], candidate([64] * 3)["name"]}),
        **{name: select({name}) for name in REFERENCES},
    }


def matrix_rows(study, config, specs, lrs, stage, seed=42):
    rows = []
    # Resolution-major order reuses a single bounded cache.
    for size in study["frame_sizes"]:
        for spec in specs:
            name = spec["name"] if isinstance(spec, dict) else spec
            c = spec if isinstance(spec, dict) else None
            rows.append(
                job(
                    stage,
                    name,
                    config,
                    c,
                    seed=seed,
                    height=size,
                    width=size,
                    learning_rate=lrs["custom" if c else name],
                )
            )
    return rows


def architecture_ranking(jobs, names, sizes, seeds=(42,), exact=False):
    """Rank only a declared common-reference matrix, never ablation winners."""
    per_seed = {}
    for seed in seeds:
        records = [
            j["result"]
            for j in jobs
            if j["result"]["name"] in names and j["result"]["seed"] == seed
        ]
        per_seed[seed] = rank_resolutions(records, names, sizes, seed, exact=exact)
    ranked = []
    for name in names:
        values = [next(v for v in per_seed[s] if v["name"] == name) for s in seeds]
        means = [v["mean_validation_accuracy"] for v in values]
        ranked.append(
            {
                "name": name,
                "mean_validation_accuracy": statistics.mean(means),
                **(
                    {
                        "accuracy_fraction": list(
                            statistics.mean(
                                Fraction(*v["accuracy_fraction"]) for v in values
                            ).as_integer_ratio()
                        )
                    }
                    if exact
                    else {}
                ),
                "mean_validation_loss": statistics.mean(
                    v["mean_validation_loss"] for v in values
                ),
                "worst_validation_accuracy": min(
                    v["worst_validation_accuracy"] for v in values
                ),
                "num_params": values[0]["num_params"],
                "seed_sample_sd": statistics.stdev(means) if len(means) > 1 else None,
                "per_seed": [{"seed": s, **v} for s, v in zip(seeds, values)],
            }
        )
    return sorted(
        ranked,
        key=lambda r: (
            -(
                Fraction(*r["accuracy_fraction"])
                if exact
                else r["mean_validation_accuracy"]
            ),
            r["mean_validation_loss"],
            r["num_params"],
            r["name"],
        ),
    )


def refinement_candidates(w, existing):
    h = max(4, w // 2)
    variants = [[w] * 4, [h, w, w], [w, h, w], [w, w, h], [32, 16], [16, 32, 16]]
    seen = {c["name"] for c in existing}
    selected = []
    for filters in variants:
        c = candidate(filters)
        if c["name"] not in seen:
            selected.append(c)
            seen.add(c["name"])
    return selected


def confirmation_specs(study, shortlist, w):
    widths = sorted(
        {
            c["convlstm_layers"][0][0]
            for c in flats(study)
            if len(c["convlstm_layers"]) == 3
        }
    )
    index = widths.index(w)
    specs = list(shortlist)
    specs.extend(candidate([v] * 3) for v in widths[max(0, index - 1) : index + 2])
    return list({c["name"]: c for c in specs}.values()) + list(REFERENCES)


def ablation_rows(study, config, shortlist, lrs, stage):
    rows = []
    if stage == "weight_decay":
        for size in study["frame_sizes"]:
            for c in shortlist:
                for wd in (1e-4, 1e-3):
                    rows.append(
                        job(
                            stage,
                            c["name"],
                            config,
                            c,
                            height=size,
                            width=size,
                            learning_rate=lrs["custom"],
                            weight_decay=wd,
                        )
                    )
    elif stage == "temporal":
        for spec in [*shortlist, *REFERENCES]:
            c = spec if isinstance(spec, dict) else None
            name = c["name"] if c else spec
            for frames, fps in ((8, 8), (8, 4), (16, 16)):
                rows.append(
                    job(
                        stage,
                        name,
                        config,
                        c,
                        sequence_length=frames,
                        target_fps=fps,
                        learning_rate=lrs["custom" if c else name],
                    )
                )
    else:
        raise ValueError("unknown ablation")
    return rows


def print_capacity_plan(study, config):
    if focused(study):
        print(
            "Ticket 073: 34 focused 1-3-layer architectures; test locked during search"
        )
        print(
            "Evidence-guided, not exhaustive or unbiased; user-reviewed filename independence, not proven subject independence."
        )
        for stage, count in stage_limits(study).items():
            print(f"{stage}: up to {count} jobs before exact-config reuse")
        print("Upper budget: 177 search jobs; 128 epochs cap, minimum 64, patience 24.")
        print(
            "Automatic notebook: then eight comparison trainings, 512 cap/minimum 128/patience 32; frozen-test guard remains."
        )
        for size in study["frame_sizes"]:
            for c in flats(study) + shape_candidates():
                print(
                    f"custom: {c['name']} | {size}x{size} | 8 frames/16 FPS | seed=42"
                )
        return
    print("Ticket 069: bounded validation-only capacity search; test locked")
    print(NOTE)
    for stage, limit in STAGE_LIMITS.items():
        print(
            f"{stage}: {'up to ' if stage in {'refinement', 'confirmation'} else ''}{limit} jobs"
        )
    print(
        "Upper budget: 156 before exact-config reuse/deduplication; stages resolve only from complete prior evidence."
    )
    print(
        "8/16 frames maximum; 4/8/16 target FPS; 200 epochs cap, minimum 64, patience 24."
    )
    print(
        "Calibration: custom_24_24_24, custom_64_64_64, r3d_18, swin3d_t at 48px; LRs 0.001/0.003/0.01."
    )
    for size in study["frame_sizes"]:
        for c in flats(study):
            print(f"flat: {c['name']} | {size}x{size} | 8 frames/16 FPS | seed=42")
        for name in REFERENCES:
            print(f"references: {name} | {size}x{size} | 8 frames/16 FPS | seed=42")
    print(
        "Refinement: peak three-layer width -> four layers/half-width placements + 32-16 and 16-32-16."
    )
    print(
        "Confirmation: top-three custom + both references + flat peak/neighbours, seed 2026, all resolutions."
    )
    print(
        "WD: confirmed custom top three x all resolutions x {0, 1e-4, 1e-3}; zero reused."
    )
    print(
        "Temporal: confirmed custom top three + references at 48px; (8f,8fps),(8f,4fps),(16f,16fps)."
    )
    print(
        "An existing split is required at execution; no data download/decoding occurs for --list-plan."
    )
