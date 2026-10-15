"""Thin one-JSON entry into the existing saved-comparison runner (no search)."""

from pathlib import Path

from .study_matrix import read_json, load_selection
from .focused_protocol import SLOTS, BASELINES

STACKS = [[64, 96, 64], [128, 64, 64], [64, 128, 64], [64, 64, 128], [96, 64, 64]]


def run_command(path, *, list_plan=False):
    from .study_config import ROOT
    from notebooks.utils.aad_study import run_saved_comparison

    values = read_json(path)
    if (
        values.get("entrypoint") != "saved_comparison"
        or values.get("comparison_only") is not True
        or values.get("mode") != "top3_comparison"
        or values.get("comparison_protocol") != "focused_final_v1"
        or values.get("models") != SLOTS + BASELINES
    ):
        raise ValueError("Expected the fixed ten-model comparison-only profile")
    training = values["training"]
    if (
        any(
            training.get(k) != v
            for k, v in dict(
                epochs=512,
                early_stopping_patience=32,
                learning_rate=0.001,
                scheduler="reduce_on_plateau",
                augment=False,
                cache_dataset=True,
            ).items()
        )
        or values["reference_input"]
        != dict(frame_size=64, sequence_length=8, weight_decay=0)
        or values["factors"]
        != dict(frame_sizes=[64], sequence_lengths=[8], weight_decays=[0])
        or values["seeds"] != [42]
        or values["minimum_epochs"] != 128
        or values["max_runs"] != 10
        or values["dataset"]["name"] != "aad"
    ):
        raise ValueError(
            "Changed comparison recipe/input requires a separately approved protocol"
        )
    selected_path = (ROOT / values["selected_config"]).resolve()
    print(
        "Comparison only: 0 search jobs; 10 models -> validation freeze -> test -> examples -> results ZIP.",
        flush=True,
    )
    for i, name in enumerate(values["models"], 1):
        print(f"Model {i}/10: {name}" + (f" {STACKS[i - 1]}" if i <= 5 else ""))
    if list_plan:
        print(f"Evidence required before execution: {selected_path}")
        print(
            "Recipe unchanged: Adam LR=.001 WD=0, 8f/64px/16FPS, 512/128/32; local GPU fit unverified."
        )
        return None
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Missing verified selection/import manifest: {selected_path}. "
            "The essential summaries ZIP is insufficient; restore selection, split, "
            "audit, receipts and validation evidence. No search will be launched."
        )
    selection = load_selection(selected_path)
    if (ROOT / values["dataset"]["path"]).resolve() != Path(
        selection["dataset_dir"]
    ).resolve():
        raise ValueError("JSON dataset.path must match the verified import mapping")
    if values["dataset"]["split_manifest"] is not None:
        from .study_matrix import file_hash

        if (
            file_hash(ROOT / values["dataset"]["split_manifest"])
            != selection["split"]["manifest_hash"]
        ):
            raise ValueError("JSON split differs from the imported selection")
    actual = [
        [layer[0] for layer in item["candidate"]["convlstm_layers"]]
        for item in selection["top5"]
    ]
    if actual != STACKS:
        raise ValueError("Imported finalists differ from the approved 083 shortlist")
    final = selection["final_input"]
    if (
        final["sequence_length"],
        final["height"],
        final["width"],
        final["target_fps"],
    ) != (8, 64, 64, 16):
        raise ValueError("Imported selection differs from the approved final input")
    split = read_json(selection["split"]["manifest_path"])
    from .dataset_source import AAD_CLASS_NAMES

    if split["class_names"] != list(AAD_CLASS_NAMES) or {
        k: split["counts"].get(k) for k in ("train", "validation", "test")
    } != dict(train=748, validation=160, test=161):
        raise ValueError("083 requires the original eleven-class 748/160/161 split")
    return run_saved_comparison(ROOT, selected_path, template_path=Path(path).resolve())
