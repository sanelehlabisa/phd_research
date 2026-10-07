"""Bounded playable prediction exports and split-safe CLI endings."""

import hashlib
import json
import random
import sys
from pathlib import Path

import av
import pytest
import torch
from torch.utils.data import Dataset, Subset

from src.experiment_config import ExperimentConfig
from src.utils import save_prediction_examples, write_json, write_video_torchvision


class ToyClips(Dataset):
    def __init__(self, root, targets):
        self.targets = targets
        self.samples = [
            (root / str(i) / "same.mp4", label) for i, label in enumerate(targets)
        ]
        self.accessed = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.accessed.append(index)
        return torch.zeros(2, 3, 8, 8), self.targets[index]


class AlwaysZero(torch.nn.Module):
    def forward(self, frames):
        return frames.new_tensor([[3.0, 0.0]]).expand(len(frames), -1)


@pytest.mark.parametrize(
    "targets,expected",
    [
        ([0, 1] * 5, {"correct": 3, "incorrect": 3}),
        ([0] * 8, {"correct": 3, "incorrect": 0}),
        ([1] * 8, {"correct": 0, "incorrect": 3}),
    ],
)
def test_counts_playable_clips_and_source_provenance(
    tmp_path, targets, expected, capsys
):
    base = ToyClips(tmp_path / "source", [9, *targets, 9])
    subset = Subset(Subset(base, list(range(1, len(targets) + 1))), range(len(targets)))
    state = random.getstate()
    model = AlwaysZero().train()
    report = save_prediction_examples(
        model,
        subset,
        ["normal", "violent"],
        "cpu",
        tmp_path / "predictions",
        partition="validation",
    )
    assert model.training and random.getstate() == state
    assert report["saved_counts"] == expected
    assert report["prediction_samples_per_category"] == 3
    assert set(base.accessed) <= set(range(1, len(targets) + 1))
    assert len({r["path"] for r in report["records"]}) == sum(expected.values())
    for record in report["records"]:
        assert record["source"] == str(
            base.samples[record["source_index"]][0].resolve()
        )
        assert record["partition"] == "validation"
        assert Path(record["path"]).parent.name == (
            "correct" if record["true"] == record["pred"] else "incorrect"
        )
        with av.open(record["path"]) as video:
            assert len(list(video.decode(video=0))) == 2
    assert json.loads(Path(report["manifest"]).read_text()) == report
    assert (
        f"correct={expected['correct']}, incorrect={expected['incorrect']}"
        in capsys.readouterr().out
    )


def test_zero_disables_decoding_and_videos(tmp_path):
    data = ToyClips(tmp_path, [0, 1])
    report = save_prediction_examples(
        AlwaysZero(),
        Subset(data, [0, 1]),
        ["a", "b"],
        "cpu",
        tmp_path / "out",
        0,
        partition="test",
    )
    assert not data.accessed and not list(tmp_path.rglob("*.mp4"))
    assert report["saved_counts"] == {"correct": 0, "incorrect": 0}


@pytest.mark.parametrize("value", [-1, True, 1.5, "3"])
def test_invalid_count_is_rejected(value, tmp_path):
    with pytest.raises(ValueError, match="prediction_samples_per_category"):
        ExperimentConfig(prediction_samples_per_category=value)
    with pytest.raises(ValueError, match="prediction_samples_per_category"):
        save_prediction_examples(
            AlwaysZero(),
            Subset(ToyClips(tmp_path, []), []),
            [],
            "cpu",
            tmp_path,
            value,
            partition="validation",
        )


def test_no_unsplit_or_training_examples(tmp_path):
    data = ToyClips(tmp_path, [0])
    with pytest.raises(ValueError, match="Subset"):
        save_prediction_examples(
            AlwaysZero(), data, [], "cpu", tmp_path, partition="validation"
        )
    with pytest.raises(ValueError, match="validation or final test"):
        save_prediction_examples(
            AlwaysZero(), Subset(data, [0]), [], "cpu", tmp_path, partition="train"
        )


@pytest.mark.parametrize("limit,correct,incorrect", [(0, 0, 0), (3, 3, 0), (3, 3, 3)])
def test_final_report_reader_accepts_current_category_counts(
    tmp_path, limit, correct, incorrect
):
    from notebooks.utils.helpers import final_test_report

    records = [
        {"path": str(i), "partition": "test", "correct": i < correct}
        for i in range(correct + incorrect)
    ]
    report = {
        "partition": "test",
        "prediction_samples_per_category": limit,
        "prediction_examples": {
            "partition": "test",
            "prediction_samples_per_category": limit,
            "records": records,
            "saved_counts": {"correct": correct, "incorrect": incorrect},
        },
        "artifacts": {"prediction_clips": records},
    }
    write_json(tmp_path / "run.json", {"status": "complete"})
    write_json(tmp_path / "metrics/final.json", report)
    assert final_test_report(tmp_path)["examples"] == records
    report["prediction_examples"]["saved_counts"]["correct"] += 1
    write_json(tmp_path / "metrics/final.json", report)
    with pytest.raises(ValueError, match="counts"):
        final_test_report(tmp_path)


def test_all_four_commands_use_only_their_partition(tmp_path, monkeypatch):
    from src import dataset, dataset_source, evaluate, experiments, model, train

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    root = tmp_path / "aad"
    for label, name in enumerate(dataset_source.AAD_CLASS_NAMES):
        for index in range(10):
            frames = torch.full((2, 3, 8, 8), (label + 1) / 12)
            write_video_torchvision(frames, root / name / f"{index}.mp4")
    manifest_path = tmp_path / "split.json"
    source = dataset.AHARDataset(root, 2, (8, 8))
    _, validation, test, _ = dataset.load_split_subsets(source, manifest_path)
    locked = {source.samples[index][0].resolve() for index in test.indices}
    accessed = []
    original_load = dataset.AHARDataset._load_video
    test_allowed = False

    def guarded_load(self, path):
        accessed.append(Path(path).resolve())
        assert test_allowed or Path(path).resolve() not in locked
        return original_load(self, path)

    monkeypatch.setattr(dataset.AHARDataset, "_load_video", guarded_load)
    values = ExperimentConfig(
        dataset_dir=str(root),
        runs_dir=str(tmp_path / "runs"),
        split_manifest=str(manifest_path),
        sequence_length=2,
        height=8,
        width=8,
        convlstm_layers=((2, (3, 3)),),
        epochs=1,
        batch_size=32,
        prediction_samples_per_category=1,
        scheduler="none",
    ).to_dict()
    config_path = write_json(tmp_path / "train.json", values)
    train.main(["--config", str(config_path)])
    train_dir = next((tmp_path / "runs" / "train").iterdir())
    report = json.loads((train_dir / "metrics/final.json").read_text())
    assert report["test_access"] == "locked"
    assert report["prediction_examples"]["partition"] == "validation"
    assert report["checkpoint_selection"]["selected_epoch"] == 1
    checkpoint = train_dir / "checkpoints/best_model.pth"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    experiments.main(
        ["--config", str(config_path), "--model", "custom_convlstm_configured"]
    )
    exp_dir = next((tmp_path / "runs" / "experiments").iterdir())
    exp = json.loads((exp_dir / "summary.json").read_text())["all"][0]
    assert exp["test_access"] == "locked"
    assert exp["prediction_examples"]["partition"] == "validation"
    assert "/models/" in Path(exp["prediction_examples"]["manifest"]).as_posix()

    smoke = write_json(tmp_path / "smoke.json", values)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model",
            "--config",
            str(smoke),
            "--height",
            "4",
            "--width",
            "4",
            "--convlstm-layer",
            "2",
            "3",
            "3",
        ],
    )
    model.main()
    model_dir = next((tmp_path / "runs" / "model").iterdir())
    smoke_report = json.loads((model_dir / "summary.json").read_text())
    assert smoke_report["test_access"] == "locked"
    assert "Random-weight" in smoke_report["warning"]
    for entry in smoke_report["models"].values():
        assert entry["prediction_examples"]["partition"] == "validation"
    assert not locked.intersection(accessed)

    test_allowed = True
    accessed.clear()
    evaluation = write_json(
        tmp_path / "eval.json",
        {
            **values,
            "training_run_dir": str(train_dir),
        },
    )
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(evaluation)])
    evaluate.main()
    assert accessed and set(accessed) <= locked
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
    eval_dir = next((tmp_path / "runs" / "evaluate").iterdir())
    final = json.loads((eval_dir / "metrics/final.json").read_text())
    assert final["partition"] == final["prediction_examples"]["partition"] == "test"
    for run in (report, exp, *smoke_report["models"].values(), final):
        examples = run["prediction_examples"]
        assert examples["prediction_samples_per_category"] == 1
        assert 1 <= sum(examples["saved_counts"].values()) <= 2
        assert all(value <= 1 for value in examples["saved_counts"].values())
