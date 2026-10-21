"""Offline checks for the separate, all-class exploratory Kinetics workflow."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "notebooks/scripts/10_train_demo_kinetics_colab.py"
NOTEBOOK = ROOT / "notebooks/10_train_demo_kinetics_colab.ipynb"
SHARED = ROOT / "notebooks/scripts/09_train_demo_winner_colab.py"


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    with patch("subprocess.run", side_effect=AssertionError("Import ran a command")):
        spec.loader.exec_module(module)
    return module


class KineticsDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.module = load_module(EXPORT)
        self.shared = load_module(SHARED)

    def class_root(self, root, names=("clapping", "juggling", "rowing"), clips=3):
        for name in names:
            directory = root / name
            directory.mkdir(parents=True)
            for number in range(clips):
                (directory / f"clip-{number}.mp4").touch()
        return root.resolve()

    def dataset_helpers(self, real_probe=False):
        namespace = {"__name__": "kinetics_dataset_test"}
        exec(self.module.DATASET_SETUP_CODE, namespace)
        if not real_probe:
            namespace["video_plan"] = Mock(return_value={"sampling_version": "timestamps_clip_v1"})
        return namespace

    def test_notebook_parity_safe_import_and_plain_python(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = EXPORT.read_text(encoding="utf-8")
        code = "".join(
            "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        self.assertEqual(code.strip(), source.strip())
        ast.parse(source)
        ast.parse(self.module.DATASET_SETUP_CODE)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertEqual(notebook["metadata"]["colab"]["name"], NOTEBOOK.name)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"]))
        imports = {
            alias.name.split(".")[0]
            for node in ast.parse(source).body if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertFalse(imports & {"torch", "torchvision", "numpy", "kagglehub", "src"})

    def test_recipe_preserves_architecture_and_infers_every_class(self):
        config = self.module.kinetics_config(self.base, self.shared.winner_config)
        expected = self.shared.winner_config(self.base)
        for key in (
            "convlstm_layers", "models", "sequence_length", "height", "width",
            "target_fps", "seed", "split_seed", "learning_rate", "weight_decay",
            "augment", "batch_size", "optimizer", "train_ratio", "val_ratio", "test_ratio",
        ):
            self.assertEqual(config[key], expected[key], key)
        self.assertEqual(config["epochs"], 128)
        self.assertEqual(self.module.MINIMUM_EPOCHS, 64)
        self.assertEqual(config["early_stopping_patience"], 16)
        self.assertFalse(config["cache_dataset"])
        self.assertEqual(config["num_workers"], 2)
        self.assertEqual(config["sampling_version"], "timestamps_clip_v1")
        self.assertNotIn("aad", json.dumps(config).lower())
        self.assertNotIn("accepted_classes", config)
        self.assertNotIn("class_names", config)
        self.assertIn("exploratory", config["dataset_name"])
        self.module.EPOCHS = 12
        self.assertEqual(self.module.kinetics_config(self.base, self.shared.winner_config)["epochs"], 12)

    def test_mocked_unversioned_download_discovers_all_nested_classes_and_counts(self):
        downloaded = self.base / "kaggle"
        names = ("clapping", "juggling", "rowing", "skipping", "swimming", "walking")
        root = self.class_root(downloaded / "nested/videos", names)
        helpers = self.dataset_helpers()
        output = self.base / "inventory.json"
        with patch("kagglehub.dataset_download", return_value=str(downloaded)) as download, redirect_stdout(io.StringIO()) as stdout:
            helpers["download_dataset"](self.module.HANDLE, output)
        download.assert_called_once_with("sanelehlabisa/kinetics-400-dataset")
        inventory = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(inventory["dataset_root"], str(root))
        self.assertEqual(inventory["class_names"], sorted(names))
        self.assertEqual(inventory["total_clips"], 18)
        self.assertEqual(inventory["class_clip_counts"], dict.fromkeys(names, 3))
        self.assertIn("Actual classes: 6 | usable clips: 18 | excluded: 0", stdout.getvalue())
        self.assertEqual(inventory["excluded_samples"], [])
        self.assertIn(str(root), stdout.getvalue())
        self.assertNotIn("full 400", stdout.getvalue())

    def test_unusable_clip_is_recorded_excluded_and_never_deleted(self):
        root = self.class_root(self.base / "download")
        bad = root / "rowing/clip-1.mp4"
        bad.write_bytes(b"unchanged broken video")
        helpers = self.dataset_helpers()

        def probe(path, fps, frames, **kwargs):
            self.assertEqual(kwargs, {"sampling_version": "timestamps_clip_v1"})
            if path == bad:
                raise ValueError(f"No video stream: {path}")
            return {}

        helpers["video_plan"] = probe
        output = self.base / "inventory.json"
        with patch("kagglehub.dataset_download", return_value=str(root)), redirect_stdout(io.StringIO()) as stdout:
            helpers["download_dataset"](self.module.HANDLE, output)
        inventory = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(inventory["original_total_clips"], 9)
        self.assertEqual(inventory["total_clips"], 8)
        self.assertEqual(inventory["excluded_samples"], ["rowing/clip-1.mp4"])
        self.assertIn("No video stream", inventory["excluded_clips"][0]["reason"])
        self.assertEqual(inventory["class_names"], ["clapping", "juggling", "rowing"])
        self.assertEqual(inventory["class_clip_counts"]["rowing"], 2)
        self.assertIn("Excluded unusable input", stdout.getvalue())
        self.assertEqual(bad.read_bytes(), b"unchanged broken video")

    def test_empty_usable_class_fails_with_a_saved_receipt_not_silent_class_drop(self):
        root = self.class_root(self.base / "download", clips=1)
        helpers = self.dataset_helpers()
        helpers["video_plan"].side_effect = [ValueError("No video stream"), {}, {}]
        output = self.base / "inventory.json"
        with patch("kagglehub.dataset_download", return_value=str(root)), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "No usable clips.*clapping"):
                helpers["download_dataset"](self.module.HANDLE, output)
        self.assertEqual(json.loads(output.read_text())["excluded_count"], 1)

    def test_preflight_does_not_hide_programming_errors_as_bad_media(self):
        root = self.class_root(self.base / "download")
        helpers = self.dataset_helpers()
        helpers["video_plan"].side_effect = TypeError("unexpected API bug")
        with patch("kagglehub.dataset_download", return_value=str(root)), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(TypeError, "unexpected API bug"):
                helpers["download_dataset"](self.module.HANDLE, self.base / "inventory.json")

    def test_real_filtered_dataset_trains_and_saves_artifacts_without_bad_clip(self):
        import torch
        from src import experiments
        from src.utils import write_video_torchvision

        torch.set_num_threads(1)
        root = self.base / "real-clips"
        for name in ("clapping", "rowing"):
            for index in range(10):
                write_video_torchvision(
                    torch.zeros(12, 3, 8, 8), root / name / f"{index}.mp4", fps=20
                )
        bad = root / "rowing/unusable.mp4"
        bad.write_bytes(b"not a usable MP4")
        helpers = self.dataset_helpers(real_probe=True)
        receipt = self.base / "inventory.json"
        with patch("kagglehub.dataset_download", return_value=str(root)), redirect_stdout(io.StringIO()):
            helpers["download_dataset"](self.module.HANDLE, receipt)
        inventory = json.loads(receipt.read_text())
        self.assertEqual(inventory["excluded_samples"], ["rowing/unusable.mp4"])
        config = self.module.kinetics_config(self.base / "runs", self.shared.winner_config)
        config.update(
            dataset_dir=str(root), excluded_samples=inventory["excluded_samples"],
            split_manifest=str(self.base / "split.json"), convlstm_layers=[[2, [3, 3]]],
            epochs=1, height=8, width=8, batch_size=4, num_workers=0,
            prediction_samples_per_category=0,
        )
        path = self.base / "config.json"
        path.write_text(json.dumps(config))
        with patch.object(torch.cuda, "is_available", return_value=False), redirect_stdout(io.StringIO()):
            run = experiments.main(["--config", str(path)])
        split = json.loads((self.base / "split.json").read_text())
        self.assertEqual(split["counts"]["total"], 20)
        self.assertNotIn("rowing/unusable.mp4", [row["path"] for row in split["samples"]])
        result = json.loads((run / "summary.json").read_text())["all"][0]
        self.assertEqual(result["test_access"], "locked")
        self.assertTrue(Path(result["selected_checkpoint"]).is_file())
        self.assertTrue(Path(result["sampling_report"]).is_file())
        self.assertTrue((run / "models" / self.shared.MODEL / "metrics/validation_confusion_matrix.png").is_file())
        self.assertEqual(bad.read_bytes(), b"not a usable MP4")

    def test_official_training_root_preferred_without_merging_splits(self):
        helpers = self.dataset_helpers()
        for name in ("train", "training"):
            downloaded = self.base / name
            training = self.class_root(downloaded / "wrapper" / name)
            self.class_root(downloaded / "wrapper/test", ("heldout_a", "heldout_b"))
            self.class_root(downloaded / "wrapper/val", ("validation_a", "validation_b"))
            self.assertEqual(helpers["find_class_root"](downloaded), (training, name))

    def test_split_like_class_labels_remain_labels(self):
        root = self.class_root(self.base / "classes", ("train", "val", "test"))
        self.assertEqual(self.dataset_helpers()["find_class_root"](root), (root, None))

    def test_flat_csv_split_folders_are_not_treated_as_activity_labels(self):
        root = self.class_root(self.base / "flat-splits", ("train", "test"))
        (root / "labels.csv").write_text("clip,label\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not activity labels"):
            self.dataset_helpers()["find_class_root"](root)

    def test_flat_csv_and_heldout_only_layouts_fail_clearly(self):
        helpers = self.dataset_helpers()
        flat = self.base / "flat"
        flat.mkdir()
        (flat / "video.mp4").touch()
        (flat / "labels.csv").write_text("clip,label\nvideo.mp4,clapping\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Flat video/CSV"):
            helpers["find_class_root"](flat)
        heldout = self.base / "heldout"
        self.class_root(heldout / "test")
        with self.assertRaisesRegex(RuntimeError, "refusing to train on validation/test"):
            helpers["find_class_root"](heldout)

    def test_dataset_setup_uses_fresh_process_and_a_separate_drive_manifest(self):
        destination = self.base / "drive"
        destination.mkdir()
        actual = self.class_root(self.base / "download")
        config = self.module.kinetics_config(self.base, self.shared.winner_config)

        def run(command, **kwargs):
            self.assertEqual(command[:4], [sys.executable, "-u", "-c", self.module.DATASET_SETUP_CODE])
            self.assertEqual(command[4], self.module.HANDLE)
            self.assertEqual(kwargs, {"cwd": ROOT, "log_path": destination / "dataset.log"})
            self.assertEqual(command[6:], ["16", "8"])
            Path(command[5]).write_text(json.dumps({
                "dataset_root": str(actual), "excluded_samples": ["rowing/clip-0.mp4"]
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        resolved = self.module.prepare_kinetics_dataset(ROOT, destination, config, run)
        self.assertEqual(resolved["dataset_dir"], str(actual))
        self.assertEqual(resolved["excluded_samples"], ["rowing/clip-0.mp4"])
        self.assertEqual(Path(resolved["split_manifest"]), destination / "kinetics_split_seed42.json")
        self.assertNotEqual(resolved["split_manifest"], self.shared.winner_config(self.base)["split_manifest"])

    def test_entry_reuses_shared_workflow_with_separate_storage_and_no_app_launch(self):
        trainer = Mock(return_value=self.base / "bundle.zip")
        with patch.object(self.module, "load_shared_runner", return_value={
            "train_demo_on_colab": trainer, "winner_config": self.shared.winner_config,
            "run_logged": Mock(),
        }), redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(self.module.train_kinetics_on_colab(), self.base / "bundle.zip")
        options = trainer.call_args.kwargs
        self.assertEqual(options["minimum_epochs"], 64)
        self.assertEqual(options["storage_name"], "phd_demo_kinetics")
        self.assertFalse(options["app_launch"])
        self.assertFalse(options["update_checkout"])
        self.assertTrue(callable(options["dataset_setup"]))
        self.assertEqual(options["config_factory"](self.base)["epochs"], 128)
        self.assertIn("not paper evidence", options["purpose"])
        self.assertIn("Prefix-only checks", stdout.getvalue())

    def test_missing_published_helper_fails_before_dataset_or_training(self):
        self.module.REPO_ROOT = self.base / "checkout"
        (self.module.REPO_ROOT / ".git").mkdir(parents=True)
        with patch.object(self.module.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Publish.*complete notebook/scripts update"):
                self.module.load_shared_runner()
        run.assert_called_once_with(
            ["git", "pull", "--ff-only", "origin", "master"],
            cwd=self.module.REPO_ROOT, check=True,
        )

    def test_bundle_keeps_actual_split_labels_checkpoint_and_validation_artifacts(self):
        destination = self.base / "drive"
        run = destination / "experiments/kinetics-run"
        model = run / "models" / self.shared.MODEL
        local_runs = self.base / "local"
        local_runs.mkdir()
        (local_runs / "cache").mkdir()
        (local_runs / "cache/excluded.bin").touch()
        labels = ["clapping", "juggling", "rowing"]
        split = destination / "kinetics_split_seed42.json"
        model.mkdir(parents=True)
        split.write_text(json.dumps({"class_names": labels, "seed": 42}), encoding="utf-8")
        files = {
            run / "run.json": {"status": "complete", "split_manifest": str(split)},
            run / "config.json": {"dataset_name": "kinetics400-exploratory"},
            run / "resolved_config.json": {"dataset_name": "kinetics400-exploratory"},
            model / "checkpoints/best_model.json": {"class_names": labels},
            model / "metrics/metrics.json": {"partition": "validation"},
            model / "metrics/history.json": [{"epoch": 1}],
            model / "predictions/predictions.json": {"records": [{"true": "rowing"}]},
            destination / "dataset_inventory.json": {"class_names": labels, "total_clips": 30},
        }
        for path, values in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(values), encoding="utf-8")
        (model / "checkpoints/best_model.pth").write_bytes(b"synthetic weights")
        (model / "predictions/validation-example.mp4").touch()
        with redirect_stdout(io.StringIO()) as stdout:
            archive = self.shared.save_bundle(
                run, local_runs, destination, ROOT,
                purpose=self.module.PURPOSE, app_launch=False,
            )
        self.assertTrue(archive.is_file())
        self.assertTrue((run / "split_manifest.json").is_file())
        with ZipFile(archive) as zipped:
            prefix = "kinetics-run/"
            self.assertEqual(json.loads(zipped.read(prefix + "split_manifest.json"))["class_names"], labels)
            receipt = json.loads(zipped.read(prefix + "demo_bundle.json"))
            self.assertEqual(receipt["class_names"], labels)
            self.assertFalse(receipt["app_compatible"])
            self.assertEqual(receipt["test_access"], "locked")
            self.assertNotIn("local_launch_after_extracting_into_implementation_runs_demo", receipt)
            for suffix in ("checkpoints/best_model.pth", "metrics/history.json", "predictions/validation-example.mp4"):
                self.assertTrue(any(name.endswith(suffix) for name in zipped.namelist()))
            self.assertIn(prefix + "dataset_inventory.json", zipped.namelist())
            self.assertFalse(any("cache" in name for name in zipped.namelist()))
        self.assertNotIn("python demo.py", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
