"""Offline Colab-export checks: no ML imports, cloud access or training."""

import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "notebooks/train_demo_winner_colab.py"
NOTEBOOK = EXPORT.with_suffix(".ipynb")


def load_export():
    spec = importlib.util.spec_from_file_location("demo_colab_under_test", EXPORT)
    module = importlib.util.module_from_spec(spec)
    with patch(
        "subprocess.run", side_effect=AssertionError("Import must not run commands")
    ):
        spec.loader.exec_module(module)
    return module


class DemoColabTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.base = Path(self.workspace.name)
        self.module = load_export()
        self.module.CONTENT_ROOT = self.base / "content"
        self.module.REPO_ROOT = self.module.CONTENT_ROOT / "phd_research"
        self.module.DRIVE_ROOT = self.module.CONTENT_ROOT / "drive"
        self.root = (
            self.module.REPO_ROOT
            / "papers/001-journal-abnormal-activity-recognition/implementation"
        )
        self.module.REPO_ROOT.mkdir(parents=True)
        (self.module.REPO_ROOT / ".git").mkdir()
        for name in self.module.REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        self.local_runs = self.module.CONTENT_ROOT / "local_runs"
        self.local_runs.mkdir()
        self.destination = self.module.DRIVE_ROOT / "MyDrive/phd_demo/session"
        (self.destination / "experiments").mkdir(parents=True)
        self.colab = ModuleType("google.colab")
        self.colab.drive = SimpleNamespace(mount=Mock())
        self.colab.files = SimpleNamespace(download=Mock())
        self.google = ModuleType("google")
        self.google.colab = self.colab

    def test_notebook_export_parity_and_safe_import(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = EXPORT.read_text(encoding="utf-8")
        code = "".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertEqual(code.strip(), source.strip())
        ast.parse(code)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"]))
        self.assertIn('if __name__ == "__main__":', source)
        top_imports = [
            node
            for node in ast.parse(source).body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imports = {
            (
                node.module.split(".")[0]
                if isinstance(node, ast.ImportFrom)
                else alias.name.split(".")[0]
            )
            for node in top_imports
            for alias in node.names
        }
        self.assertFalse(
            imports & {"torch", "torchvision", "numpy", "src", "notebooks"}
        )

    def test_embedded_recipe_matches_local_winner_and_uses_published_helpers(self):
        expected = json.loads(
            (ROOT / "configs/experiments/aad_demo_selected_local.json").read_text(
                encoding="utf-8"
            )
        )
        values = self.module.winner_config(self.local_runs)
        self.assertEqual(set(values), set(expected))
        for key in set(expected) - {"dataset_name", "dataset_dir", "runs_dir"}:
            self.assertEqual(values[key], expected[key], key)
        self.assertEqual(values["dataset_name"], "aad")
        self.assertEqual(Path(values["runs_dir"]), self.local_runs)
        self.assertNotIn("minimum_epochs", values)
        self.assertEqual(values["models"], ["custom_convlstm_configured"])
        for required in self.module.REQUIRED_FILES:
            self.assertTrue((ROOT / required).is_file(), required)
            self.assertNotIn("demo", required)
        self.assertNotIn("src.demo_inference", EXPORT.read_text(encoding="utf-8"))

    def test_training_command_is_exactly_one_validation_only_leaf(self):
        config_path = self.base / "config.json"
        command = self.module.train_command(config_path)
        self.assertEqual(command[:4], [sys.executable, "-u", "-m", "src.experiments"])
        self.assertEqual(command[command.index("--config") + 1], str(config_path))
        self.assertEqual(command[command.index("--minimum-epochs") + 1], "128")
        self.assertIn("--matched-training-evaluation", command)
        self.assertIn("--per-class-reporting", command)
        self.assertFalse(
            set(command)
            & {"src.evaluate", "--run-plan-stage", "--study-config", "--resume"}
        )

    def test_drive_must_mount_before_any_training_setup(self):
        self.colab.drive.mount.side_effect = RuntimeError("Drive authorization failed")
        with patch.dict(
            sys.modules, {"google": self.google, "google.colab": self.colab}
        ), patch.object(self.module.subprocess, "run") as runner:
            with self.assertRaisesRegex(RuntimeError, "Drive authorization"):
                self.module.train_demo_on_colab()
        runner.assert_not_called()

    def test_drive_storage_probe_routes_experiments_but_not_cache(self):
        with patch.dict(
            sys.modules, {"google": self.google, "google.colab": self.colab}
        ):
            try:
                local_runs, destination = self.module.prepare_storage()
            except OSError as error:
                if getattr(error, "winerror", None) == 1314:
                    self.skipTest("Windows lacks symlink privilege; Colab uses Linux.")
                raise
        self.colab.drive.mount.assert_called_once_with(str(self.module.DRIVE_ROOT))
        self.assertEqual(
            (local_runs / "experiments").resolve(),
            (destination / "experiments").resolve(),
        )
        self.assertIn(
            "Drive save verified",
            (destination / "experiments/drive_test.txt").read_text(encoding="utf-8"),
        )
        self.assertNotEqual((local_runs / "cache").parent, destination)
        self.assertFalse((destination / "cache").exists())

    def _workflow(self, bootstrap_code=0, gpu_error=False, preview_error=False):
        events = []
        commands = []
        archive = self.destination / "demo_model.zip"

        def run(command, **kwargs):
            commands.append(command)
            if any(str(item).endswith("colab_bootstrap.py") for item in command):
                events.append("bootstrap")
                return subprocess.CompletedProcess(command, bootstrap_code)
            if "-c" in command:
                self.assertEqual(command[0], sys.executable)
                self.assertIn("validate_runtime(require_cuda=True)", command[-1])
                events.append("gpu")
                if gpu_error:
                    raise subprocess.CalledProcessError(1, command)
            if "src.experiments" in command:
                events.append("train")
                config = json.loads(
                    Path(command[command.index("--config") + 1]).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(config, self.module.winner_config(self.local_runs))
                self.assertEqual(kwargs["cwd"], self.root)
                run_dir = self.destination / "experiments/synthetic_run"
                run_dir.mkdir()
                (run_dir / "run.json").write_text(
                    '{"status":"complete"}', encoding="utf-8"
                )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        def save(*_args):
            events.append("save")
            return archive

        def preview(*_args):
            events.append("preview")
            if preview_error:
                raise RuntimeError("Browser unavailable")

        self.colab.files.download.side_effect = lambda _path: events.append("download")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.dict(sys.modules, {"google": self.google, "google.colab": self.colab})
        )
        stack.enter_context(
            patch.object(
                self.module,
                "prepare_storage",
                return_value=(self.local_runs, self.destination),
            )
        )
        stack.enter_context(
            patch.object(self.module.subprocess, "run", side_effect=run)
        )
        stack.enter_context(patch.object(self.module, "save_bundle", side_effect=save))
        stack.enter_context(
            patch.object(self.module, "show_validation_preview", side_effect=preview)
        )
        return events, commands, archive

    def test_repaired_packages_use_fresh_gpu_process_then_one_training(self):
        events, commands, archive = self._workflow(bootstrap_code=75)
        self.assertEqual(self.module.train_demo_on_colab(), archive)
        self.assertEqual(
            events, ["bootstrap", "gpu", "train", "save", "preview", "download"]
        )
        self.assertEqual(sum("src.experiments" in command for command in commands), 1)
        self.assertFalse(any("src.evaluate" in command for command in commands))

    def test_bootstrap_failure_prevents_gpu_and_training(self):
        events, _, _ = self._workflow(bootstrap_code=2)
        with self.assertRaisesRegex(RuntimeError, "Dependency setup failed"):
            self.module.train_demo_on_colab()
        self.assertEqual(events, ["bootstrap"])

    def test_gpu_failure_prevents_training(self):
        events, _, _ = self._workflow(gpu_error=True)
        with self.assertRaises(subprocess.CalledProcessError):
            self.module.train_demo_on_colab()
        self.assertEqual(events, ["bootstrap", "gpu"])

    def test_missing_checkout_files_fail_before_bootstrap(self):
        (self.root / "src/experiments.py").unlink()
        events, _, _ = self._workflow()
        with self.assertRaisesRegex(RuntimeError, "missing required training files"):
            self.module.train_demo_on_colab()
        self.assertEqual(events, [])

    def test_saved_bundle_survives_optional_preview_failure(self):
        events, _, archive = self._workflow(preview_error=True)
        self.assertEqual(self.module.train_demo_on_colab(), archive)
        self.assertLess(events.index("save"), events.index("preview"))
        self.assertIn("download", events)

    def test_bundle_contains_one_model_and_metadata_but_no_cache(self):
        run_dir = self.destination / "experiments/synthetic"
        model_dir = run_dir / "models" / self.module.MODEL
        required = [
            model_dir / "checkpoints/best_model.pth",
            model_dir / "checkpoints/best_model.json",
            model_dir / "metrics/metrics.json",
            run_dir / "config.json",
            run_dir / "resolved_config.json",
        ]
        for path in required:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"{}")
        (run_dir / "run.json").write_text(
            '{"status":"complete","git_revision":"synthetic"}', encoding="utf-8"
        )
        cache = self.local_runs / "cache"
        cache.mkdir()
        (cache / "excluded.bin").write_bytes(b"not a model artifact")
        archive = self.module.save_bundle(
            run_dir, self.local_runs, self.destination, self.root
        )
        with ZipFile(archive) as zipped:
            names = zipped.namelist()
            self.assertTrue(any(name.endswith("best_model.pth") for name in names))
            self.assertTrue(any(name.endswith("config.json") for name in names))
            self.assertTrue(any(name.endswith("split_manifest.json") for name in names))
            self.assertFalse(
                any("cache" in name or "excluded.bin" in name for name in names)
            )
            receipt = json.loads(zipped.read("synthetic/demo_bundle.json"))
        self.assertEqual(receipt["test_access"], "locked")
        self.assertEqual(
            receipt["checkpoint_sha256"], hashlib.sha256(b"{}").hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
