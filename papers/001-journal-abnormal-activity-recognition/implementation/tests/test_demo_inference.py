"""Local synthetic checks only: no training, dataset downloads or research scores."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.demo_inference import (
    BUNDLED_CHECKPOINT,
    DemoPredictor,
    MAX_UPLOAD_BYTES,
    RANDOM_WARNING,
    WINNER_INPUT,
    WINNER_LAYERS,
)
from src.temporal_sampling import load_timestamp_clip, video_plan
from src.utils import write_video_torchvision


class DemoInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.random = DemoPredictor(random_weights=True, device="cpu")
        cls.metadata = {
            "class_names": cls.random.class_names,
            "experiment_config": {
                **WINNER_INPUT,
                "convlstm_layers": WINNER_LAYERS,
                "hidden_classifier_width": None,
            },
            "split": {"manifest_hash": "synthetic-split-not-research"},
        }
        cls.saved = {
            "model_config": cls.random.model.configuration(),
            "model_state_dict": cls.random.model.state_dict(),
            "experiment_config": cls.metadata["experiment_config"],
            "checkpoint_role": "validation_selected_lowest_loss",
            "selection_partition": "validation",
            "split_manifest_hash": "synthetic-split-not-research",
        }
        cls.checkpoint = cls.root / "run/models/custom/checkpoints/best_model.pth"
        cls.checkpoint.parent.mkdir(parents=True)
        torch.save(cls.saved, cls.checkpoint)
        (cls.root / "run/config.json").write_text(
            json.dumps(cls.metadata), encoding="utf-8"
        )
        cls.video = cls.root / "synthetic-short.mp4"
        frames = torch.linspace(0, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
        write_video_torchvision(frames, cls.video, fps=8)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        cls.workspace.cleanup()

    def test_random_mode_is_explicit_and_unmistakable(self):
        self.assertEqual(self.random.mode, "random")
        self.assertEqual(self.random.status, RANDOM_WARNING)
        self.assertFalse(self.random.model.training)
        self.assertEqual(
            sum(p.numel() for p in self.random.model.parameters()), 1336459
        )
        with self.assertRaisesRegex(ValueError, "checkpoint is required"):
            DemoPredictor(device="cpu")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            DemoPredictor(checkpoint=self.checkpoint, random_weights=True, device="cpu")
        with self.assertRaisesRegex(FileNotFoundError, "No random fallback"):
            DemoPredictor(checkpoint=self.root / "missing.pth", device="cpu")

    def test_real_checkpoint_loads_once_with_inferred_run_metadata(self):
        with patch("src.demo_inference.torch.load", wraps=torch.load) as loader:
            predictor = DemoPredictor(checkpoint=self.checkpoint, device="cpu")
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(predictor.mode, "trained")
            self.assertIn("not the verified original", predictor.status)
            self.assertEqual(predictor.class_names, self.metadata["class_names"])
            self.assertFalse(predictor.model.training)

    def test_missing_metadata_is_not_silently_guessed(self):
        checkpoint = self.root / "isolated/best_model.pth"
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(self.saved, checkpoint)
        with self.assertRaisesRegex(ValueError, "Matching run config.json"):
            DemoPredictor(checkpoint=checkpoint, device="cpu")

    def test_lfs_pointer_is_rejected_before_metadata_or_torch_loading(self):
        pointer = self.root / "lfs_pointer.pth"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{'0' * 64}\n"
            "size 16052597\n",
            encoding="utf-8",
        )
        with patch(
            "src.demo_inference.torch.load",
            side_effect=AssertionError("An LFS pointer must never reach torch.load"),
        ) as loader, patch(
            "src.demo_inference._read_metadata",
            side_effect=AssertionError("The LFS check must precede metadata loading"),
        ):
            with self.assertRaisesRegex(ValueError, "Git LFS pointer.*git lfs pull"):
                DemoPredictor(checkpoint=pointer, device="cpu")
        loader.assert_not_called()

    def test_missing_bundled_model_gives_lfs_recovery_command(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Bundled.*git lfs pull"):
                DemoPredictor(checkpoint=BUNDLED_CHECKPOINT, device="cpu")

    def test_metadata_class_order_and_preprocessing_must_match(self):
        reversed_labels = copy.deepcopy(self.metadata)
        reversed_labels["class_names"].reverse()
        with self.assertRaisesRegex(ValueError, "class names/order"):
            DemoPredictor(self.checkpoint, reversed_labels, device="cpu")
        for key, value in (
            ("target_fps", 8),
            ("sampling_version", "legacy"),
            ("height", 32),
        ):
            metadata = copy.deepcopy(self.metadata)
            metadata["experiment_config"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                DemoPredictor(self.checkpoint, metadata, device="cpu")

    def test_optional_hash_and_split_hash_are_checked(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            DemoPredictor(self.checkpoint, metadata, device="cpu")
        metadata["checkpoint_sha256"] = hashlib.sha256(
            self.checkpoint.read_bytes()
        ).hexdigest()
        self.assertEqual(
            DemoPredictor(self.checkpoint, metadata, device="cpu").checkpoint_sha256,
            metadata["checkpoint_sha256"],
        )
        metadata["split"]["manifest_hash"] = "another-split"
        with self.assertRaisesRegex(ValueError, "split hashes"):
            DemoPredictor(self.checkpoint, metadata, device="cpu")

    def test_checkpoint_hash_guards_work_without_python311_file_digest(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["checkpoint_sha256"] = hashlib.sha256(
            self.checkpoint.read_bytes()
        ).hexdigest()
        pointer = self.root / "lfs_without_file_digest.pth"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{'0' * 64}\nsize 16052597\n",
            encoding="utf-8",
        )
        missing = object()
        original = getattr(hashlib, "file_digest", missing)
        if original is not missing:
            delattr(hashlib, "file_digest")
        try:
            self.assertFalse(hasattr(hashlib, "file_digest"))
            predictor = DemoPredictor(self.checkpoint, metadata, device="cpu")
            self.assertEqual(predictor.mode, "trained")
            self.assertEqual(predictor.checkpoint_sha256, metadata["checkpoint_sha256"])
            metadata["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                DemoPredictor(self.checkpoint, metadata, device="cpu")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer.*git lfs pull"):
                DemoPredictor(pointer, device="cpu")
        finally:
            if original is not missing:
                setattr(hashlib, "file_digest", original)

    def test_incompatible_checkpoint_fails_before_inference(self):
        for field, value, message in (
            (
                "model_config",
                {**self.saved["model_config"], "num_classes": 2},
                "architecture",
            ),
            (
                "experiment_config",
                {**self.saved["experiment_config"], "sequence_length": 16},
                "sequence_length",
            ),
            ("selection_partition", "test", "validation loss"),
            ("model_state_dict", {}, "tensors"),
            (
                "class_names",
                list(reversed(self.metadata["class_names"])),
                "class names/order",
            ),
        ):
            checkpoint = {**self.saved, field: value}
            with self.subTest(field=field), patch(
                "src.demo_inference.torch.load", return_value=checkpoint
            ), self.assertRaisesRegex(ValueError, message):
                DemoPredictor(self.checkpoint, self.metadata, device="cpu")

    def test_short_low_fps_video_matches_campaign_sampler_and_pads(self):
        expected_plan = video_plan(self.video, 16, 8)
        expected_frames = load_timestamp_clip(self.video, expected_plan, (64, 64))
        logits = torch.arange(11, dtype=torch.float32).unsqueeze(0)
        with patch.object(self.random.model, "forward", return_value=logits) as forward:
            result = self.random.predict(self.video)
        self.assertTrue(torch.equal(forward.call_args.args[0][0], expected_frames))
        self.assertEqual(result["sampling"], expected_plan)
        self.assertGreater(result["sampling"]["padding_fraction"], 0)
        self.assertGreater(result["sampling"]["duplicate_fraction"], 0)
        self.assertEqual(list(result["probabilities"]), self.random.class_names)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=6)
        self.assertEqual(result["prediction"], self.random.class_names[-1])
        self.assertIn(RANDOM_WARNING, result["status"])
        self.assertIn("not the whole video", result["status"])

    def test_actual_cpu_forward_returns_finite_probabilities(self):
        result = self.random.predict(self.video)
        self.assertEqual(len(result["probabilities"]), 11)
        self.assertTrue(
            all(0 <= value <= 1 for value in result["probabilities"].values())
        )
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=6)

    def test_bad_missing_empty_and_oversized_videos_fail_clearly(self):
        empty = self.root / "empty.mp4"
        empty.touch()
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"this is not a video")
        for video, message in (
            (None, "Choose a local"),
            (self.root / "missing.mp4", "does not exist"),
            (empty, "empty"),
            (corrupt, "Cannot decode"),
        ):
            with self.subTest(video=video), self.assertRaisesRegex(ValueError, message):
                self.random.predict(video)
        with patch("src.demo_inference.MAX_UPLOAD_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "200 MiB"):
                self.random.predict(self.video)
        self.assertEqual(MAX_UPLOAD_BYTES, 200 * 1024 * 1024)

    def test_nonfinite_model_output_is_rejected(self):
        with patch.object(
            self.random.model, "forward", return_value=torch.full((1, 11), float("nan"))
        ), self.assertRaisesRegex(ValueError, "invalid class scores"):
            self.random.predict(self.video)


if __name__ == "__main__":
    unittest.main()
