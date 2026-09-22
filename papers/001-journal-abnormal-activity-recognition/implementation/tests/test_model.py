"""Focused CPU tests for ConvLSTM architecture preparation."""

from __future__ import annotations

import unittest

import torch

from src.model import (
    ConvLSTM2D,
    ConvLSTMCustom,
    ConvLSTMModel,
    ConvLSTMOriginal,
    ConvLSTMPooledModel,
    ConvLSTMPooledModelV1,
    OriginalPaperConvLSTM,
    StackedConvLSTM,
    count_trainable_parameters,
)


class ConvLSTM2DTests(unittest.TestCase):
    """Check sequence return modes and two-dimensional kernels."""

    def test_return_modes_preserve_spatial_dimensions(self) -> None:
        inputs = torch.randn(2, 3, 4, 5, 7)

        sequence_model = ConvLSTM2D(4, 6, kernel_size=(3, 5), return_sequences=True)
        final_model = ConvLSTM2D(4, 6, kernel_size=3, return_sequences=False)

        self.assertEqual(sequence_model(inputs).shape, (2, 3, 6, 5, 7))
        self.assertEqual(final_model(inputs).shape, (2, 6, 5, 7))


class StackedConvLSTMTests(unittest.TestCase):
    """Check configurable stacks, parameter counts, and validation."""

    def test_two_layers_pass_sequence_to_second_layer(self) -> None:
        model = StackedConvLSTM(
            num_classes=5,
            input_shape=(3, 7, 9),
            convlstm_layers=[(8, (3, 3)), (16, (5, 5))],
            dropout=0.1,
            hidden_classifier_width=12,
        )
        observed_shapes: list[tuple[int, ...]] = []
        hooks = [
            layer.register_forward_hook(
                lambda _module, _inputs, output: observed_shapes.append(
                    tuple(output.shape)
                )
            )
            for layer in model.convlstm_layers
        ]

        try:
            logits = model(torch.randn(2, 4, 3, 7, 9))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(observed_shapes, [(2, 4, 8, 7, 9), (2, 16, 7, 9)])
        self.assertEqual(logits.shape, (2, 5))

    def test_one_two_and_three_layer_models_have_distinct_counts(self) -> None:
        specifications = [
            [(4, (3, 3))],
            [(4, (3, 3)), (6, (3, 3))],
            [(4, (3, 3)), (6, (3, 3)), (8, (5, 5))],
        ]
        counts: list[int] = []

        for layers in specifications:
            with self.subTest(layers=layers):
                model = StackedConvLSTM(
                    num_classes=3,
                    input_shape=(3, 5, 7),
                    convlstm_layers=layers,
                    dropout=0.0,
                )
                features = model.forward_features(torch.randn(1, 2, 3, 5, 7))
                logits = model(torch.randn(1, 2, 3, 5, 7))
                self.assertEqual(features.shape[-2:], (5, 7))
                self.assertEqual(logits.shape, (1, 3))
                counts.append(count_trainable_parameters(model))

        self.assertEqual(len(set(counts)), 3)

    def test_invalid_layer_specifications_fail_before_forward(self) -> None:
        cases = [
            ([], "at least one layer"),
            ([(0, (3, 3))], "filters"),
            ([(8, (2, 3))], "odd"),
            ([(8, (0, 3))], "positive"),
            ([[8, (3, 3)]], "convlstm_layers\\[0\\]"),
            ([(8, 3)], "kernel_size"),
        ]

        for layers, message in cases:
            with self.subTest(layers=layers):
                with self.assertRaisesRegex(ValueError, message):
                    StackedConvLSTM(num_classes=3, convlstm_layers=layers)  # type: ignore[arg-type]


class OriginalPaperConvLSTMTests(unittest.TestCase):
    """Check the faithful baseline without allocating its full dense layer."""

    def test_reduced_shape_baseline_preserves_time_before_flatten(self) -> None:
        model = OriginalPaperConvLSTM(
            num_classes=4, input_shape=(3, 4, 5), sequence_length=2
        )
        observed: list[tuple[int, ...]] = []
        hook = model.conv_post.register_forward_hook(
            lambda _module, _inputs, output: observed.append(tuple(output.shape))
        )
        try:
            logits = model(torch.randn(2, 2, 3, 4, 5))
        finally:
            hook.remove()

        self.assertEqual(observed, [(4, 16, 4, 5)])
        self.assertEqual(model.fc1.in_features, 2 * 16 * 4 * 5)
        self.assertEqual(logits.shape, (2, 4))
        self.assertFalse(
            any(isinstance(layer, torch.nn.ReLU) for layer in model.modules())
        )

    def test_default_baseline_matches_published_trainable_parameter_count(self) -> None:
        with torch.device("meta"):
            model = OriginalPaperConvLSTM(num_classes=11)

        self.assertTrue(model.convlstm.return_sequences)
        self.assertEqual(model.fc1.in_features, 50 * 16 * 50 * 50)
        self.assertEqual(count_trainable_parameters(model), 512_197_467)


class CompatibilityTests(unittest.TestCase):
    """Protect existing model imports used by current scripts."""

    def test_existing_public_model_classes_remain_available(self) -> None:
        existing_classes = (
            ConvLSTMOriginal,
            ConvLSTMPooledModel,
            ConvLSTMPooledModelV1,
            ConvLSTMModel,
            ConvLSTMCustom,
        )
        for model_class in existing_classes:
            self.assertTrue(issubclass(model_class, torch.nn.Module))


if __name__ == "__main__":
    unittest.main()
