"""Reconstructible diagnostic models; no pretrained downloads or AAD changes."""

import torch
from torchvision.models import video

from .model import CustomConvLSTM, PaperConvLSTM, custom_model_from_checkpoint

BASELINES = ("r3d_18", "mc3_18", "r2plus1d_18")


class VideoClassifier(torch.nn.Module):
    def __init__(self, name, num_classes, input_shape, sequence_length):
        super().__init__()
        self.specification = dict(
            model_type="NotebookVideoClassifier",
            name=name,
            num_classes=num_classes,
            input_shape=list(input_shape),
            sequence_length=sequence_length,
        )
        if name == "paper_convlstm_published":
            if tuple(input_shape) != (3, 50, 50) or sequence_length != 50:
                raise ValueError(
                    "Published topology requires native 50-frame/50x50 inputs"
                )
            self.network = PaperConvLSTM(
                num_classes,
                input_shape=tuple(input_shape),
                sequence_length=sequence_length,
            )
        elif name in BASELINES:
            self.network = getattr(video, name)(weights=None, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown diagnostic architecture: {name}")

    def forward(self, inputs):
        if self.specification["name"] in BASELINES:
            inputs = inputs.permute(0, 2, 1, 3, 4)
        return self.network(inputs)

    def configuration(self):
        return dict(self.specification)


def build_model(spec, num_classes, input_shape, sequence_length):
    if spec["name"] == "custom":
        return CustomConvLSTM(
            num_classes,
            layers=[(n, tuple(k)) for n, k in spec["layers"]],
            dropout=spec["dropout"],
            hidden_classifier_width=spec.get("hidden_classifier_width"),
        )
    return VideoClassifier(spec["name"], num_classes, input_shape, sequence_length)


def from_checkpoint(checkpoint):
    config = checkpoint["model_config"]
    if config.get("model_type") != "NotebookVideoClassifier":
        return custom_model_from_checkpoint(checkpoint)
    # Allocate without random initialization: especially important for the 512M model.
    with torch.device("meta"):
        model = VideoClassifier(
            config["name"],
            config["num_classes"],
            config["input_shape"],
            config["sequence_length"],
        )
    model.load_state_dict(checkpoint["model_state_dict"], assign=True)
    return model
