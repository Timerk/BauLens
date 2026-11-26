"""Transfer-learning backbones with a single crack logit, freezing helpers and device choice."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torchvision import models  # type: ignore[import-untyped]

type Architecture = Literal["resnet18", "efficientnet_b0"]
type DeviceChoice = Literal["auto", "cpu", "cuda"]


def weights_name(architecture: Architecture, pretrained: bool) -> str | None:
    """Name of the torchvision weights enum member used, recorded in the model metadata."""
    if not pretrained:
        return None
    match architecture:
        case "resnet18":
            return f"ResNet18_Weights.{models.ResNet18_Weights.DEFAULT.name}"
        case "efficientnet_b0":
            return f"EfficientNet_B0_Weights.{models.EfficientNet_B0_Weights.DEFAULT.name}"


def build_model(architecture: Architecture, pretrained: bool) -> nn.Module:
    """Backbone with ImageNet weights (optional) and a fresh one-logit head."""
    match architecture:
        case "resnet18":
            resnet: nn.Module = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            fc = resnet.get_submodule("fc")
            if not isinstance(fc, nn.Linear):
                raise TypeError("unexpected ResNet head layout")
            resnet.fc = nn.Linear(fc.in_features, 1)
            return resnet
        case "efficientnet_b0":
            effnet: nn.Module = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            classifier = effnet.get_submodule("classifier")
            if not isinstance(classifier, nn.Sequential):
                raise TypeError("unexpected EfficientNet classifier layout")
            last = classifier[-1]
            if not isinstance(last, nn.Linear):
                raise TypeError("unexpected EfficientNet classifier layout")
            classifier[-1] = nn.Linear(last.in_features, 1)
            return effnet


def feature_blocks(model: nn.Module, architecture: Architecture) -> list[nn.Module]:
    """Backbone stages from input to output; the last ones are unfrozen for fine-tuning."""
    match architecture:
        case "resnet18":
            return [getattr(model, name) for name in ("layer1", "layer2", "layer3", "layer4")]
        case "efficientnet_b0":
            features = model.get_submodule("features")
            return list(features.children())[1:]


def head(model: nn.Module, architecture: Architecture) -> nn.Module:
    match architecture:
        case "resnet18":
            return model.get_submodule("fc")
        case "efficientnet_b0":
            return model.get_submodule("classifier")


def gradcam_layer(model: nn.Module, architecture: Architecture) -> nn.Module:
    """Last convolutional block, whose activations Grad-CAM weights by their gradients."""
    return feature_blocks(model, architecture)[-1]


def configure_trainable(
    model: nn.Module, architecture: Architecture, finetune_blocks: int
) -> list[nn.Module]:
    """Freeze everything except the head and the last ``finetune_blocks`` stages."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = feature_blocks(model, architecture)
    trainable = [*blocks[len(blocks) - finetune_blocks :], head(model, architecture)]
    for module in trainable:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return trainable


def set_train_mode(model: nn.Module, trainable: Sequence[nn.Module]) -> None:
    """Train mode only for trainable modules; frozen BatchNorm layers keep ImageNet statistics."""
    model.eval()
    for module in trainable:
        module.train()


def resolve_device(choice: DeviceChoice) -> torch.device:
    """``auto`` picks CUDA when available (ROCm builds of PyTorch expose it as ``cuda``)."""
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    if choice == "cpu" or (choice == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
