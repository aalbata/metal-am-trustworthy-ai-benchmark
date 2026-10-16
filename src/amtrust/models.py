"""Model construction with ImageNet weights and a two-class head.

Each network is built with ``weights=None``, the ImageNet state dict is loaded
with ``strict=True`` and the classification layer is then replaced by a new
``nn.Linear(in_features, 2)``.  This order matches the original training code,
so the random initialisation of the new head is drawn from the global torch
generator after the full backbone initialisation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .data import file_digest

BUILDERS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "mobilenet_v3_large": models.mobilenet_v3_large,
}

WEIGHT_ENUMS = {
    "resnet18": ("ResNet18_Weights", "IMAGENET1K_V1"),
    "resnet50": ("ResNet50_Weights", "IMAGENET1K_V2"),
    "efficientnet_b0": ("EfficientNet_B0_Weights", "IMAGENET1K_V1"),
    "mobilenet_v3_large": ("MobileNet_V3_Large_Weights", "IMAGENET1K_V2"),
}

DISPLAY_NAMES = {
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNetV3-Large",
}


def replace_head(model: nn.Module, architecture: str, num_classes: int = 2) -> nn.Module:
    """Replace the ImageNet classification layer with a new linear layer."""
    if architecture in ("resnet18", "resnet50"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "mobilenet_v3_large":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def build_classifier(architecture: str, num_classes: int = 2) -> nn.Module:
    """Return the architecture with a two-class head and random weights."""
    if architecture not in BUILDERS:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return replace_head(BUILDERS[architecture](weights=None), architecture, num_classes)


def pretrained_file(architecture: str, config: dict, weights_dir: Path) -> Path:
    """Return the local ImageNet weights file, downloading it with torchvision if absent.

    The file is stored as ``<weights_dir>/<torchvision filename>`` and its
    SHA-256 is compared with the configured value.
    """
    spec = config["training"]["architectures"][architecture]
    weights_dir = Path(weights_dir)
    path = weights_dir / spec["weights_filename"]
    if not path.is_file():
        enum_name, member = WEIGHT_ENUMS[architecture]
        weights = getattr(getattr(models, enum_name), member)
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.load_state_dict_from_url(weights.url, model_dir=str(weights_dir), progress=True, check_hash=True)
    observed = file_digest(path, "sha256")
    if observed != spec["weights_sha256"]:
        raise RuntimeError(f"ImageNet weights SHA-256 mismatch for {architecture}: {observed}")
    return path


def create_model(architecture: str, weights_path: Path | None, num_classes: int = 2) -> nn.Module:
    """Build the network, load ImageNet weights (if given) and attach a two-class head."""
    if architecture not in BUILDERS:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model = BUILDERS[architecture](weights=None)
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    model = replace_head(model, architecture, num_classes)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def load_trained(architecture: str, state_path: Path, num_classes: int = 2) -> nn.Module:
    """Rebuild a trained classifier from a saved model state file."""
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model = build_classifier(architecture, num_classes)
    model.load_state_dict(state, strict=True)
    return model


def model_state_sha256(model: nn.Module) -> str:
    """SHA-256 over parameter names, dtypes, shapes and bytes of the model state."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def resolve_layer(model: nn.Module, dotted_name: str) -> nn.Module:
    """Return the sub-module addressed by a dotted name such as ``features.6``."""
    modules = dict(model.named_modules())
    if dotted_name not in modules:
        raise KeyError(f"Layer not found: {dotted_name}")
    return modules[dotted_name]
