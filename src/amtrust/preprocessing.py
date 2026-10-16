"""Image preprocessing shared by training, evaluation, robustness, baselines and Grad-CAM.

OT frames (2000 x 2000 RGB) are resized directly to 512 x 512.  PB frames
(1280 x 1024 greyscale) are resized so that the long side is 512 pixels
(512 x 410), centre padded with the constant 128 to 512 x 512 and replicated to
three channels.  Resizing is bilinear with antialiasing.  Pixel values are
scaled to [0, 1] and normalised with the ImageNet mean and standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

TARGET = 512
PAD_VALUE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Geometry:
    """Placement of a resized frame on the 512 x 512 canvas."""

    width: int
    height: int
    new_width: int
    new_height: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def content_area(self) -> int:
        return self.new_width * self.new_height


def geometry(modality: str, width: int, height: int, target: int = TARGET) -> Geometry:
    """Return the resize and padding geometry for a native frame size."""
    if modality == "OT":
        return Geometry(width, height, target, target, 0, 0, 0, 0)
    if modality == "PB":
        scale = target / max(width, height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        rem_w, rem_h = target - new_width, target - new_height
        left, top = rem_w // 2, rem_h // 2
        return Geometry(width, height, new_width, new_height, left, top, rem_w - left, rem_h - top)
    raise ValueError(f"Unknown modality: {modality}")


def _resized_pil(image: Image.Image, modality: str, target: int, pad_value: int) -> Image.Image:
    if modality == "OT":
        if image.mode != "RGB":
            raise RuntimeError(f"Unexpected OT image mode: {image.mode}")
        return TF.resize(image, [target, target], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if modality == "PB":
        if image.mode != "L":
            raise RuntimeError(f"Unexpected PB image mode: {image.mode}")
        geo = geometry("PB", image.size[0], image.size[1], target)
        resized = TF.resize(image, [geo.new_height, geo.new_width],
                            interpolation=InterpolationMode.BILINEAR, antialias=True)
        return TF.pad(resized, [geo.left, geo.top, geo.right, geo.bottom], fill=pad_value)
    raise ValueError(f"Unknown modality: {modality}")


def load_uint8_tensor(path, modality: str, target: int = TARGET, pad_value: int = PAD_VALUE) -> torch.Tensor:
    """Return the resized (and for PB padded) image as a uint8 tensor [3, H, W]."""
    with Image.open(path) as image:
        canvas = _resized_pil(image, modality, target, pad_value)
        tensor = TF.pil_to_tensor(canvas)
    if modality == "PB":
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def normalise(tensor01: torch.Tensor, mean=MEAN, std=STD) -> torch.Tensor:
    """Apply ImageNet normalisation to a float tensor in [0, 1]."""
    return TF.normalize(tensor01, mean=list(mean), std=list(std))


def preprocess_image(path, modality: str, target: int = TARGET, pad_value: int = PAD_VALUE,
                     mean=MEAN, std=STD) -> torch.Tensor:
    """Return the model input tensor [3, 512, 512] used for training and evaluation."""
    tensor = load_uint8_tensor(path, modality, target, pad_value).to(dtype=torch.float32) / 255.0
    return normalise(tensor, mean, std)


def base_array(path, modality: str, target: int = TARGET, pad_value: int = PAD_VALUE) -> np.ndarray:
    """Return the resized (and padded) image as float32 [H, W, 3] in [0, 1].

    This is the array on which image corruptions are applied.
    """
    with Image.open(path) as image:
        canvas = _resized_pil(image, modality, target, pad_value)
        if modality == "OT":
            return np.asarray(canvas, dtype=np.float32) / 255.0
        gray = np.asarray(canvas, dtype=np.float32) / 255.0
        return np.repeat(gray[..., None], 3, axis=2)


def array_batch_to_input(batch_hwc: np.ndarray, mean=MEAN, std=STD) -> torch.Tensor:
    """Convert a float32 batch [N, H, W, 3] in [0, 1] into a normalised tensor [N, 3, H, W]."""
    tensor = torch.from_numpy(np.ascontiguousarray(batch_hwc)).permute(0, 3, 1, 2)
    return TF.normalize(tensor, mean=list(mean), std=list(std))
