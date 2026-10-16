"""Grad-CAM generation and quantitative localisation metrics.

Grad-CAM targets the Defect logit (class index 1) for every sample, whatever
the predicted class.  Two layers per architecture (primary and complementary)
are hooked and both maps come from one backward pass.  A map is
``ReLU(sum_c mean_spatial(dlogit/dA_c) * A_c)``, bilinearly upsampled to
512 x 512 with ``align_corners=False`` and min-max normalised per map:
``(cam - min) / (max - min + 1e-12)``.

Reference masks come from the valid XML boxes of a Defect image projected onto
the 512 x 512 canvas.  A pixel (x, y) belongs to the mask when its centre
(x + 0.5, y + 0.5) lies in [x1, x2) x [y1, y2) of any box.

Cohort: for each of the eight tasks, the Defect test samples with at least one
valid box are sorted by
SHA-256("METAL_AM_XAI_36A_CANDIDATE_V1|<protocol>|<modality>|<fold id>|<sample id>")
(ties by sample id) and the first 30 are taken.  The cohort does not use any
model output and is shared by the 12 runs of a task.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .models import resolve_layer
from .preprocessing import geometry

MAP_SIZE = 512
N_PIXELS = MAP_SIZE * MAP_SIZE


def top_k(fraction: float = 0.10, n_pixels: int = N_PIXELS) -> int:
    """Number of pixels in the top fraction, ceil(fraction * n)."""
    return int(math.ceil(fraction * n_pixels))


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def normalise_map(cam: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cam = cam - cam.min()
    return cam / (cam.max() + eps)


class TwoLayerGradCAM:
    """Grad-CAM on several named layers from a single backward pass."""

    def __init__(self, model: torch.nn.Module, layer_names: dict):
        self.model = model
        self.activations, self.gradients, self.handles = {}, {}, []
        for key, name in layer_names.items():
            layer = resolve_layer(model, name)
            self.handles.append(layer.register_forward_hook(self._forward(key)))
            self.handles.append(layer.register_full_backward_hook(self._backward(key)))

    def _forward(self, key):
        def hook(module, inputs, output):
            self.activations[key] = output
        return hook

    def _backward(self, key):
        def hook(module, grad_input, grad_output):
            self.gradients[key] = grad_output[0]
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, input_tensor: torch.Tensor, target_index: int = 1, normalise: bool = True,
                 size: int = MAP_SIZE) -> tuple[dict, np.ndarray]:
        """Return ({layer key: map [size, size]}, logits) for a single image [1, 3, H, W]."""
        self.activations.clear()
        self.gradients.clear()
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        if tuple(output.shape) != (1, 2):
            raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}")
        output[0, target_index].backward()
        maps = {}
        for key in self.activations:
            activation, gradient = self.activations[key], self.gradients[key]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1))
            cam = F.interpolate(cam.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False)[0, 0]
            cam = cam.detach().cpu().numpy()
            maps[key] = normalise_map(cam) if normalise else cam
        return maps, output.detach().cpu().numpy().astype(np.float64)[0]


def layers_for_run(run_id: str, architecture: str, config: dict) -> tuple[dict, bool]:
    """Layer names and normalisation flag for a run, honouring configured overrides."""
    xai_cfg = config["xai"]
    layers = dict(xai_cfg["layers"][architecture])
    normalise = bool(xai_cfg["normalise_maps"])
    if xai_cfg.get("apply_run_layer_overrides") and run_id in xai_cfg.get("run_layer_overrides", {}):
        override = xai_cfg["run_layer_overrides"][run_id]
        layers = {"primary": override["primary"], "complementary": override["complementary"]}
        normalise = bool(override.get("normalise_maps", normalise))
    return layers, normalise


# ---------------------------------------------------------------------------
# Cohort and reference boxes
# ---------------------------------------------------------------------------
def selection_key(task: tuple[str, str, str], sample_id: str, namespace: str) -> str:
    return hashlib.sha256("|".join((namespace, *task, sample_id)).encode("utf-8")).hexdigest()


def select_cohort(master: pd.DataFrame, valid_box_ids: set, namespace: str, per_task: int = 30) -> pd.DataFrame:
    """First ``per_task`` eligible Defect test samples of each task by selection key."""
    rows = []
    test = master[(master["partition"] == "test") & (master["binary_label"] == 1)]
    for task, sub in sorted(test.groupby(["protocol", "modality", "fold_id"]), key=lambda item: item[0]):
        eligible = [sid for sid in sub["benchmark_sample_id"] if sid in valid_box_ids]
        ordered = sorted(eligible, key=lambda sid: (selection_key(task, sid, namespace), sid))
        if len(ordered) < per_task:
            raise RuntimeError(f"Task {task} has fewer than {per_task} eligible samples.")
        for position, sid in enumerate(ordered[:per_task], start=1):
            rows.append({"protocol": task[0], "modality": task[1], "fold_id": task[2],
                         "cohort_position": position, "benchmark_sample_id": sid,
                         "selection_key": selection_key(task, sid, namespace)})
    return pd.DataFrame(rows)


def project_box(box, modality: str, width: int, height: int, mode: str = "resized_extent", size: int = MAP_SIZE):
    """Map a native (xmin, ymin, xmax, ymax) box onto the 512 x 512 canvas.

    ``resized_extent`` scales x by new_width / width and y by new_height / height
    (the rounded resized extent) and adds the padding offsets; this is the
    projection of the reference cohort.  ``long_side`` scales both axes by
    size / max(width, height).
    """
    geo = geometry(modality, width, height, size)
    if mode == "resized_extent":
        sx, sy = geo.new_width / width, geo.new_height / height
    elif mode == "long_side":
        sx = sy = size / max(width, height)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
    x0, y0, x1, y1 = (float(v) for v in box)
    return [x0 * sx + geo.left, y0 * sy + geo.top, x1 * sx + geo.left, y1 * sy + geo.top]


def continuous_union_area(boxes) -> float:
    """Exact area of the union of axis-aligned rectangles."""
    xs = sorted({x for box in boxes for x in (box[0], box[2])})
    total = 0.0
    for xa, xb in zip(xs[:-1], xs[1:]):
        width = xb - xa
        if width <= 0:
            continue
        intervals = sorted((y0, y1) for x0, y0, x1, y1 in boxes if x0 < xb and x1 > xa)
        if not intervals:
            continue
        start, end = intervals[0]
        covered = 0.0
        for y0, y1 in intervals[1:]:
            if y0 <= end:
                end = max(end, y1)
            else:
                covered += end - start
                start, end = y0, y1
        covered += end - start
        total += width * covered
    return total


def pixel_centre_mask(boxes, size: int = MAP_SIZE) -> np.ndarray:
    """Boolean mask of pixels whose centres fall inside any half-open box."""
    centres = np.arange(size, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(centres, centres)
    mask = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask |= (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    return mask


def boxes_json(boxes) -> str:
    return json.dumps([list(map(float, b)) for b in boxes], separators=(",", ":"))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def threshold_free_metrics(cam: np.ndarray, mask: np.ndarray, reference_area_fraction: float,
                           top_fraction: float = 0.10) -> dict:
    """Energy fraction, enrichment ratio, pointing game, spatial entropy and top-k energy.

    All values are None for an all-zero map.  The enrichment ratio divides the
    discrete-mask energy fraction by the continuous reference area fraction
    (union area / unpadded content area).
    """
    cam = np.asarray(cam, dtype=np.float32)
    if float(cam.max()) == 0.0 and float(cam.min()) == 0.0:
        return {"zero_map": True, "reference_energy_fraction": None, "energy_enrichment_ratio": None,
                "pointing_game_hit": None, "normalized_spatial_entropy": None, "top10_energy_fraction": None}
    total = float(np.sum(cam, dtype=np.float64))
    inside = float(np.sum(cam[mask], dtype=np.float64))
    fraction = inside / total
    maximum = float(cam.max())
    hit = int(np.any((cam == maximum) & mask))
    positive = cam[cam > 0].astype(np.float64) / total
    n_pixels = cam.size
    entropy = float(-np.sum(positive * np.log(positive), dtype=np.float64) / math.log(n_pixels))
    k = top_k(top_fraction, n_pixels)
    flat = cam.reshape(-1)
    top_values = np.partition(flat, n_pixels - k)[n_pixels - k:]
    return {
        "zero_map": False,
        "reference_energy_fraction": fraction,
        "energy_enrichment_ratio": fraction / reference_area_fraction,
        "pointing_game_hit": hit,
        "normalized_spatial_entropy": entropy,
        "top10_energy_fraction": float(np.sum(top_values, dtype=np.float64) / total),
    }


def overlap_metrics(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.50) -> dict:
    """IoU, Dice, precision and recall of ``cam >= threshold`` against the reference mask."""
    predicted = np.asarray(cam) >= threshold
    predicted_area = int(np.count_nonzero(predicted))
    reference_area = int(np.count_nonzero(mask))
    if reference_area == 0:
        raise ValueError("Empty reference mask.")
    intersection = int(np.count_nonzero(predicted & mask))
    union = int(np.count_nonzero(predicted | mask))
    return {
        "iou": intersection / union,
        "dice": 2.0 * intersection / (predicted_area + reference_area),
        "precision": 0.0 if predicted_area == 0 else intersection / predicted_area,
        "recall": intersection / reference_area,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cosine similarity of two maps; None when either norm is zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.sum(a * a)), float(np.sum(b * b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.sum(a * b) / (math.sqrt(na) * math.sqrt(nb)))


THRESHOLD_FREE = ("reference_energy_fraction", "energy_enrichment_ratio", "pointing_game_hit",
                  "normalized_spatial_entropy", "top10_energy_fraction", "primary_complementary_cosine")
OVERLAP = ("iou_050", "dice_050", "precision_050", "recall_050")


def aggregate(samples: pd.DataFrame, metrics, ddof: int = 1) -> pd.DataFrame:
    """Sample -> run mean -> structural cell (seed mean) -> group.

    A group is architecture x protocol x modality.  Protocol R has one cell, so
    its value is the seed mean and its SD is undefined.  Protocol S reports the
    mean and sample SD across the three held-out-build cells.
    """
    run_keys = ["architecture", "protocol", "modality", "fold_id", "seed"]
    run_means = samples.groupby(run_keys)[list(metrics)].mean().reset_index()
    cell_keys = ["architecture", "protocol", "modality", "fold_id"]
    cells = run_means.groupby(cell_keys)[list(metrics)].mean().reset_index()
    seeds_per_cell = run_means.groupby(cell_keys)["seed"].nunique().reset_index(name="seeds")
    cells = cells.merge(seeds_per_cell, on=cell_keys)
    rows = []
    for key, sub in cells.groupby(["architecture", "protocol", "modality"], sort=True):
        record = {"architecture": key[0], "protocol": key[1], "modality": key[2], "n_cell": int(len(sub)),
                  "seed_coverage": f"{int(sub['seeds'].sum())}/{3 * len(sub)}"}
        for metric in metrics:
            values = sub[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_sd"] = float(np.std(values, ddof=ddof)) if values.size > ddof else float("nan")
        rows.append(record)
    return pd.DataFrame(rows)
