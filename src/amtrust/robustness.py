"""Generic image corruptions and robustness summaries.

Corruptions are applied to the resized (and for PB padded) image in [0, 1]
before ImageNet normalisation; for PB the padding band is corrupted as well.
Five families with three severities give 15 conditions:

* Gaussian intensity noise, sigma 0.02 / 0.05 / 0.10: one H x W standard
  normal field shared by the three channels, drawn from
  ``numpy.random.Generator(PCG64(seed))`` with ``seed`` equal to the first 8
  hexadecimal digits of SHA-256("ROBUSTNESS_V1|gaussian_intensity_noise|S{k}|<sample id>").
  The field therefore depends only on the severity and the sample.
* Gaussian blur, sigma 0.5 / 1.0 / 2.0 pixels: separable kernel of radius
  ceil(3 sigma) with reflect padding, horizontal pass then vertical pass.
* Brightness attenuation, factor 0.85 / 0.70 / 0.50: ``factor * x``.
* Contrast reduction, factor 0.85 / 0.65 / 0.45:
  ``mean_c + factor * (x - mean_c)`` with a per-image, per-channel mean.
* JPEG compression, quality 80 / 50 / 20: ``rint(clip(x) * 255)`` to uint8 RGB,
  Pillow JPEG with subsampling 0, no optimisation, baseline encoding.

All outputs are clipped to [0, 1] and returned as float32.
"""

from __future__ import annotations

import hashlib
import io
import math

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

SEED_TEMPLATE = "ROBUSTNESS_V1|{family}|{severity_label}|{sample_id}"


def conditions(config: dict) -> list[dict]:
    """The 15 corruption conditions in protocol order."""
    rob = config["robustness"]
    out = []
    for spec in rob["families"]:
        for index, value in enumerate(spec["values"], start=1):
            label = f"S{index}"
            out.append({
                "condition_index": len(out) + 1,
                "family": spec["family"],
                "severity_index": index,
                "severity_label": label,
                "parameter": spec["parameter"],
                "value": value,
                "condition_id": f"{spec['family']}__{label}",
            })
    return out


def noise_seed(severity_label: str, sample_id: str, family: str = "gaussian_intensity_noise", hex_digits: int = 8) -> int:
    """Integer seed for the Gaussian noise field of one sample and severity."""
    material = SEED_TEMPLATE.format(family=family, severity_label=severity_label, sample_id=sample_id)
    return int(hashlib.sha256(material.encode()).hexdigest()[:hex_digits], 16)


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(math.ceil(3.0 * sigma))
    support = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(support * support) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def corrupt_batch(x: np.ndarray, condition: dict, sample_ids) -> np.ndarray:
    """Apply one corruption to a float32 batch [N, H, W, 3] in [0, 1]."""
    family = condition["family"]
    value = condition["value"]
    y = x.astype(np.float64)
    if family == "gaussian_intensity_noise":
        for i, sample_id in enumerate(sample_ids):
            rng = np.random.Generator(np.random.PCG64(noise_seed(condition["severity_label"], sample_id)))
            field = rng.standard_normal(y.shape[1:3], dtype=np.float64)
            y[i] += float(value) * field[..., None]
    elif family == "gaussian_blur":
        kernel = gaussian_kernel(float(value))
        radius = (len(kernel) - 1) // 2
        padded = np.pad(y, ((0, 0), (0, 0), (radius, radius), (0, 0)), mode="reflect")
        y = sum(kernel[j] * padded[:, :, j:j + x.shape[2], :] for j in range(len(kernel)))
        padded = np.pad(y, ((0, 0), (radius, radius), (0, 0), (0, 0)), mode="reflect")
        y = sum(kernel[j] * padded[:, j:j + x.shape[1], :, :] for j in range(len(kernel)))
    elif family == "brightness_attenuation":
        y *= float(value)
    elif family == "contrast_reduction":
        mean = y.mean(axis=(1, 2), keepdims=True)
        y = mean + float(value) * (y - mean)
    elif family == "jpeg_compression":
        decoded = []
        for image in y:
            u8 = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(u8, "RGB").save(buffer, "JPEG", quality=int(value), subsampling=0,
                                            optimize=False, progressive=False)
            buffer.seek(0)
            with Image.open(buffer) as reopened:
                decoded.append(np.asarray(reopened.convert("RGB"), dtype=np.float32) / 255.0)
        return np.stack(decoded).astype(np.float32)
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    return np.clip(y, 0, 1).astype(np.float32)


def run_condition_metrics(labels, predictions, p_defect) -> dict:
    """Macro-F1, balanced accuracy and ROC AUC for one run and condition."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=[0, 1], zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, np.asarray(p_defect, dtype=np.float64))),
    }


METRICS = ("macro_f1", "balanced_accuracy", "roc_auc")


def group_degradation(run_metrics: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Three-seed mean and SD of clean, corrupted and degradation (clean minus corrupted)."""
    keys = ["architecture", "protocol", "modality", "fold_id", "condition_id", "family", "severity_label"]
    rows = []
    for key, sub in run_metrics.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record["group_id"] = "__".join(key[:4])
        record["n_seeds"] = int(len(sub))
        for metric in METRICS:
            for prefix in ("clean", "corrupted"):
                values = sub[f"{prefix}_{metric}"].to_numpy(dtype=np.float64)
                record[f"{prefix}_{metric}_mean"] = float(values.mean())
            degradation = (sub[f"clean_{metric}"] - sub[f"corrupted_{metric}"]).to_numpy(dtype=np.float64)
            record[f"{metric}_degradation_mean"] = float(degradation.mean())
            record[f"{metric}_degradation_sd"] = float(degradation.std(ddof=ddof)) if degradation.size > ddof else float("nan")
        for extra in ("clean_to_corrupt_flip_rate", "mean_absolute_p_defect_shift"):
            if extra in sub:
                record[f"{extra}_mean"] = float(sub[extra].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def direction_counts(groups: pd.DataFrame, condition_list: list[dict]) -> pd.DataFrame:
    """Number of groups with lower (Dec), equal (Same) or higher (Inc) corrupted performance.

    Dec means the three-seed mean degradation is strictly positive; Same means
    it is exactly zero; Inc means it is strictly negative.
    """
    names = {"macro_f1": "macro_f1", "balanced_accuracy": "balanced_accuracy", "roc_auc": "auroc"}
    rows = []
    for condition in condition_list:
        sub = groups[groups["condition_id"] == condition["condition_id"]]
        record = {"condition_id": condition["condition_id"], "family": condition["family"],
                  "severity": condition["severity_label"], "groups": int(len(sub))}
        for metric, short in names.items():
            values = sub[f"{metric}_degradation_mean"].to_numpy(dtype=np.float64)
            record[f"{short}_dec"] = int(np.sum(values > 0))
            record[f"{short}_same"] = int(np.sum(values == 0))
            record[f"{short}_inc"] = int(np.sum(values < 0))
        rows.append(record)
    return pd.DataFrame(rows)
