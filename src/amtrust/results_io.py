"""Reading the results CSV files and shared labelling helpers for tables and figures."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import load_config

RESULT_FILES = (
    "split_counts.csv", "predictive_groups.csv", "calibration_groups.csv", "uq_groups.csv", "uq_summary.csv",
    "robustness_direction.csv", "xai_threshold_free.csv", "xai_overlap.csv", "efficiency.csv",
    "stats_tests.csv", "baselines.csv",
)

ARCH_ORDER = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large"]
ARCH_LABELS = {"resnet18": "ResNet18", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0",
               "mobilenet_v3_large": "MobileNetV3-Large"}
MODALITY_LABELS = {"OT": "OT", "PB": "PB"}
METRIC_LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced accuracy", "auroc": "AUROC",
                 "roc_auc": "AUROC", "auprc": "AUPRC", "nll": "NLL", "brier": "Brier"}
FAMILY_LABELS = {"gaussian_intensity_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur",
                 "brightness_attenuation": "Brightness", "contrast_reduction": "Contrast",
                 "jpeg_compression": "JPEG"}
FAMILY_ORDER = list(FAMILY_LABELS)

COLUMN_ALIASES = {
    "fold": "fold_id",
    "brier_delta_mean_derived": "brier_delta_mean",
    "ece_delta_mean_derived": "ece_delta_mean",
}


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename alias columns and add protocol, modality and fold id from a task key when absent."""
    renames = {c: COLUMN_ALIASES[c] for c in frame.columns if c in COLUMN_ALIASES and COLUMN_ALIASES[c] not in frame.columns}
    frame = frame.rename(columns=renames)
    if "fold_id" not in frame.columns and "task_key" in frame.columns:
        parts = frame["task_key"].astype(str).str.split("|", expand=True)
        frame["protocol"], frame["modality"], frame["fold_id"] = parts[0], parts[1], parts[2]
    return frame


def load(name: str, results_dir: Path) -> pd.DataFrame | None:
    """Read ``results_dir/name``; return None when the file is absent."""
    path = Path(results_dir) / name
    if not path.is_file():
        return None
    frame = pd.read_csv(path, keep_default_na=True, dtype={"fold_id": str, "fold": str})
    return normalise_columns(frame)


def find_column(frame: pd.DataFrame, candidates=(), tokens=()) -> str:
    """Return the first existing candidate column, else the first column containing all tokens."""
    for name in candidates:
        if name in frame.columns:
            return name
    lowered = {c: c.lower() for c in frame.columns}
    for column, low in lowered.items():
        if tokens and all(t in low for t in tokens):
            return column
    raise KeyError(f"No column among {list(candidates)} or with tokens {list(tokens)} in {list(frame.columns)}")


def build_label_of(fold_id: str, config: dict | None = None) -> str:
    """Held-out build label (B1, B2, B3) of a Protocol S fold id; empty for Protocol R."""
    config = config or load_config()
    fold_id = str(fold_id)
    for label, token in config["dataset"]["builds"].items():
        if token in fold_id:
            return label
    match = re.match(r"S(\d)", fold_id)
    return f"B{match.group(1)}" if match else ""


def task_label(protocol: str, modality: str, fold_id: str, config: dict | None = None) -> str:
    """Short label such as ``R OT`` or ``S PB B2``."""
    if str(protocol) == "R":
        return f"R {modality}"
    return f"S {modality} {build_label_of(fold_id, config)}"


TASK_ORDER = ["R OT", "R PB", "S OT B1", "S OT B2", "S OT B3", "S PB B1", "S PB B2", "S PB B3"]


def add_labels(frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add ``task_label``, ``arch_label`` and sort keys to a per-group frame."""
    frame = frame.copy()
    if "fold_id" in frame.columns:
        frame["task_label"] = [task_label(p, m, f, config) for p, m, f in
                               zip(frame["protocol"], frame["modality"], frame["fold_id"])]
        frame["_task_rank"] = frame["task_label"].map({t: i for i, t in enumerate(TASK_ORDER)})
    if "architecture" in frame.columns:
        frame["arch_label"] = frame["architecture"].map(ARCH_LABELS)
        frame["_arch_rank"] = frame["architecture"].map({a: i for i, a in enumerate(ARCH_ORDER)})
    return frame


def precision_label(value: str) -> str:
    text = str(value).upper()
    if "AMP" in text or "FP16" in text:
        return "AMP/FP16"
    if "FP32" in text:
        return "FP32"
    return str(value)


def architecture_key(value: str) -> str:
    """Map display or code names of an architecture to the code name."""
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    for code in ARCH_ORDER:
        if re.sub(r"[^a-z0-9]", "", code) == text or re.sub(r"[^a-z0-9]", "", ARCH_LABELS[code].lower()) == text:
            return code
    if "mobilenet" in text or "mnv3" in text or "mv3" in text:
        return "mobilenet_v3_large"
    if "efficient" in text or "enb0" in text:
        return "efficientnet_b0"
    if "50" in text:
        return "resnet50"
    if "18" in text:
        return "resnet18"
    return str(value)
