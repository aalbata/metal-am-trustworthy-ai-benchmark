"""Three-seed ensemble uncertainty on uncalibrated test probabilities.

For one group (architecture x protocol x modality x fold) the members are the
three training seeds.  With member probabilities ``p_m`` (Good, Defect):

* ensemble ``p_bar = mean_m p_m`` (equal weights, no renormalisation);
* predictive entropy ``H[p_bar] = -sum_c p_bar ln p_bar`` in nats
  (``scipy.special.xlogy``, so 0 ln 0 = 0);
* expected member entropy ``mean_m H[p_m]``;
* entropy gap ``H[p_bar] - mean_m H[p_m]`` (generalised Jensen-Shannon
  divergence of the members; values in [-1e-12, 0) are reported as 0);
* P(Defect) variance across members with ddof = 0;
* variation ratio ``1 - (modal vote count) / 3`` with member votes
  ``P(Defect) > 0.5``;
* error indicator from the ensemble prediction ``P_bar(Defect) > 0.5``;
* error-detection AUROC of entropy and of entropy gap
  (``sklearn.metrics.roc_auc_score``), undefined if a group has no errors or
  no correct predictions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import roc_auc_score

LN2 = math.log(2.0)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats along the last axis."""
    return -np.sum(xlogy(probabilities, probabilities), axis=-1, dtype=np.float64)


def ensemble_statistics(member_probabilities: np.ndarray, labels: np.ndarray, sum_tolerance: float = 1e-6,
                        bound_tolerance: float = 1e-12, gap_tolerance: float = 1e-12) -> dict:
    """Per-sample ensemble quantities for a [M, N, 2] probability array."""
    tensor = np.asarray(member_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if tensor.ndim != 3 or tensor.shape[2] != 2:
        raise ValueError("Expected an [M, N, 2] array.")
    if np.max(np.abs(tensor.sum(axis=2) - 1.0)) > sum_tolerance:
        raise ValueError("Member probabilities do not sum to one within tolerance.")
    members = tensor.shape[0]
    ensemble = np.mean(tensor, axis=0, dtype=np.float64)
    s = np.sum(ensemble, axis=1, dtype=np.float64)
    h = entropy(ensemble)
    h_bound = -xlogy(s, s / 2.0)
    if np.any(h - h_bound > bound_tolerance):
        raise ValueError("Predictive entropy exceeds the sum-aware bound.")
    expected_h = np.mean(entropy(tensor), axis=0, dtype=np.float64)
    gap = h - expected_h
    if np.any(gap < -gap_tolerance):
        raise ValueError("Entropy gap is negative beyond tolerance.")
    small_negative = (gap < 0.0) & (gap >= -gap_tolerance)
    gap = np.where(small_negative, 0.0, gap)
    variance = np.var(tensor[:, :, 1], axis=0, ddof=0, dtype=np.float64)
    votes = (tensor[:, :, 1] > 0.5).astype(np.int64)
    defect_votes = votes.sum(axis=0)
    modal = np.maximum(defect_votes, members - defect_votes)
    variation_ratio = 1.0 - modal.astype(np.float64) / members
    prediction = (ensemble[:, 1] > 0.5).astype(np.int64)
    error = (prediction != labels).astype(np.int64)
    return {
        "ensemble": ensemble,
        "predictive_entropy": h,
        "expected_member_entropy": expected_h,
        "entropy_gap": gap,
        "pdefect_variance": variance,
        "variation_ratio": variation_ratio,
        "ensemble_prediction": prediction,
        "error": error,
        "entropy_above_ln2": int(np.sum(h > LN2 + bound_tolerance)),
        "gap_clamped": int(np.sum(small_negative)),
    }


def group_record(stats: dict) -> dict:
    """Group-level means and error-detection AUROCs."""
    error = stats["error"]
    n = int(error.size)
    errors = int(error.sum())
    record = {
        "n": n,
        "errors": errors,
        "mean_predictive_entropy_nats": float(np.mean(stats["predictive_entropy"])),
        "mean_expected_member_entropy_nats": float(np.mean(stats["expected_member_entropy"])),
        "mean_entropy_gap_nats": float(np.mean(stats["entropy_gap"])),
        "mean_pdefect_variance": float(np.mean(stats["pdefect_variance"])),
        "mean_variation_ratio": float(np.mean(stats["variation_ratio"])),
        "entropy_drift_flags": stats["entropy_above_ln2"],
    }
    if 0 < errors < n:
        record["error_auroc_entropy"] = float(roc_auc_score(error, stats["predictive_entropy"]))
        record["error_auroc_entropy_gap"] = float(roc_auc_score(error, stats["entropy_gap"]))
    else:
        record["error_auroc_entropy"] = float("nan")
        record["error_auroc_entropy_gap"] = float("nan")
    return record


SUMMARY_MEASURES = (
    "mean_predictive_entropy_nats",
    "mean_entropy_gap_nats",
    "mean_pdefect_variance",
    "mean_variation_ratio",
    "error_auroc_entropy",
    "error_auroc_entropy_gap",
)


def summarise_groups(groups: pd.DataFrame) -> pd.DataFrame:
    """Distribution of each group-level quantity across groups (linear quantiles)."""
    rows = []
    for measure in SUMMARY_MEASURES:
        values = groups[measure].astype(float).dropna().to_numpy()
        rows.append({
            "measure": measure,
            "n_groups": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25, method="linear")),
            "q75": float(np.quantile(values, 0.75, method="linear")),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count_gt_0_5": int(np.sum(values > 0.5)),
            "count_lt_0_5": int(np.sum(values < 0.5)),
        })
    return pd.DataFrame(rows)
