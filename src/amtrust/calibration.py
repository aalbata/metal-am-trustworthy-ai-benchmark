"""Scalar temperature scaling fitted on validation logits only.

The temperature is parameterised as ``T = exp(eta)`` and ``eta`` minimises the
mean validation NLL (log-sum-exp on logits) with
``scipy.optimize.minimize_scalar(method="bounded")`` over
``[ln 0.01, ln 100]`` with ``xatol = 1e-8`` and ``maxiter = 500``.  When the
solution lies within 1e-7 of either bound the temperature is reported as not
estimable: the run keeps its uncalibrated metrics, contributes no calibrated
values, and no value is imputed and the bounds are not widened.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .metrics import binary_brier, binary_ece, logit_nll, softmax_probabilities


def eta_objective(eta: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean validation NLL at temperature exp(eta); +inf for invalid values."""
    eta = float(eta)
    if not math.isfinite(eta):
        return float("inf")
    temperature = math.exp(eta)
    if not math.isfinite(temperature) or temperature <= 0.0:
        return float("inf")
    scaled = logits / temperature
    losses = logsumexp(scaled, axis=1) - scaled[np.arange(scaled.shape[0]), labels]
    value = float(np.mean(losses, dtype=np.float64))
    return value if math.isfinite(value) else float("inf")


def fit_temperature(logits, labels, t_lower: float = 0.01, t_upper: float = 100.0,
                    xatol: float = 1e-8, maxiter: int = 500, boundary_tolerance: float = 1e-7) -> dict:
    """Fit T on validation logits and apply the boundary rule."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite validation logits.")
    eta_lower, eta_upper = math.log(t_lower), math.log(t_upper)
    nll_at_one = eta_objective(0.0, logits, labels)
    result = minimize_scalar(lambda eta: eta_objective(eta, logits, labels), bounds=(eta_lower, eta_upper),
                             method="bounded", options={"xatol": xatol, "maxiter": maxiter})
    eta = float(result.x)
    temperature = math.exp(eta)
    at_lower = eta <= eta_lower + boundary_tolerance
    at_upper = eta >= eta_upper - boundary_tolerance
    boundary = bool(at_lower or at_upper)
    finite = all(math.isfinite(v) for v in (eta, temperature, float(result.fun)))
    estimable = bool(result.success and finite and not boundary and t_lower <= temperature <= t_upper)
    return {
        "eta": eta,
        "temperature": temperature if estimable else None,
        "diagnostic_boundary_temperature": temperature if boundary else None,
        "validation_nll_at_t1": nll_at_one,
        "validation_nll_after_fit": float(result.fun) if estimable else None,
        "solver_success": bool(result.success),
        "solver_nfev": int(getattr(result, "nfev", -1)),
        "solver_nit": int(getattr(result, "nit", -1)),
        "boundary_flag": boundary,
        "fit_status": "PASS" if estimable else ("NOT_ESTIMABLE_BOUNDARY" if boundary else "FAIL"),
    }


def calibration_metrics(logits, labels, temperature: float = 1.0, n_bins: int = 10) -> dict:
    """Logit NLL, binary Brier and binary ECE at a given temperature."""
    probabilities = softmax_probabilities(logits, temperature)
    ece, bins = binary_ece(probabilities[:, 1], labels, n_bins)
    return {
        "nll": logit_nll(logits, labels, temperature),
        "brier": binary_brier(probabilities[:, 1], labels),
        "ece": ece,
        "bins": bins,
    }


def direction(delta: float) -> str:
    """Direction of a calibrated-minus-uncalibrated change (lower is better)."""
    if delta < 0:
        return "IMPROVED_LOWER_ERROR"
    if delta > 0:
        return "WORSENED_HIGHER_ERROR"
    return "UNCHANGED"


def group_summary(runs: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Aggregate per-run calibration results into the 32 groups.

    Uncalibrated means use every run of the group.  Calibrated means and the
    paired deltas use only runs with an estimable temperature.  The direction
    of each metric is the sign of the mean paired delta (calibrated minus
    uncalibrated); an exact zero is "unchanged".
    """
    rows = []
    keys = ["architecture", "protocol", "modality", "fold_id"]
    for key, sub in runs.groupby(keys, sort=True):
        cal = sub[sub["fit_status"] == "PASS"]
        record = dict(zip(keys, key))
        record["group_id"] = "__".join(key)
        record["uncalibrated_n"] = int(len(sub))
        record["calibrated_n"] = int(len(cal))
        for metric in ("nll", "brier", "ece"):
            pre = sub[f"test_{metric}_pre"].to_numpy(dtype=np.float64)
            post = cal[f"test_{metric}_post"].to_numpy(dtype=np.float64)
            delta = post - cal[f"test_{metric}_pre"].to_numpy(dtype=np.float64)
            record[f"{metric}_pre_mean"] = float(np.mean(pre))
            record[f"{metric}_pre_sd"] = float(np.std(pre, ddof=ddof)) if pre.size > ddof else float("nan")
            record[f"{metric}_post_mean"] = float(np.mean(post)) if post.size else float("nan")
            record[f"{metric}_post_sd"] = float(np.std(post, ddof=ddof)) if post.size > ddof else float("nan")
            record[f"{metric}_delta_mean"] = float(np.mean(delta)) if delta.size else float("nan")
            record[f"{metric}_delta_sd"] = float(np.std(delta, ddof=ddof)) if delta.size > ddof else float("nan")
            record[f"{metric}_direction"] = direction(record[f"{metric}_delta_mean"]) if delta.size else "NOT_ESTIMABLE"
        dirs = {record[f"{m}_direction"] for m in ("nll", "brier", "ece")}
        if dirs == {"NOT_ESTIMABLE"}:
            record["joint_pattern"] = "NOT_ESTIMABLE"
        elif dirs == {"IMPROVED_LOWER_ERROR"}:
            record["joint_pattern"] = "ALL_THREE_IMPROVED"
        elif dirs == {"WORSENED_HIGHER_ERROR"}:
            record["joint_pattern"] = "ALL_THREE_WORSENED"
        elif dirs == {"UNCHANGED"}:
            record["joint_pattern"] = "ALL_THREE_UNCHANGED"
        else:
            record["joint_pattern"] = "MIXED_OR_PARTIAL"
        rows.append(record)
    return pd.DataFrame(rows)
