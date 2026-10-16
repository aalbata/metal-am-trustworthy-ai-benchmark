"""Classification and calibration metrics with the definitions used for the reported tables.

The positive class is Defect (label 1).  Two NLL definitions are provided.
``clipped_probability_nll`` is the per-run predictive table NLL: the stored
two-class softmax probabilities are clipped to [1e-12, 1] without
renormalisation and the mean negative log probability of the true class is
taken, which caps each sample at -ln(1e-12) = 27.63 nats.
``logit_nll`` is the calibration NLL: a log-sum-exp cross-entropy on (scaled)
logits without clipping.

``step_average_precision`` sorts by decreasing score with a stable mergesort
and averages the precision at the rank of every positive, without grouping
tied scores.  It differs from ``sklearn.metrics.average_precision_score`` when
scores are tied.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def confusion(labels, predictions) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "tn": int(((labels == 0) & (predictions == 0)).sum()),
        "fp": int(((labels == 0) & (predictions == 1)).sum()),
        "fn": int(((labels == 1) & (predictions == 0)).sum()),
        "tp": int(((labels == 1) & (predictions == 1)).sum()),
    }


def rank_auroc(labels, scores) -> float | None:
    """AUROC from average ranks (ties share the mean rank); None if one class is absent."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    positive_rank_sum = float(ranks[labels == 1].sum())
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def step_average_precision(labels, scores) -> float | None:
    """Mean precision at positive ranks after a stable descending sort, no tie grouping."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    positions = np.arange(1, len(sorted_labels) + 1)
    precision = cumulative_tp / positions
    return float(precision[sorted_labels == 1].sum() / n_pos)


def clipped_probability_nll(labels, probabilities, clip=(1e-12, 1.0)) -> float:
    """Mean -log p(true class) on clipped two-class probabilities, no renormalisation."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, clip[0], clip[1])
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))


def logit_nll(logits, labels, temperature: float = 1.0) -> float:
    """Mean cross-entropy of temperature-scaled logits computed with log-sum-exp."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scaled = logits / float(temperature)
    losses = logsumexp(scaled, axis=1) - scaled[np.arange(scaled.shape[0]), labels]
    return float(np.mean(losses, dtype=np.float64))


def softmax_probabilities(logits, temperature: float = 1.0) -> np.ndarray:
    """Two-class softmax of temperature-scaled logits in float64."""
    logits = np.asarray(logits, dtype=np.float64)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Invalid temperature: {temperature}")
    scaled = logits / temperature
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))


def binary_brier(p_defect, labels) -> float:
    """Mean squared difference between P(Defect) and the Defect indicator."""
    p_defect = np.asarray(p_defect, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p_defect - labels) ** 2, dtype=np.float64))


def binary_ece(p_defect, labels, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Binary ECE on P(Defect) against the Defect frequency with equal-width bins.

    Bin index is ``min(floor(10 p), 9)`` so the last bin is [0.9, 1.0].  Each
    non-empty bin contributes ``n_bin / N * |mean p - Defect frequency|``.
    """
    p_defect = np.asarray(p_defect, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        raise ValueError("Empty input.")
    index = np.clip(np.floor(p_defect * n_bins).astype(np.int64), 0, n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        record = {"bin_index": b, "lower": b / n_bins, "upper": (b + 1) / n_bins, "count": count,
                  "mean_probability": None, "defect_frequency": None, "gap": None}
        if count:
            mean_p = float(np.mean(p_defect[mask], dtype=np.float64))
            freq = float(np.mean(labels[mask].astype(np.float64), dtype=np.float64))
            gap = abs(mean_p - freq)
            ece += count / n * gap
            record.update(mean_probability=mean_p, defect_frequency=freq, gap=gap)
        bins.append(record)
    return float(ece), bins


def classification_metrics(labels, probabilities, predictions, nll_clip=(1e-12, 1.0)) -> dict:
    """Per-run test metrics with the definitions used in the predictive tables.

    ``probabilities`` is the [N, 2] softmax matrix (Good, Defect) and
    ``predictions`` the hard labels (P(Defect) >= 0.5).
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    cm = confusion(labels, predictions)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    precision_good = safe_div(tn, tn + fn)
    recall_good = safe_div(tn, tn + fp)
    f1_good = safe_div(2 * precision_good * recall_good, precision_good + recall_good)
    precision_defect = safe_div(tp, tp + fp)
    recall_defect = safe_div(tp, tp + fn)
    f1_defect = safe_div(2 * precision_defect * recall_defect, precision_defect + recall_defect)
    mcc_den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = 0.0 if mcc_den == 0.0 else float((tp * tn - fp * fn) / mcc_den)
    p_defect = probabilities[:, 1]
    return {
        "macro_f1": float((f1_good + f1_defect) / 2.0),
        "balanced_accuracy": float((recall_good + recall_defect) / 2.0),
        "accuracy": safe_div(tn + tp, len(labels)),
        "mcc": mcc,
        "auroc": rank_auroc(labels, p_defect),
        "auprc": step_average_precision(labels, p_defect),
        "nll": clipped_probability_nll(labels, probabilities, nll_clip),
        "brier": binary_brier(p_defect, labels),
        "classwise": {
            "Good": {"precision": precision_good, "recall": recall_good, "f1": f1_good},
            "Defect": {"precision": precision_defect, "recall": recall_defect, "f1": f1_defect},
        },
        "confusion_matrix": cm,
    }


def mean_sd(values, ddof: int = 1) -> tuple[float, float]:
    """Mean and standard deviation (sample SD by default) of a sequence."""
    array = np.asarray(values, dtype=np.float64)
    sd = float(np.std(array, ddof=ddof)) if array.size > ddof else float("nan")
    return float(np.mean(array)), sd
