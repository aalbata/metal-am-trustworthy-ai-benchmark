"""Paired, build-stratified block resampling tests for architecture contrasts.

Test observations of one task are ordered by build token and, within a build,
by sequence index.  For block length L:

* Bootstrap (B = 5000): within every build series of length n, ceil(n / L)
  block starts are drawn uniformly from ``0 .. n - L`` and the concatenated
  blocks are truncated to n.  The same resampled indices are used for every
  architecture, seed and contrast of the task.  The statistic is the mean over
  the three seeds of the metric difference (model A minus model B); the 95 %
  interval is the 2.5 and 97.5 percentiles.
* Randomisation (10000 iterations): each build series is cut into consecutive
  non-overlapping blocks of length L (the last block may be shorter).  In each
  iteration every block independently swaps the predictions of models A and B
  with probability 1/2, using one swap matrix per task shared by all seeds and
  contrasts.  The two-sided p-value is (E + 1) / (B + 1) with
  E = #{|randomised statistic| >= |observed statistic|}.
* Holm adjustment across the three contrasts within each task, metric and
  block length.

The random generator of each plan is ``numpy.random.default_rng`` seeded with
the first 16 hexadecimal digits of
SHA-256("STATISTICAL_AMENDMENT_V1_1|BASE=20260916|PHASE=<BOOTSTRAP|RANDOMIZATION>|TASK=<protocol>|<modality>|<fold id>|L=<L>")
modulo 2**32.  A block length is used for a task only when every build series
of its test set has at least 2L observations.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


def task_string(task: tuple[str, str, str]) -> str:
    return f"{task[0]}|{task[1]}|{task[2]}"


def task_name(task: tuple[str, str, str]) -> str:
    """Short task name such as R_OT or S_PB_S2."""
    return f"R_{task[1]}" if task[0] == "R" else f"S_{task[1]}_{task[2].split('_')[0]}"


def derive_rng_seed(phase: str, task: tuple[str, str, str], block_length: int, base_seed: int = 20260916) -> int:
    identity = (f"STATISTICAL_AMENDMENT_V1_1|BASE={base_seed}|PHASE={phase}|"
                f"TASK={task_string(task)}|L={block_length}")
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 32)


def safe_div_array(numerator, denominator) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.zeros(np.broadcast(numerator, denominator).shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=(denominator != 0.0))
    return out


def metrics_from_confusion(tn, fp, fn, tp) -> dict:
    """Vectorised Macro-F1 and balanced accuracy from confusion counts."""
    tn, fp, fn, tp = (np.asarray(v, dtype=np.float64) for v in (tn, fp, fn, tp))
    precision_good = safe_div_array(tn, tn + fn)
    recall_good = safe_div_array(tn, tn + fp)
    f1_good = safe_div_array(2.0 * precision_good * recall_good, precision_good + recall_good)
    precision_defect = safe_div_array(tp, tp + fp)
    recall_defect = safe_div_array(tp, tp + fn)
    f1_defect = safe_div_array(2.0 * precision_defect * recall_defect, precision_defect + recall_defect)
    return {"macro_f1": (f1_good + f1_defect) / 2.0, "balanced_accuracy": (recall_good + recall_defect) / 2.0}


def confusion_vector(labels, predictions) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return np.asarray([
        np.sum((labels == 0) & (predictions == 0)),
        np.sum((labels == 0) & (predictions == 1)),
        np.sum((labels == 1) & (predictions == 0)),
        np.sum((labels == 1) & (predictions == 1)),
    ], dtype=np.int64)


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p_values = np.asarray(p_values, dtype=np.float64)
    m = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    adjusted_sorted = np.zeros(m, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.zeros(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def eligible(series_lengths, block_length: int) -> bool:
    """A block length is usable when every build series has n >= 2L."""
    return all(n >= 2 * block_length for n in series_lengths)


def bootstrap_plan(series_lengths: list[int], block_length: int, replicates: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap indices [replicates, N] over concatenated build series."""
    total = int(sum(series_lengths))
    offsets = np.concatenate([[0], np.cumsum(series_lengths)[:-1]]).astype(int)
    matrix = np.empty((replicates, total), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for r in range(replicates):
        cursor = 0
        for n, offset in zip(series_lengths, offsets):
            starts = rng.integers(low=0, high=n - block_length + 1, size=int(math.ceil(n / block_length)))
            positions = []
            for start in starts.tolist():
                positions.extend(range(int(start), int(start) + block_length))
            matrix[r, cursor:cursor + n] = np.asarray(positions[:n], dtype=np.int32) + int(offset)
            cursor += n
    return matrix


def randomization_blocks(series_lengths: list[int], block_length: int) -> list[np.ndarray]:
    """Consecutive non-overlapping blocks of positions within each build series."""
    blocks = []
    offset = 0
    for n in series_lengths:
        for start in range(0, n, block_length):
            end = min(start + block_length, n)
            blocks.append(np.arange(offset + start, offset + end, dtype=np.int32))
        offset += n
    return blocks


def swap_matrix(n_blocks: int, iterations: int, seed: int) -> np.ndarray:
    """Block swap indicators drawn with the default integer dtype and cast to int8."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=2, size=(iterations, n_blocks)).astype(np.int8)


def bootstrap_metrics(labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray) -> dict:
    """Macro-F1 and balanced accuracy for each bootstrap replicate."""
    y = labels[indices]
    p = predictions[indices]
    tn = np.sum((y == 0) & (p == 0), axis=1)
    fp = np.sum((y == 0) & (p == 1), axis=1)
    fn = np.sum((y == 1) & (p == 0), axis=1)
    tp = np.sum((y == 1) & (p == 1), axis=1)
    return metrics_from_confusion(tn, fp, fn, tp)


def randomized_differences(labels, pred_a, pred_b, blocks, swaps) -> dict:
    """Metric differences (A minus B) for every randomisation iteration."""
    base_a = confusion_vector(labels, pred_a)
    base_b = confusion_vector(labels, pred_b)
    block_a = np.stack([confusion_vector(labels[b], pred_a[b]) for b in blocks])
    block_b = np.stack([confusion_vector(labels[b], pred_b[b]) for b in blocks])
    swaps = swaps.astype(np.int64, copy=False)
    randomized_a = base_a[None, :] + swaps @ (block_b - block_a)
    randomized_b = base_b[None, :] + swaps @ (block_a - block_b)
    metrics_a = metrics_from_confusion(*(randomized_a[:, k] for k in range(4)))
    metrics_b = metrics_from_confusion(*(randomized_b[:, k] for k in range(4)))
    return {name: metrics_a[name] - metrics_b[name] for name in metrics_a}


def contrast_tests(task, labels: np.ndarray, predictions: dict, series_lengths: list[int], block_length: int,
                   contrasts: list[dict], seeds: list[int], metrics=("macro_f1", "balanced_accuracy"),
                   replicates: int = 5000, iterations: int = 10000, base_seed: int = 20260916,
                   confidence: float = 0.95) -> list[dict]:
    """Bootstrap intervals, randomisation p-values and Holm adjustment for one task and L.

    ``predictions[architecture][seed]`` holds hard predictions in the ordered
    test sequence and ``labels`` the matching labels.
    """
    labels = np.asarray(labels, dtype=np.int64)
    indices = bootstrap_plan(series_lengths, block_length, replicates,
                             derive_rng_seed("BOOTSTRAP", task, block_length, base_seed))
    blocks = randomization_blocks(series_lengths, block_length)
    swaps = swap_matrix(len(blocks), iterations, derive_rng_seed("RANDOMIZATION", task, block_length, base_seed))
    observed, booted = {}, {}
    for arch, by_seed in predictions.items():
        observed[arch], booted[arch] = {}, {}
        for seed in seeds:
            pred = np.asarray(by_seed[seed], dtype=np.int64)
            cm = confusion_vector(labels, pred)
            observed[arch][seed] = {k: float(v) for k, v in metrics_from_confusion(*cm).items()}
            booted[arch][seed] = bootstrap_metrics(labels, pred, indices)
    lower_q = 100.0 * (1.0 - confidence) / 2.0
    rows = []
    for metric in metrics:
        for contrast in contrasts:
            a, b = contrast["model_a"], contrast["model_b"]
            obs = float(np.mean([observed[a][s][metric] - observed[b][s][metric] for s in seeds]))
            boot = np.mean(np.stack([booted[a][s][metric] - booted[b][s][metric] for s in seeds]), axis=0)
            rows.append({
                "block_length": block_length, "task": task_name(task), "task_key": task_string(task),
                "protocol": task[0], "modality": task[1], "fold_id": task[2], "metric": metric,
                "contrast": contrast["name"], "model_a": a, "model_b": b, "estimate": obs,
                "ci95_low": float(np.percentile(boot, lower_q)), "ci95_high": float(np.percentile(boot, 100.0 - lower_q)),
                "bootstrap_replicates": replicates, "randomization_iterations": iterations,
            })
    lookup = {(r["metric"], r["contrast"]): r for r in rows}
    for contrast in contrasts:
        a, b = contrast["model_a"], contrast["model_b"]
        per_seed = {m: [] for m in metrics}
        for seed in seeds:
            diffs = randomized_differences(labels, np.asarray(predictions[a][seed]), np.asarray(predictions[b][seed]),
                                           blocks, swaps)
            for m in metrics:
                per_seed[m].append(diffs[m])
        for m in metrics:
            statistic = np.mean(np.stack(per_seed[m]), axis=0)
            row = lookup[(m, contrast["name"])]
            extreme = int(np.sum(np.abs(statistic) >= abs(row["estimate"])))
            row["extreme_count"] = extreme
            row["raw_p"] = (extreme + 1) / (iterations + 1)
    for m in metrics:
        family = [lookup[(m, c["name"])] for c in contrasts]
        for row, adjusted in zip(family, holm_adjust([r["raw_p"] for r in family])):
            row["holm_p"] = float(adjusted)
    return rows
