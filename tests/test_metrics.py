"""Metric definitions on small synthetic inputs."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from amtrust import metrics


class StepAveragePrecisionTest(unittest.TestCase):
    def test_matches_sklearn_without_ties(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = rng.random(200)
        self.assertAlmostEqual(metrics.step_average_precision(y, s), average_precision_score(y, s), places=12)

    def test_differs_from_sklearn_with_ties(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        s = np.array([0.9, 0.9, 0.9, 0.2, 0.2, 0.1])
        # Stable order on -s keeps the input order inside each tie: 0,1,2 then 3,4 then 5.
        # Positives at ranks 2, 3 and 5: (1/2 + 2/3 + 3/5) / 3.
        expected = (1 / 2 + 2 / 3 + 3 / 5) / 3
        self.assertAlmostEqual(metrics.step_average_precision(y, s), expected, places=12)
        self.assertNotAlmostEqual(metrics.step_average_precision(y, s), average_precision_score(y, s), places=6)

    def test_order_dependence_inside_ties(self):
        y1 = np.array([1, 0])
        y2 = np.array([0, 1])
        s = np.array([0.5, 0.5])
        self.assertEqual(metrics.step_average_precision(y1, s), 1.0)
        self.assertEqual(metrics.step_average_precision(y2, s), 0.5)

    def test_no_positive(self):
        self.assertIsNone(metrics.step_average_precision([0, 0], [0.1, 0.2]))


class AurocTest(unittest.TestCase):
    def test_rank_auroc_matches_sklearn_with_ties(self):
        y = np.array([0, 1, 1, 0, 1, 0, 1])
        s = np.array([0.1, 0.4, 0.4, 0.4, 0.8, 0.2, 0.8])
        self.assertAlmostEqual(metrics.rank_auroc(y, s), roc_auc_score(y, s), places=12)

    def test_single_class(self):
        self.assertIsNone(metrics.rank_auroc([1, 1], [0.2, 0.3]))


class NllTest(unittest.TestCase):
    def test_clipped_nll_caps_each_sample(self):
        p = np.array([[1.0, 0.0], [0.5, 0.5]])
        y = np.array([1, 0])
        expected = (-math.log(1e-12) - math.log(0.5)) / 2
        self.assertAlmostEqual(metrics.clipped_probability_nll(y, p), expected, places=10)

    def test_clipped_nll_does_not_renormalise(self):
        p = np.array([[0.6, 0.5]])
        self.assertAlmostEqual(metrics.clipped_probability_nll([0], p), -math.log(0.6), places=12)

    def test_logit_nll_equals_softmax_cross_entropy(self):
        logits = np.array([[2.0, -1.0], [0.3, 0.7], [-5.0, 5.0]])
        y = np.array([0, 1, 0])
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        expected = -np.mean(np.log(p[np.arange(3), y]))
        self.assertAlmostEqual(metrics.logit_nll(logits, y), expected, places=12)

    def test_logit_nll_is_unclipped(self):
        logits = np.array([[100.0, -100.0]])
        self.assertAlmostEqual(metrics.logit_nll(logits, [1]), 200.0, places=8)
        probs = metrics.softmax_probabilities(logits)
        self.assertAlmostEqual(metrics.clipped_probability_nll([1], probs), -math.log(1e-12), places=8)


class CalibrationMetricTest(unittest.TestCase):
    def test_brier(self):
        self.assertAlmostEqual(metrics.binary_brier([0.2, 0.9], [0, 1]), (0.04 + 0.01) / 2, places=12)

    def test_ece_bins_and_value(self):
        p = np.array([0.05, 0.15, 0.95, 1.0, 0.95])
        y = np.array([0, 1, 1, 1, 0])
        ece, bins = metrics.binary_ece(p, y, 10)
        self.assertEqual([b["count"] for b in bins], [1, 1, 0, 0, 0, 0, 0, 0, 0, 3])
        expected = 1 / 5 * 0.05 + 1 / 5 * 0.85 + 3 / 5 * abs((0.95 + 1.0 + 0.95) / 3 - 2 / 3)
        self.assertAlmostEqual(ece, expected, places=12)


class ClassificationMetricsTest(unittest.TestCase):
    def test_macro_f1_and_balanced_accuracy(self):
        y = np.array([0, 0, 1, 1, 1])
        pred = np.array([0, 1, 1, 1, 0])
        probs = np.column_stack([1 - pred * 0.8, pred * 0.8])
        m = metrics.classification_metrics(y, probs, pred)
        f1_good = 2 * (1 / 2) * (1 / 2) / 1.0
        f1_defect = 2 * (2 / 3) * (2 / 3) / (4 / 3)
        self.assertAlmostEqual(m["macro_f1"], (f1_good + f1_defect) / 2, places=12)
        self.assertAlmostEqual(m["balanced_accuracy"], (1 / 2 + 2 / 3) / 2, places=12)
        self.assertEqual(m["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 2})


if __name__ == "__main__":
    unittest.main()
