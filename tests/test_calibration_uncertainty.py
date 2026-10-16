"""Temperature fitting with the boundary rule and the ensemble uncertainty identities."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from amtrust import calibration, uncertainty


class TemperatureTest(unittest.TestCase):
    def test_recovers_known_temperature(self):
        rng = np.random.default_rng(1)
        n = 4000
        true_logit = rng.normal(0, 2, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(int)
        logits = np.column_stack([np.zeros(n), true_logit * 3.0])
        fit = calibration.fit_temperature(logits, y)
        self.assertEqual(fit["fit_status"], "PASS")
        self.assertAlmostEqual(fit["temperature"], 3.0, delta=0.25)
        self.assertLessEqual(fit["validation_nll_after_fit"], fit["validation_nll_at_t1"])

    def test_perfect_separation_hits_lower_bound(self):
        # Completely separated validation set with small margins (minimum 0.2).
        logits = np.array([[0.2, 0.0], [0.5, 0.0], [0.0, 0.25], [0.0, 0.4]])
        y = np.array([0, 0, 1, 1])
        fit = calibration.fit_temperature(logits, y)
        self.assertTrue(fit["boundary_flag"])
        self.assertEqual(fit["fit_status"], "NOT_ESTIMABLE_BOUNDARY")
        self.assertIsNone(fit["temperature"])
        self.assertAlmostEqual(fit["diagnostic_boundary_temperature"], 0.01, places=5)

    def test_uninformative_logits_hit_upper_bound(self):
        logits = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]])
        y = np.array([1, 0, 1, 0])
        fit = calibration.fit_temperature(logits, y)
        self.assertGreater(fit["eta"], math.log(100.0) - 1e-6)
        flagged = calibration.fit_temperature(logits, y, boundary_tolerance=1e-6)
        self.assertTrue(flagged["boundary_flag"])
        self.assertIsNone(flagged["temperature"])

    def test_eta_objective_at_zero_is_t1_nll(self):
        logits = np.array([[0.2, 1.0], [2.0, -0.5]])
        y = np.array([1, 0])
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        expected = -np.mean(np.log(p[[0, 1], y]))
        self.assertAlmostEqual(calibration.eta_objective(0.0, logits, y), expected, places=12)

    def test_group_summary_excludes_non_estimable_runs(self):
        base = {"architecture": "resnet18", "protocol": "S", "modality": "OT", "fold_id": "S3_HOLDOUT_X"}
        runs = pd.DataFrame([
            {**base, "seed": 1, "fit_status": "PASS", "test_nll_pre": 1.0, "test_nll_post": 0.5,
             "test_brier_pre": 0.2, "test_brier_post": 0.1, "test_ece_pre": 0.1, "test_ece_post": 0.2},
            {**base, "seed": 2, "fit_status": "PASS", "test_nll_pre": 2.0, "test_nll_post": 1.0,
             "test_brier_pre": 0.3, "test_brier_post": 0.2, "test_ece_pre": 0.1, "test_ece_post": 0.2},
            {**base, "seed": 3, "fit_status": "NOT_ESTIMABLE_BOUNDARY", "test_nll_pre": 9.0,
             "test_nll_post": np.nan, "test_brier_pre": 0.5, "test_brier_post": np.nan,
             "test_ece_pre": 0.4, "test_ece_post": np.nan},
        ])
        g = calibration.group_summary(runs).iloc[0]
        self.assertEqual(g["uncalibrated_n"], 3)
        self.assertEqual(g["calibrated_n"], 2)
        self.assertAlmostEqual(g["nll_pre_mean"], 4.0)
        self.assertAlmostEqual(g["nll_delta_mean"], -0.75)
        self.assertEqual(g["nll_direction"], "IMPROVED_LOWER_ERROR")
        self.assertEqual(g["ece_direction"], "WORSENED_HIGHER_ERROR")
        self.assertEqual(g["joint_pattern"], "MIXED_OR_PARTIAL")


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        pd_ = rng.random((3, 50))
        self.tensor = np.stack([1 - pd_, pd_], axis=2)
        self.labels = rng.integers(0, 2, 50)
        self.stats = uncertainty.ensemble_statistics(self.tensor, self.labels)

    def test_entropy_gap_identity(self):
        h_mean = uncertainty.entropy(self.tensor.mean(axis=0))
        mean_h = uncertainty.entropy(self.tensor).mean(axis=0)
        np.testing.assert_allclose(self.stats["entropy_gap"], np.maximum(h_mean - mean_h, 0.0), atol=1e-15)

    def test_entropy_gap_is_generalised_js_divergence(self):
        mean = self.tensor.mean(axis=0)
        kl = np.sum(self.tensor * np.log(self.tensor / mean[None]), axis=2).mean(axis=0)
        np.testing.assert_allclose(self.stats["entropy_gap"], kl, atol=1e-12)

    def test_identical_members_give_zero_gap_and_variance(self):
        same = np.repeat(self.tensor[:1], 3, axis=0)
        s = uncertainty.ensemble_statistics(same, self.labels)
        np.testing.assert_allclose(s["entropy_gap"], 0.0, atol=1e-15)
        np.testing.assert_allclose(s["pdefect_variance"], 0.0, atol=1e-15)
        np.testing.assert_allclose(s["variation_ratio"], 0.0)

    def test_entropy_bounds_and_xlogy(self):
        self.assertTrue(np.all(self.stats["predictive_entropy"] <= math.log(2) + 1e-12))
        self.assertEqual(float(uncertainty.entropy(np.array([[1.0, 0.0]]))[0]), 0.0)

    def test_variance_ddof0_and_variation_ratio(self):
        np.testing.assert_allclose(self.stats["pdefect_variance"], self.tensor[:, :, 1].var(axis=0, ddof=0))
        self.assertTrue(set(np.round(self.stats["variation_ratio"], 12)) <= {0.0, round(1 / 3, 12)})

    def test_vote_tie_goes_to_good(self):
        t = np.array([[[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]])
        s = uncertainty.ensemble_statistics(t, np.array([1]))
        self.assertEqual(int(s["ensemble_prediction"][0]), 0)
        self.assertEqual(int(s["error"][0]), 1)

    def test_group_record_auroc(self):
        rec = uncertainty.group_record(self.stats)
        self.assertEqual(rec["n"], 50)
        self.assertTrue(0.0 <= rec["error_auroc_entropy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
