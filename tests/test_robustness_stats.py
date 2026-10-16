"""Corruption determinism and the block resampling helpers."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from amtrust import robustness, stats

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "benchmark.json").read_text(encoding="utf-8"))


class CorruptionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.random((2, 32, 32, 3)).astype(np.float32)
        self.ids = ["OT__SI383820211201123521__000001", "PB__SI383820240226095904__000010"]
        self.conditions = robustness.conditions(CONFIG)

    def test_fifteen_conditions_in_order(self):
        self.assertEqual(len(self.conditions), 15)
        self.assertEqual(self.conditions[0]["condition_id"], "gaussian_intensity_noise__S1")
        self.assertEqual(self.conditions[-1]["condition_id"], "jpeg_compression__S3")
        self.assertEqual([c["value"] for c in self.conditions[:3]], [0.02, 0.05, 0.10])

    def test_noise_seed_rule(self):
        material = "ROBUSTNESS_V1|gaussian_intensity_noise|S2|" + self.ids[0]
        expected = int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)
        self.assertEqual(robustness.noise_seed("S2", self.ids[0]), expected)

    def test_noise_is_deterministic_per_sample_and_shared_across_channels(self):
        c = self.conditions[1]
        a = robustness.corrupt_batch(self.x, c, self.ids)
        b = robustness.corrupt_batch(self.x[::-1].copy(), c, self.ids[::-1])
        np.testing.assert_array_equal(a[0], b[1])
        seed = robustness.noise_seed("S2", self.ids[0])
        field = np.random.Generator(np.random.PCG64(seed)).standard_normal((32, 32))
        expected = np.clip(self.x[0].astype(np.float64) + 0.05 * field[..., None], 0, 1).astype(np.float32)
        np.testing.assert_array_equal(a[0], expected)

    def test_all_conditions_are_deterministic_and_bounded(self):
        for c in self.conditions:
            a = robustness.corrupt_batch(self.x, c, self.ids)
            b = robustness.corrupt_batch(self.x, c, self.ids)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.dtype, np.float32)
            self.assertEqual(a.shape, self.x.shape)
            self.assertTrue(a.min() >= 0.0 and a.max() <= 1.0)

    def test_brightness_and_contrast_formulas(self):
        bright = robustness.corrupt_batch(self.x, {"family": "brightness_attenuation", "value": 0.5,
                                                   "severity_label": "S3"}, self.ids)
        np.testing.assert_allclose(bright, np.clip(self.x.astype(np.float64) * 0.5, 0, 1), atol=1e-7)
        contrast = robustness.corrupt_batch(self.x, {"family": "contrast_reduction", "value": 0.45,
                                                     "severity_label": "S3"}, self.ids)
        mean = self.x.astype(np.float64).mean(axis=(1, 2), keepdims=True)
        np.testing.assert_allclose(contrast, np.clip(mean + 0.45 * (self.x - mean), 0, 1), atol=1e-6)

    def test_blur_kernel(self):
        k = robustness.gaussian_kernel(1.0)
        self.assertEqual(len(k), 7)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=12)
        flat = np.full((1, 16, 16, 3), 0.25, dtype=np.float32)
        out = robustness.corrupt_batch(flat, {"family": "gaussian_blur", "value": 2.0, "severity_label": "S3"}, ["a"])
        np.testing.assert_allclose(out, 0.25, atol=1e-6)


class StatsTest(unittest.TestCase):
    def test_holm(self):
        adjusted = stats.holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_eligibility(self):
        self.assertTrue(stats.eligible([64, 100], 32))
        self.assertFalse(stats.eligible([43, 300], 32))

    def test_bootstrap_plan_stays_within_build(self):
        plan = stats.bootstrap_plan([20, 30], 8, 50, seed=5)
        self.assertEqual(plan.shape, (50, 50))
        self.assertTrue(np.all(plan[:, :20] < 20))
        self.assertTrue(np.all((plan[:, 20:] >= 20) & (plan[:, 20:] < 50)))

    def test_randomization_blocks(self):
        blocks = stats.randomization_blocks([20, 5], 8)
        self.assertEqual([len(b) for b in blocks], [8, 8, 4, 5])

    def test_identical_models_give_p_one(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 64)
        pred = rng.integers(0, 2, 64)
        preds = {a: {s: pred for s in (1, 2, 3)} for a in ("resnet18", "resnet50", "efficientnet_b0",
                                                           "mobilenet_v3_large")}
        rows = stats.contrast_tests(("R", "OT", "R_MAIN"), y, preds, [32, 32], 8,
                                    CONFIG["statistics"]["contrasts"], [1, 2, 3], replicates=200, iterations=200)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["estimate"], 0.0)
            self.assertEqual(row["raw_p"], 1.0)
            self.assertEqual(row["holm_p"], 1.0)

    def test_rng_seed_derivation(self):
        identity = "STATISTICAL_AMENDMENT_V1_1|BASE=20260916|PHASE=BOOTSTRAP|TASK=R|OT|R_MAIN|L=16"
        expected = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) % 2 ** 32
        self.assertEqual(stats.derive_rng_seed("BOOTSTRAP", ("R", "OT", "R_MAIN"), 16), expected)


if __name__ == "__main__":
    unittest.main()
