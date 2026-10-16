"""Box projection, reference masks, localisation metrics, Grad-CAM shapes and split stratification."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from amtrust import preprocessing, splits, xai
from amtrust.models import build_classifier, resolve_layer

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "benchmark.json").read_text(encoding="utf-8"))


class GeometryTest(unittest.TestCase):
    def test_pb_geometry(self):
        g = preprocessing.geometry("PB", 1280, 1024)
        self.assertEqual((g.new_width, g.new_height, g.left, g.top, g.right, g.bottom), (512, 410, 0, 51, 0, 51))
        self.assertEqual(g.content_area, 512 * 410)

    def test_ot_geometry(self):
        g = preprocessing.geometry("OT", 2000, 2000)
        self.assertEqual((g.new_width, g.new_height, g.top), (512, 512, 0))


class BoxProjectionTest(unittest.TestCase):
    def test_ot_projection(self):
        box = xai.project_box((100, 200, 300, 400), "OT", 2000, 2000)
        np.testing.assert_allclose(box, [25.6, 51.2, 76.8, 102.4])

    def test_pb_projection_modes(self):
        resized = xai.project_box((100, 100, 200, 300), "PB", 1280, 1024, "resized_extent")
        long_side = xai.project_box((100, 100, 200, 300), "PB", 1280, 1024, "long_side")
        np.testing.assert_allclose(resized, [40.0, 51 + 100 * 410 / 1024, 80.0, 51 + 300 * 410 / 1024])
        np.testing.assert_allclose(long_side, [40.0, 51 + 40.0, 80.0, 51 + 120.0])

    def test_pixel_centre_mask(self):
        mask = xai.pixel_centre_mask([(1.0, 2.0, 3.6, 3.4)], size=8)
        # Columns whose centres lie in [1, 3.6): 1, 2, 3; rows in [2, 3.4): 2.
        expected = np.zeros((8, 8), dtype=bool)
        expected[2, 1:4] = True
        np.testing.assert_array_equal(mask, expected)

    def test_half_open_edges(self):
        mask = xai.pixel_centre_mask([(0.0, 0.0, 2.5, 1.5)], size=4)
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(mask[0, 0] and mask[0, 1] and not mask[0, 2])

    def test_union_area(self):
        boxes = [(0, 0, 2, 2), (1, 1, 3, 3)]
        self.assertAlmostEqual(xai.continuous_union_area(boxes), 7.0)


class LocalisationMetricTest(unittest.TestCase):
    def test_metrics_on_constructed_map(self):
        size = 16
        cam = np.zeros((size, size), dtype=np.float32)
        cam[4:8, 4:8] = 1.0
        cam[0, 0] = 0.5
        mask = xai.pixel_centre_mask([(4, 4, 8, 8)], size=size)
        free = xai.threshold_free_metrics(cam, mask, reference_area_fraction=16 / 256, top_fraction=0.10)
        self.assertAlmostEqual(free["reference_energy_fraction"], 16 / 16.5)
        self.assertAlmostEqual(free["energy_enrichment_ratio"], (16 / 16.5) / (16 / 256))
        self.assertEqual(free["pointing_game_hit"], 1)
        k = xai.top_k(0.10, size * size)
        self.assertEqual(k, 26)
        self.assertAlmostEqual(free["top10_energy_fraction"], 1.0)
        p = np.array([1 / 16.5] * 16 + [0.5 / 16.5])
        self.assertAlmostEqual(free["normalized_spatial_entropy"], float(-(p * np.log(p)).sum() / np.log(256)))
        over = xai.overlap_metrics(cam, mask, 0.5)
        self.assertAlmostEqual(over["iou"], 16 / 17)
        self.assertAlmostEqual(over["dice"], 32 / 33)
        self.assertAlmostEqual(over["precision"], 16 / 17)
        self.assertAlmostEqual(over["recall"], 1.0)

    def test_top_k_full_size(self):
        self.assertEqual(xai.top_k(0.10, 512 * 512), 26215)

    def test_zero_map(self):
        cam = np.zeros((8, 8), dtype=np.float32)
        out = xai.threshold_free_metrics(cam, np.ones((8, 8), bool), 0.5)
        self.assertTrue(out["zero_map"])
        self.assertIsNone(out["pointing_game_hit"])
        self.assertIsNone(xai.cosine_similarity(cam, cam))

    def test_normalise_map(self):
        m = xai.normalise_map(np.array([[2.0, 4.0], [3.0, 6.0]]))
        self.assertAlmostEqual(float(m.min()), 0.0)
        self.assertAlmostEqual(float(m.max()), 1.0, places=9)

    def test_cohort_selection_key(self):
        master = pd.DataFrame({
            "protocol": ["R"] * 5, "modality": ["OT"] * 5, "fold_id": ["R_MAIN"] * 5,
            "partition": ["test"] * 4 + ["train"], "binary_label": [1, 1, 1, 0, 1],
            "benchmark_sample_id": [f"OT__B__{i:06d}" for i in range(5)],
        })
        valid = {f"OT__B__{i:06d}" for i in range(5)}
        cohort = xai.select_cohort(master, valid, "NS", per_task=2)
        keys = sorted((xai.selection_key(("R", "OT", "R_MAIN"), s, "NS"), s) for s in
                      ["OT__B__000000", "OT__B__000001", "OT__B__000002"])
        self.assertEqual(cohort["benchmark_sample_id"].tolist(), [k[1] for k in keys[:2]])


class GradCamTest(unittest.TestCase):
    def test_two_layer_shapes_and_normalisation(self):
        torch.manual_seed(0)
        model = build_classifier("resnet18").eval()
        engine = xai.TwoLayerGradCAM(model, {"primary": "layer3", "complementary": "layer4"})
        maps, logits = engine(torch.randn(1, 3, 64, 64), target_index=1, normalise=True, size=96)
        engine.remove()
        self.assertEqual(set(maps), {"primary", "complementary"})
        for cam in maps.values():
            self.assertEqual(cam.shape, (96, 96))
            self.assertTrue(np.isfinite(cam).all())
            self.assertGreaterEqual(float(cam.min()), 0.0)
            self.assertLessEqual(float(cam.max()), 1.0)
        self.assertEqual(logits.shape, (2,))

    def test_configured_layers_exist(self):
        for architecture, layers in CONFIG["xai"]["layers"].items():
            model = build_classifier(architecture)
            for name in layers.values():
                self.assertIsNotNone(resolve_layer(model, name))
        for run_id, override in CONFIG["xai"]["run_layer_overrides"].items():
            model = build_classifier(run_id.split("__")[0])
            resolve_layer(model, override["primary"])
            resolve_layer(model, override["complementary"])

    def test_layer_map_and_overrides(self):
        layers, norm = xai.layers_for_run("efficientnet_b0__R__OT__R_MAIN__seed_20260916", "efficientnet_b0", CONFIG)
        self.assertEqual(layers["primary"], "features.3")
        self.assertTrue(norm)
        layers, norm = xai.layers_for_run("resnet18__R__OT__R_MAIN__seed_20260916", "resnet18", CONFIG)
        self.assertEqual(layers["primary"], "layer2.1")
        self.assertFalse(norm)
        layers, norm = xai.layers_for_run("resnet18__R__OT__R_MAIN__seed_20260917", "resnet18", CONFIG)
        self.assertEqual(layers["primary"], "layer3")
        self.assertTrue(norm)
        cfg = json.loads(json.dumps(CONFIG))
        cfg["xai"]["apply_run_layer_overrides"] = False
        layers, norm = xai.layers_for_run("efficientnet_b0__R__OT__R_MAIN__seed_20260916", "efficientnet_b0", cfg)
        self.assertEqual(layers["primary"], "features.6")
        self.assertTrue(norm)


class SplitTest(unittest.TestCase):
    def synthetic_manifest(self):
        rng = np.random.default_rng(11)
        rows = []
        for modality in ("OT", "PB"):
            for build, n, prevalence in (("SI0001", 300, 0.1), ("SI0002", 120, 0.5), ("SI0003", 400, 0.7)):
                for seq in range(n):
                    rows.append({"modality": modality, "build_token": build, "sequence_index": seq,
                                 "binary_label": int(rng.random() < prevalence), "cohort_primary_binary": True,
                                 "benchmark_sample_id": f"{modality}__{build}__{seq:06d}"})
        df = pd.DataFrame(rows)
        for column in ("binary_label_name", "annotation_status", "original_sha256", "stem", "original_member",
                       "image_relpath"):
            df[column] = df["benchmark_sample_id"] if column != "binary_label_name" else df["binary_label"].astype(str)
        df["cohort_strict_sensitivity"] = True
        return df

    def test_protocol_r_stratification(self):
        manifest = self.synthetic_manifest()
        result = splits.generate_all(manifest, CONFIG)
        r = result["files"]["protocol_R_OT_70_15_15_seed_20260916.csv"]
        self.assertEqual(r["benchmark_sample_id"].nunique(), len(r))
        source = manifest[manifest["modality"] == "OT"]
        for (build, label), sub in source.groupby(["build_token", "binary_label"]):
            share = r[(r["build_token"] == build) & (r["binary_label"] == label)]["partition"].value_counts()
            n = len(sub)
            self.assertLessEqual(abs(share.get("train", 0) - 0.7 * n), 2)
            self.assertLessEqual(abs(share.get("test", 0) - 0.15 * n), 2)

    def test_protocol_s_holds_out_whole_build(self):
        result = splits.generate_all(self.synthetic_manifest(), CONFIG)
        name = "protocol_S_PB_fold_2_holdout_SI0002_seed_20260916.csv"
        frame = result["files"][name]
        test = frame[frame["partition"] == "test"]
        self.assertEqual(set(test["build_token"]), {"SI0002"})
        self.assertEqual(len(test), 120)
        dev = frame[frame["partition"] != "test"]
        self.assertNotIn("SI0002", set(dev["build_token"]))
        self.assertAlmostEqual((dev["partition"] == "validation").mean(), 0.15, delta=0.01)

    def test_split_is_reproducible(self):
        a = splits.generate_all(self.synthetic_manifest(), CONFIG)["master"]
        b = splits.generate_all(self.synthetic_manifest(), CONFIG)["master"]
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 2 * (820 + 3 * 820))


if __name__ == "__main__":
    unittest.main()
