"""Filename and annotation parsing, cohort rules, and table generation from synthetic results."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from amtrust import data, tables

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "benchmark.json").read_text(encoding="utf-8"))

XML = b"""<annotation><size><width>1280</width><height>1024</height><depth>1</depth></size>
<object><name>defects</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
<object><name>defects</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>50</xmax><ymax>40</ymax></bndbox></object>
</annotation>"""


class ParsingTest(unittest.TestCase):
    def test_filename(self):
        self.assertEqual(data.parse_filename("SI383820240226095904_120_002_420"), ("SI383820240226095904", 120))
        self.assertEqual(data.parse_filename("SI383820211201123521_00087_20211201T125928.564000"),
                         ("SI383820211201123521", 87))

    def test_xml_validity(self):
        parsed = data.parse_xml_annotation(XML)
        self.assertEqual((parsed["width"], parsed["height"]), (1280, 1024))
        self.assertEqual([o["valid"] for o in parsed["objects"]], [True, False])

    def test_annotation_status(self):
        self.assertEqual(data.annotation_status(0, 0, None), "GOOD_NO_XML")
        self.assertEqual(data.annotation_status(1, 0, None), "DEFECT_EMPTY_XML")
        self.assertEqual(data.annotation_status(1, 2, False), "DEFECT_INVALID_BBOX")
        self.assertEqual(data.annotation_status(1, 2, True), "DEFECT_VALID_LOCALIZATION")

    def test_manifest_dedup_keeps_lowest_sequence(self):
        labels, images, annotations = {}, {}, {}
        for seq, sha in ((358, "aa"), (658, "aa"), (12, "bb")):
            stem = f"SI0001_{seq}_000_000"
            labels[("OT", stem)] = {"class_folder": "Good", "binary_label": 0, "classification_member": stem}
            images[("OT", stem)] = {"original_member": f"x/OT/{stem}.jpg", "original_sha256": sha}
        manifest = data.build_manifest(labels, images, annotations)
        kept = manifest[manifest["cohort_primary_binary"]]["sequence_index"].tolist()
        self.assertEqual(sorted(kept), [12, 358])
        self.assertEqual(manifest["sequence_index"].tolist(), [12, 358, 658])


class TableTest(unittest.TestCase):
    def test_tables_from_synthetic_results(self):
        rows = []
        for arch in ("resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large"):
            for protocol, fold in (("R", "R_MAIN"), ("S", "S1_HOLDOUT_SI383820211201123521")):
                rows.append({"architecture": arch, "protocol": protocol, "modality": "OT", "fold_id": fold,
                             "test_n": 10, **{f"{m}_{s}": 0.5 for m in ("macro_f1", "balanced_accuracy", "auroc",
                                                                         "auprc", "nll", "brier") for s in ("mean", "sd")}})
        with tempfile.TemporaryDirectory() as tmp:
            res, out = Path(tmp) / "results", Path(tmp) / "tables"
            res.mkdir()
            pd.DataFrame(rows).to_csv(res / "predictive_groups.csv", index=False)
            status = tables.make_all(res, out, CONFIG)
            self.assertEqual(status["table_predictive"], "written")
            self.assertTrue(status["table_efficiency"].startswith("skipped"))
            tex = (out / "table_predictive.tex").read_text(encoding="utf-8")
            self.assertIn("\\toprule", tex)
            self.assertIn("S OT B1", tex)
            self.assertIn("$0.5000 \\pm 0.5000$", tex)
            plain = pd.read_csv(out / "table_predictive.csv")
            self.assertEqual(len(plain), 8)


if __name__ == "__main__":
    unittest.main()
