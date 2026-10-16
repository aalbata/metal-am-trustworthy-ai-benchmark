"""Grad-CAM maps and quantitative localisation metrics on the 30-sample cohort of each task.

Outputs (below the output root):
  derived/xai_cohort.csv           240 cohort entries with projected reference boxes
  derived/xai_maps/<run id>.npz    primary and complementary maps per sample (float32)
  derived/xai_samples.csv          per run and sample metrics
  results/xai_threshold_free.csv   group summaries of the threshold-free metrics (primary maps)
  results/xai_overlap.csv          group summaries of IoU, Dice, precision and recall at 0.50
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import numpy as np
import pandas as pd
import torch

import _setup
from amtrust import splits, train, xai
from amtrust.data import image_path
from amtrust.models import load_trained
from amtrust.preprocessing import geometry, preprocess_image


def build_cohort(master, boxes, config) -> pd.DataFrame:
    xcfg = config["xai"]
    valid = boxes[boxes["box_valid"].astype(str).str.lower().isin(["true", "1"])]
    cohort = xai.select_cohort(master, set(valid["benchmark_sample_id"]), xcfg["cohort_namespace"],
                               int(xcfg["samples_per_task"]))
    by_sample = {sid: g for sid, g in valid.groupby("benchmark_sample_id")}
    records = []
    for _, row in cohort.iterrows():
        g = by_sample[row["benchmark_sample_id"]]
        width, height = int(g["native_width"].iloc[0]), int(g["native_height"].iloc[0])
        mapped = [xai.project_box((b.xmin, b.ymin, b.xmax, b.ymax), row["modality"], width, height,
                                  xcfg["box_projection"]) for b in g.itertuples()]
        union = xai.continuous_union_area(mapped)
        content = geometry(row["modality"], width, height).content_area
        records.append({**row.to_dict(), "valid_box_count": len(mapped), "mapped_boxes_xyxy_json": xai.boxes_json(mapped),
                        "reference_union_area_px2": union, "image_content_area_px2": content,
                        "reference_area_fraction": union / content})
    return pd.DataFrame(records)


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()
    layout, config = _setup.context(args)
    xcfg = config["xai"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        print("CUDA is not available; pass --allow-cpu for a CPU run.")
        return 1
    master = splits.load_master(layout.master_split_csv)
    boxes = pd.read_csv(layout.boxes_csv, dtype={"benchmark_sample_id": str})
    cohort = build_cohort(master, boxes, config)
    layout.derived_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(layout.derived_dir / "xai_cohort.csv", index=False, encoding="utf-8")
    relpaths = dict(zip(master["benchmark_sample_id"], master["image_relpath"]))
    maps_dir = layout.derived_dir / "xai_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    excluded = set(xcfg["threshold_metric_excluded_runs"]) if xcfg["apply_threshold_metric_exclusion"] else set()
    rows = []
    for run in train.run_manifest(master, config):
        state = layout.run_dir(run["run_id"]) / "model_state.pt"
        if not state.is_file():
            continue
        layers, normalise = xai.layers_for_run(run["run_id"], run["architecture"], config)
        model = load_trained(run["architecture"], state).to(device).eval()
        engine = xai.TwoLayerGradCAM(model, layers)
        task_cohort = cohort[(cohort["protocol"] == run["protocol"]) & (cohort["modality"] == run["modality"])
                             & (cohort["fold_id"] == run["fold_id"])]
        stored = {}
        for _, entry in task_cohort.iterrows():
            sid = entry["benchmark_sample_id"]
            tensor = preprocess_image(image_path(layout.data, relpaths[sid]), run["modality"]).unsqueeze(0)
            maps, logits = engine(tensor.to(device=device, dtype=torch.float32), int(xcfg["target_class_index"]),
                                  normalise)
            stored[f"{sid}__primary"] = maps["primary"].astype(np.float32)
            stored[f"{sid}__complementary"] = maps["complementary"].astype(np.float32)
            mask = xai.pixel_centre_mask(xai_boxes(entry))
            free = xai.threshold_free_metrics(maps["primary"], mask, float(entry["reference_area_fraction"]),
                                              float(xcfg["top_fraction"]))
            record = {**{k: run[k] for k in ("run_id", "architecture", "protocol", "modality", "fold_id", "seed")},
                      "benchmark_sample_id": sid, "primary_layer": layers["primary"],
                      "complementary_layer": layers["complementary"], "maps_normalised": normalise,
                      "logit_good": logits[0], "logit_defect": logits[1],
                      **{k: v for k, v in free.items()},
                      "primary_complementary_cosine": xai.cosine_similarity(maps["primary"], maps["complementary"])}
            thresholds = [float(xcfg["overlap_threshold"]), *map(float, xcfg["sensitivity_thresholds"])]
            for t in thresholds:
                suffix = f"{int(round(t * 100)):03d}"
                if run["run_id"] in excluded:
                    values = {"iou": None, "dice": None, "precision": None, "recall": None}
                else:
                    values = xai.overlap_metrics(maps["primary"], mask, t)
                for name, value in values.items():
                    record[f"{name}_{suffix}"] = value
            rows.append(record)
        engine.remove()
        np.savez_compressed(maps_dir / f"{run['run_id']}.npz", **stored)
        print(f"[{run['manifest_index']:02d}] {run['run_id']}: {len(task_cohort)} samples")
    if not rows:
        print("No trained runs found.")
        return 1
    samples = pd.DataFrame(rows)
    samples.to_csv(layout.derived_dir / "xai_samples.csv", index=False, encoding="utf-8")
    numeric = samples.copy()
    for column in (*xai.THRESHOLD_FREE, *xai.OVERLAP):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    ddof = int(config["metrics"]["group_sd_ddof"])
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    xai.aggregate(numeric, xai.THRESHOLD_FREE, ddof).to_csv(layout.results_dir / "xai_threshold_free.csv",
                                                           index=False, encoding="utf-8")
    overlap = numeric[~numeric["run_id"].isin(excluded)]
    xai.aggregate(overlap, xai.OVERLAP, ddof).to_csv(layout.results_dir / "xai_overlap.csv", index=False,
                                                    encoding="utf-8")
    return 0


def xai_boxes(entry) -> list:
    """Projected reference boxes of a cohort entry."""
    return [tuple(b) for b in json.loads(entry["mapped_boxes_xyxy_json"])]


if __name__ == "__main__":
    raise SystemExit(main())
