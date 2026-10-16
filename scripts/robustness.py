"""Corrupted-input inference for every trained run and the robustness direction summary.

Each test batch (8 images in stored prediction order) is resized and padded,
corrupted under the 15 conditions on [0, 1], normalised and classified in FP32;
Defect is predicted when P(Defect) >= 0.5.  Clean reference metrics come from
the stored test predictions of the run.

Outputs (below the output root):
  derived/robustness_predictions.csv   per sample, run and condition
  derived/robustness_run_metrics.csv   per run and condition
  derived/robustness_group_metrics.csv three-seed means per group and condition
  results/robustness_direction.csv     groups with lower / equal / higher corrupted performance
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv

import numpy as np
import pandas as pd
import torch

import _setup
from amtrust import robustness, splits, train
from amtrust.data import image_path
from amtrust.models import load_trained
from amtrust.preprocessing import array_batch_to_input, base_array


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()
    layout, config = _setup.context(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        print("CUDA is not available; pass --allow-cpu for a CPU run (not comparable with reported results).")
        return 1
    threshold = float(config["training"]["decision_threshold"])
    batch_size = int(config["robustness"]["inference_batch_size"])
    conditions = robustness.conditions(config)
    master = splits.load_master(layout.master_split_csv)
    relpaths = dict(zip(master["benchmark_sample_id"], master["image_relpath"]))
    layout.derived_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "sample_id", "y_true", "condition_id", "p_good", "p_defect", "y_pred"]
    metric_rows = []
    with open(layout.derived_dir / "robustness_predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in train.run_manifest(master, config):
            run_dir = layout.run_dir(run["run_id"])
            if not (run_dir / "test_predictions.csv").is_file():
                continue
            clean = pd.read_csv(run_dir / "test_predictions.csv", dtype={"sample_id": str})
            model = load_trained(run["architecture"], run_dir / "model_state.pt").to(device).eval()
            store = {c["condition_id"]: {"p": [], "pred": []} for c in conditions}
            for start in range(0, len(clean), batch_size):
                batch = clean.iloc[start:start + batch_size]
                ids = batch["sample_id"].tolist()
                bases = np.stack([base_array(image_path(layout.data, relpaths[s]), run["modality"]) for s in ids])
                for condition in conditions:
                    corrupted = robustness.corrupt_batch(bases, condition, ids)
                    tensor = array_batch_to_input(corrupted).to(device=device, dtype=torch.float32)
                    with torch.inference_mode():
                        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()
                    preds = (probs[:, 1] >= threshold).astype(int)
                    store[condition["condition_id"]]["p"].extend(probs[:, 1].tolist())
                    store[condition["condition_id"]]["pred"].extend(preds.tolist())
                    for sid, y, pr, yp in zip(ids, batch["y_true"], probs, preds):
                        writer.writerow({"run_id": run["run_id"], "sample_id": sid, "y_true": int(y),
                                         "condition_id": condition["condition_id"], "p_good": float(pr[0]),
                                         "p_defect": float(pr[1]), "y_pred": int(yp)})
            y = clean["y_true"].to_numpy(np.int64)
            clean_metrics = robustness.run_condition_metrics(y, clean["y_pred"], clean["p_defect"])
            for condition in conditions:
                values = store[condition["condition_id"]]
                corrupted_metrics = robustness.run_condition_metrics(y, values["pred"], values["p"])
                metric_rows.append({
                    **{k: run[k] for k in ("run_id", "architecture", "protocol", "modality", "fold_id", "seed")},
                    "condition_id": condition["condition_id"], "family": condition["family"],
                    "severity_label": condition["severity_label"],
                    **{f"clean_{k}": v for k, v in clean_metrics.items()},
                    **{f"corrupted_{k}": v for k, v in corrupted_metrics.items()},
                    "clean_to_corrupt_flip_rate": float(np.mean(np.asarray(values["pred"]) != clean["y_pred"].to_numpy())),
                    "mean_absolute_p_defect_shift": float(np.mean(np.abs(np.asarray(values["p"]) - clean["p_defect"].to_numpy()))),
                })
            print(f"[{run['manifest_index']:02d}] {run['run_id']}")
    if not metric_rows:
        print("No completed runs found.")
        return 1
    run_metrics = pd.DataFrame(metric_rows)
    run_metrics.to_csv(layout.derived_dir / "robustness_run_metrics.csv", index=False, encoding="utf-8")
    groups = robustness.group_degradation(run_metrics, int(config["metrics"]["group_sd_ddof"]))
    groups.to_csv(layout.derived_dir / "robustness_group_metrics.csv", index=False, encoding="utf-8")
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    robustness.direction_counts(groups, conditions).to_csv(layout.results_dir / "robustness_direction.csv",
                                                           index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
