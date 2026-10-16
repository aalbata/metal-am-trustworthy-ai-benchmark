"""Fit validation-only temperature scaling for every trained run and summarise the 32 groups.

For each run the validation logits are produced from the saved model state
(batch 8, FP32) and stored as ``validation_logits.csv`` in the run folder; the
temperature is fitted on them; uncalibrated (T = 1) and calibrated logit NLL,
binary Brier score and binary ECE are computed on the test logits.

Outputs (below the output root):
  derived/calibration_runs.csv, derived/reliability_bins.csv
  results/calibration_groups.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import _setup
from amtrust import calibration, splits, train
from amtrust.models import load_trained


def validation_logits(run: dict, master, layout, config, device) -> pd.DataFrame:
    path = layout.run_dir(run["run_id"]) / "validation_logits.csv"
    if path.is_file():
        return pd.read_csv(path, dtype={"sample_id": str})
    parts = train.partition_rows(master, run["protocol"], run["modality"], run["fold_id"])
    loader = train.make_loader(parts["val"], layout.data, int(config["training"]["evaluation_batch_size"]))
    model = load_trained(run["architecture"], layout.run_dir(run["run_id"]) / "model_state.pt").to(device)
    result = train.predict(model, loader, device)
    frame = pd.DataFrame({"sample_id": result["sample_ids"], "y_true": result["labels"],
                          "logit_good": result["logits"][:, 0].astype(float),
                          "logit_defect": result["logits"][:, 1].astype(float)})
    frame.to_csv(path, index=False, encoding="utf-8")
    return frame


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--allow-cpu", action="store_true", help="Produce validation logits on the CPU.")
    args = p.parse_args()
    layout, config = _setup.context(args)
    ccfg = config["calibration"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        print("CUDA is not available; pass --allow-cpu to produce validation logits on the CPU.")
        return 1
    master = splits.load_master(layout.master_split_csv)
    rows, bins = [], []
    for run in train.run_manifest(master, config):
        run_dir = layout.run_dir(run["run_id"])
        if not (run_dir / "test_predictions.csv").is_file():
            continue
        val = validation_logits(run, master, layout, config, device)
        test = pd.read_csv(run_dir / "test_predictions.csv", dtype={"sample_id": str})
        fit = calibration.fit_temperature(val[["logit_good", "logit_defect"]].to_numpy(np.float64),
                                          val["y_true"].to_numpy(np.int64), ccfg["temperature_lower"],
                                          ccfg["temperature_upper"], ccfg["xatol"], ccfg["maxiter"],
                                          ccfg["boundary_eta_tolerance"])
        logits = test[["logit_good", "logit_defect"]].to_numpy(np.float64)
        labels = test["y_true"].to_numpy(np.int64)
        pre = calibration.calibration_metrics(logits, labels, 1.0, ccfg["ece_bins"])
        record = {**{k: run[k] for k in ("manifest_index", "run_id", "architecture", "protocol", "modality",
                                          "fold_id", "seed")},
                  "fit_status": fit["fit_status"], "temperature": fit["temperature"],
                  "diagnostic_boundary_temperature": fit["diagnostic_boundary_temperature"],
                  "validation_nll_at_t1": fit["validation_nll_at_t1"],
                  "validation_nll_after_fit": fit["validation_nll_after_fit"],
                  "test_nll_pre": pre["nll"], "test_brier_pre": pre["brier"], "test_ece_pre": pre["ece"]}
        for b in pre["bins"]:
            bins.append({"run_id": run["run_id"], "condition": "uncalibrated", **b})
        if fit["fit_status"] == "PASS":
            post = calibration.calibration_metrics(logits, labels, fit["temperature"], ccfg["ece_bins"])
            record.update(test_nll_post=post["nll"], test_brier_post=post["brier"], test_ece_post=post["ece"])
            for b in post["bins"]:
                bins.append({"run_id": run["run_id"], "condition": "temperature_scaled", **b})
        else:
            record.update(test_nll_post=np.nan, test_brier_post=np.nan, test_ece_post=np.nan)
        rows.append(record)
        print(f"[{run['manifest_index']:02d}] {fit['fit_status']:<24} T={fit['temperature']} {run['run_id']}")
    if not rows:
        print("No completed runs found.")
        return 1
    runs = pd.DataFrame(rows)
    layout.derived_dir.mkdir(parents=True, exist_ok=True)
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(layout.derived_dir / "calibration_runs.csv", index=False, encoding="utf-8")
    pd.DataFrame(bins).to_csv(layout.derived_dir / "reliability_bins.csv", index=False, encoding="utf-8")
    groups = calibration.group_summary(runs, int(config["metrics"]["group_sd_ddof"]))
    groups.to_csv(layout.results_dir / "calibration_groups.csv", index=False, encoding="utf-8")
    print(f"Groups written: {len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
