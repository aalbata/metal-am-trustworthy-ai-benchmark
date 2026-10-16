"""Three-seed ensemble uncertainty for the 32 groups from the stored test predictions.

Outputs (below the output root):
  derived/uq_samples.csv   per-sample ensemble quantities
  results/uq_groups.csv    per-group means and error-detection AUROCs
  results/uq_summary.csv   distribution of each group quantity across groups
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import _setup
from amtrust import splits, train, uncertainty


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    args = p.parse_args()
    layout, config = _setup.context(args)
    ucfg = config["uncertainty"]
    master = splits.load_master(layout.master_split_csv)
    runs = train.run_manifest(master, config)
    groups, samples = [], []
    keys = sorted({(r["architecture"], r["protocol"], r["modality"], r["fold_id"]) for r in runs})
    for key in keys:
        members = [r for r in runs if (r["architecture"], r["protocol"], r["modality"], r["fold_id"]) == key]
        members.sort(key=lambda r: r["seed"])
        frames = []
        for run in members:
            path = layout.run_dir(run["run_id"]) / "test_predictions.csv"
            if not path.is_file():
                break
            frames.append(pd.read_csv(path, dtype={"sample_id": str}))
        if len(frames) != len(config["training"]["seeds"]):
            continue
        ids = frames[0]["sample_id"].tolist()
        if any(f["sample_id"].tolist() != ids for f in frames[1:]):
            raise RuntimeError(f"Sample order differs across seeds for {key}.")
        tensor = np.stack([f[["p_good", "p_defect"]].to_numpy(np.float64) for f in frames])
        labels = frames[0]["y_true"].to_numpy(np.int64)
        stats = uncertainty.ensemble_statistics(tensor, labels, ucfg["probability_sum_tolerance"],
                                                ucfg["entropy_bound_tolerance"], ucfg["entropy_gap_negative_tolerance"])
        group_id = "__".join(key)
        groups.append({"group_id": group_id, "architecture": key[0], "protocol": key[1], "modality": key[2],
                       "fold_id": key[3], **uncertainty.group_record(stats)})
        samples.append(pd.DataFrame({
            "group_id": group_id, "sample_id": ids, "y_true": labels,
            "ensemble_p_defect": stats["ensemble"][:, 1], "predictive_entropy": stats["predictive_entropy"],
            "expected_member_entropy": stats["expected_member_entropy"], "entropy_gap": stats["entropy_gap"],
            "pdefect_variance": stats["pdefect_variance"], "variation_ratio": stats["variation_ratio"],
            "ensemble_prediction": stats["ensemble_prediction"], "error": stats["error"]}))
    if not groups:
        print("No complete three-seed groups found.")
        return 1
    layout.derived_dir.mkdir(parents=True, exist_ok=True)
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(samples).to_csv(layout.derived_dir / "uq_samples.csv", index=False, encoding="utf-8")
    group_frame = pd.DataFrame(groups)
    group_frame.to_csv(layout.results_dir / "uq_groups.csv", index=False, encoding="utf-8")
    uncertainty.summarise_groups(group_frame).to_csv(layout.results_dir / "uq_summary.csv", index=False,
                                                     encoding="utf-8")
    print(f"Groups written: {len(group_frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
