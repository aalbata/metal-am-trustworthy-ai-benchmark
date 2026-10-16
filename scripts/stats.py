"""Paired block bootstrap intervals and block randomisation tests for the architecture contrasts.

Block length 16 is the primary analysis; 8 and 32 are sensitivity analyses
(32 only for tasks whose build series all have at least 64 test images).

Output (below the output root): results/stats_tests.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import _setup
from amtrust import splits, stats, train


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--block-lengths", default=None, help="Comma-separated block lengths (default: 16,8,32).")
    args = p.parse_args()
    layout, config = _setup.context(args)
    scfg = config["statistics"]
    seeds = list(config["training"]["seeds"])
    lengths = ([int(v) for v in args.block_lengths.split(",")] if args.block_lengths
               else [scfg["primary_block_length"], *scfg["sensitivity_block_lengths"]])
    master = splits.load_master(layout.master_split_csv)
    tasks = splits.task_table(master)
    runs = train.run_manifest(master, config)
    rows = []
    for _, task_row in tasks.iterrows():
        task = (task_row.protocol, task_row.modality, task_row.fold_id)
        test = master[(master["protocol"] == task[0]) & (master["modality"] == task[1]) & (master["fold_id"] == task[2])
                      & (master["partition"] == "test")]
        ordered = test.sort_values(["build_token", "sequence_index", "benchmark_sample_id"]).reset_index(drop=True)
        series = [int(n) for _, n in ordered.groupby("build_token", sort=True).size().items()]
        order = ordered["benchmark_sample_id"].tolist()
        labels = ordered["binary_label"].to_numpy()
        predictions = {}
        complete = True
        for run in runs:
            if (run["protocol"], run["modality"], run["fold_id"]) != task:
                continue
            path = layout.run_dir(run["run_id"]) / "test_predictions.csv"
            if not path.is_file():
                complete = False
                break
            frame = pd.read_csv(path, dtype={"sample_id": str}).set_index("sample_id").loc[order]
            predictions.setdefault(run["architecture"], {})[run["seed"]] = frame["y_pred"].to_numpy()
        if not complete:
            print(f"Skipping {stats.task_string(task)}: not all runs are available.")
            continue
        for length in lengths:
            if not stats.eligible(series, length):
                print(f"{stats.task_string(task)} L={length}: not eligible (build series {series}).")
                continue
            rows.extend(stats.contrast_tests(task, labels, predictions, series, length, scfg["contrasts"], seeds,
                                             scfg["metrics"], scfg["bootstrap_replicates"],
                                             scfg["randomization_iterations"], scfg["base_resampling_seed"],
                                             scfg["confidence_level"]))
            print(f"{stats.task_string(task)} L={length}: done")
    if not rows:
        print("No tests computed.")
        return 1
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(layout.results_dir / "stats_tests.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
