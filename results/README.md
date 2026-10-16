# Results

This folder holds the reported benchmark results as CSV files. `scripts/reproduce.py --fast`, `scripts/make_tables.py` and `scripts/make_figures.py` read them from here.

| File | Content |
|---|---|
| `split_counts.csv` | Train, validation and test sizes and class counts of the eight tasks |
| `predictive_groups.csv` | Predictive metrics (mean and SD over three seeds) of the 32 groups |
| `calibration_groups.csv` | NLL, Brier score and ECE before and after temperature scaling, with directions |
| `uq_groups.csv` | Three-seed ensemble uncertainty and error-detection AUROC per group |
| `uq_summary.csv` | Distribution of the uncertainty quantities across the 32 groups |
| `robustness_direction.csv` | Groups with lower, equal or higher performance under each of the 15 corruptions |
| `xai_threshold_free.csv` | Threshold-free Grad-CAM localisation metrics |
| `xai_overlap.csv` | Grad-CAM overlap metrics at threshold 0.50 |
| `efficiency.csv` | Latency, throughput and peak memory of eight architecture x precision contexts |
| `stats_tests.csv` | Block bootstrap intervals and block randomisation tests of architecture contrasts |
| `baselines.csv` | Frozen-embedding baselines (majority, linear SVM, random forest) |

Column definitions are given in the repository `README.md` (section "Results files and column definitions"). The full pipeline (`scripts/run_all.py`) writes files with the same names to `<output root>/results/` and leaves this folder unchanged.
