"""Generate the Protocol R and Protocol S split files from the benchmark manifest.

Outputs (below the output root):
  splits/<eight task files>.csv and splits/all_split_assignments.csv
  results/split_counts.csv   per-task partition sizes and class counts
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _setup
from amtrust import data, splits


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    args = p.parse_args()
    layout, config = _setup.context(args)
    manifest = data.load_manifest(layout.manifest_csv)
    result = splits.generate_all(manifest, config)
    written = splits.write_splits(result, layout.splits_dir)
    for name, path in written.items():
        print(f"{data.file_digest(path)}  {name}")
    counts = splits.split_counts(result["master"])
    layout.results_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(layout.results_dir / "split_counts.csv", index=False, encoding="utf-8")
    comparison = splits.compare_with_expected(counts, config["splits"]["expected_counts"])
    print(comparison.to_string(index=False))
    ok = bool(comparison["match"].all())
    print("All task counts match configuration:", "yes" if ok else "no")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
