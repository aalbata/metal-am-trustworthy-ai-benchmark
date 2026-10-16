"""Verify the Zenodo archives, extract the full frames and build the benchmark manifest.

Outputs (below the output root):
  manifest/benchmark_manifest.csv  one row per full frame with labels and cohort flags
  manifest/reference_boxes.csv     one row per XML object with native coordinates
  manifest/cohort_counts.csv       primary-cohort class counts per modality and build
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import _setup
from amtrust import data


def main() -> int:
    p = _setup.parser(__doc__.splitlines()[0])
    p.add_argument("--skip-md5", action="store_true", help="Do not recompute archive MD5 checksums.")
    p.add_argument("--no-extract", action="store_true", help="Build the manifest without writing image files.")
    args = p.parse_args()
    layout, config = _setup.context(args)
    ds = config["dataset"]
    extensions = set(ds["image_extensions"])

    print(f"Data root   : {layout.data}")
    print(f"Output root : {layout.outputs}")
    if not args.skip_md5:
        status = data.verify_archives(layout.data, config)
        for key, record in status.items():
            print(f"{key:<15}: present={record['present']} md5_match={record['md5_match']}")
        if not all(r["md5_match"] for r in status.values()):
            print("Archive check failed; download both archives from Zenodo into the data root.")
            return 1

    original = layout.archive(ds["archives"]["original"]["filename"])
    classification = layout.archive(ds["archives"]["classification"]["filename"])
    print("Reading classification labels ...")
    labels = data.read_classification_labels(classification, extensions, ds["defect_folder_name"])
    print(f"  labelled images: {len(labels)}")
    print("Reading full frames and annotations ...")
    images, annotations = data.read_original_archive(original, extensions,
                                                      None if args.no_extract else layout.extracted)
    print(f"  full frames: {len(images)}  XML files: {len(annotations)}")

    manifest = data.build_manifest(labels, images, annotations)
    boxes = data.build_box_table(manifest, annotations)
    layout.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(layout.manifest_csv, index=False, encoding="utf-8")
    boxes.to_csv(layout.boxes_csv, index=False, encoding="utf-8")
    counts = data.cohort_counts(manifest)
    counts.to_csv(layout.manifest_dir / "cohort_counts.csv", index=False, encoding="utf-8")

    summary = {
        "source_images": int(len(manifest)),
        "primary_binary": int(manifest["cohort_primary_binary"].sum()),
        "strict_sensitivity": int(manifest["cohort_strict_sensitivity"].sum()),
        "localization_eligible": int(manifest["localization_eligible"].sum()),
        "exact_duplicate_copies_excluded": int((~manifest["cohort_primary_binary"]).sum()),
    }
    print(json.dumps(summary, indent=2))
    print(counts.to_string(index=False))
    expected = ds["expected_counts"]
    mismatches = {k: (v, expected[k]) for k, v in summary.items() if expected.get(k) != v}
    print("Cohort counts match configuration:", "yes" if not mismatches else f"no {mismatches}")
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
