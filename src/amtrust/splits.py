"""Protocol R (mixed-build) and Protocol S (leave-one-build-out) splits.

Both protocols use ``sklearn.model_selection.train_test_split`` with
``random_state`` equal to the split seed, ``shuffle=True`` and stratification
by the joint build-token x class stratum.  The result depends on the input row
order, which is the primary cohort in manifest order (modality, build token,
sequence index), restricted to one modality and re-indexed from zero.

Protocol R: 70 % train and 30 % temporary, then the temporary part is split
50/50 into validation and test.  Protocol S: each build in turn is the complete
test set; the remaining two builds are split 85/15 into train and validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import MANIFEST_SORT_KEYS

ASSIGNMENT_COLUMNS = [
    "protocol", "modality", "fold_id", "held_out_build", "partition", "split_seed",
    "benchmark_sample_id", "build_token", "sequence_index", "binary_label",
    "binary_label_name", "annotation_status", "cohort_strict_sensitivity",
    "original_sha256", "stem", "original_member", "image_relpath",
]
PARTITIONS = ("train", "validation", "test")


def make_stratum(frame: pd.DataFrame) -> pd.Series:
    """Joint build-token x class stratum label."""
    return frame["build_token"].astype(str) + "__LABEL_" + frame["binary_label"].astype(int).astype(str)


def primary_cohort(manifest: pd.DataFrame) -> pd.DataFrame:
    """Primary cohort rows in manifest order."""
    ordered = manifest.sort_values(MANIFEST_SORT_KEYS).reset_index(drop=True)
    return ordered[ordered["cohort_primary_binary"]].copy()


def _annotate(frame, protocol, modality, fold_id, partition, held_out, seed) -> pd.DataFrame:
    out = frame.copy()
    out["protocol"] = protocol
    out["modality"] = modality
    out["fold_id"] = fold_id
    out["partition"] = partition
    out["held_out_build"] = held_out or ""
    out["split_seed"] = seed
    return out[ASSIGNMENT_COLUMNS].copy()


def _assemble(parts: dict, protocol, modality, fold_id, held_out, seed) -> pd.DataFrame:
    frames = [_annotate(frame, protocol, modality, fold_id, name, held_out, seed) for name, frame in parts.items()]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["partition", "build_token", "sequence_index"])
        .reset_index(drop=True)
    )


def protocol_r(primary: pd.DataFrame, modality: str, seed: int, temporary_fraction: float = 0.30,
               test_fraction_of_temporary: float = 0.50, fold_id: str = "R_MAIN") -> pd.DataFrame:
    """Protocol R assignment for one modality."""
    df = primary[primary["modality"] == modality].copy().reset_index(drop=True)
    df["_stratum"] = make_stratum(df)
    train_df, temp_df = train_test_split(
        df, test_size=temporary_fraction, random_state=seed, shuffle=True, stratify=df["_stratum"])
    val_df, test_df = train_test_split(
        temp_df, test_size=test_fraction_of_temporary, random_state=seed, shuffle=True, stratify=temp_df["_stratum"])
    return _assemble({"train": train_df, "validation": val_df, "test": test_df}, "R", modality, fold_id, None, seed)


def protocol_s(primary: pd.DataFrame, modality: str, seed: int, validation_fraction: float = 0.15,
               fold_template: str = "S{fold}_HOLDOUT_{build}") -> list[tuple[int, str, str, pd.DataFrame]]:
    """Protocol S assignments for one modality: list of (fold number, fold id, held-out build, frame)."""
    df = primary[primary["modality"] == modality].copy().reset_index(drop=True)
    builds = sorted(df["build_token"].unique())
    if len(builds) != 3:
        raise RuntimeError(f"{modality} does not contain exactly three builds.")
    folds = []
    for number, held_out in enumerate(builds, start=1):
        test_df = df[df["build_token"] == held_out].copy().reset_index(drop=True)
        dev_df = df[df["build_token"] != held_out].copy().reset_index(drop=True)
        dev_df["_stratum"] = make_stratum(dev_df)
        train_df, val_df = train_test_split(
            dev_df, test_size=validation_fraction, random_state=seed, shuffle=True, stratify=dev_df["_stratum"])
        fold_id = fold_template.format(fold=number, build=held_out)
        frame = _assemble({"train": train_df, "validation": val_df, "test": test_df}, "S", modality, fold_id, held_out, seed)
        folds.append((number, fold_id, held_out, frame))
    return folds


def generate_all(manifest: pd.DataFrame, config: dict) -> dict:
    """Generate all eight task assignments; return {filename: frame} plus the master frame."""
    split_cfg = config["splits"]
    seed = int(split_cfg["seed"])
    primary = primary_cohort(manifest)
    files = {}
    for modality in ("OT", "PB"):
        frame = protocol_r(primary, modality, seed,
                           split_cfg["protocol_R"]["temporary_fraction"],
                           split_cfg["protocol_R"]["test_fraction_of_temporary"],
                           split_cfg["protocol_R"]["fold_id"])
        files[f"protocol_R_{modality}_70_15_15_seed_{seed}.csv"] = frame
    for modality in ("OT", "PB"):
        for number, _fold_id, held_out, frame in protocol_s(
                primary, modality, seed, split_cfg["protocol_S"]["validation_fraction"],
                split_cfg["protocol_S"]["fold_id_template"]):
            files[f"protocol_S_{modality}_fold_{number}_holdout_{held_out}_seed_{seed}.csv"] = frame
    master = (
        pd.concat(list(files.values()), ignore_index=True)
        .sort_values(["protocol", "modality", "fold_id", "partition", "build_token", "sequence_index"])
        .reset_index(drop=True)
    )
    return {"files": files, "master": master}


def task_key(protocol: str, modality: str, fold_id: str) -> str:
    """Short task name such as R_OT or S_PB_S2."""
    if protocol == "R":
        return f"R_{modality}"
    return f"S_{modality}_{fold_id.split('_')[0]}"


def task_table(master: pd.DataFrame) -> pd.DataFrame:
    """Distinct tasks sorted by protocol, modality and fold id."""
    return (
        master[["protocol", "modality", "fold_id"]].drop_duplicates()
        .sort_values(["protocol", "modality", "fold_id"]).reset_index(drop=True)
    )


def split_counts(master: pd.DataFrame) -> pd.DataFrame:
    """Per-task train/validation/test counts with class composition."""
    rows = []
    for _, task in task_table(master).iterrows():
        sub = master[(master["protocol"] == task.protocol) & (master["modality"] == task.modality)
                     & (master["fold_id"] == task.fold_id)]
        record = {
            "task": task_key(task.protocol, task.modality, task.fold_id),
            "protocol": task.protocol, "modality": task.modality, "fold_id": task.fold_id,
            "held_out_build_token": sub["held_out_build"].iloc[0] if task.protocol == "S" else "",
        }
        for name, short in (("train", "train"), ("validation", "val"), ("test", "test")):
            part = sub[sub["partition"] == name]
            record[f"{short}_n"] = int(len(part))
            record[f"{short}_good"] = int((part["binary_label"] == 0).sum())
            record[f"{short}_defect"] = int((part["binary_label"] == 1).sum())
            record[f"{short}_prevalence_defect"] = round(record[f"{short}_defect"] / max(len(part), 1), 4)
            record[f"{short}_builds"] = int(part["build_token"].nunique())
            record[f"strict_{short}_n"] = int(part["cohort_strict_sensitivity"].astype(bool).sum())
        rows.append(record)
    return pd.DataFrame(rows)


def compare_with_expected(counts: pd.DataFrame, expected: dict) -> pd.DataFrame:
    """Compare split counts with the configured historical counts."""
    rows = []
    for _, row in counts.iterrows():
        exp = expected[row["task"]]
        observed = {"train": row["train_n"], "val": row["val_n"], "test": row["test_n"],
                    "test_good": row["test_good"], "test_defect": row["test_defect"]}
        match = all(int(observed[k]) == int(exp[k]) for k in exp)
        rows.append({"task": row["task"], **{f"observed_{k}": v for k, v in observed.items()},
                     **{f"expected_{k}": v for k, v in exp.items()}, "match": match})
    return pd.DataFrame(rows)


def write_splits(result: dict, directory: Path) -> dict:
    """Write the eight task files and the master file; return {name: path}."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in result["files"].items():
        path = directory / name
        frame.to_csv(path, index=False, encoding="utf-8")
        written[name] = path
    master_path = directory / "all_split_assignments.csv"
    result["master"].to_csv(master_path, index=False, encoding="utf-8")
    written["all_split_assignments.csv"] = master_path
    return written


def load_master(path: str | Path) -> pd.DataFrame:
    """Read the master split file with typed columns."""
    df = pd.read_csv(path, dtype={"benchmark_sample_id": str, "build_token": str, "held_out_build": str,
                                  "fold_id": str}, keep_default_na=False)
    df["binary_label"] = df["binary_label"].astype(int)
    df["sequence_index"] = df["sequence_index"].astype(int)
    df["cohort_strict_sensitivity"] = df["cohort_strict_sensitivity"].astype(str).str.lower().isin(["true", "1"])
    return df
