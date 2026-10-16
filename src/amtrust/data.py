"""Cohort construction from the two Zenodo archives.

Labels come from the folder structure of ``Croped Defects and Classification.zip``
(``Croped Defects/<modality>/<Defects|Good>/<stem>.jpg``) and are mapped by
(modality, filename stem) onto the full frames of
``Original Images and Manual Labels.zip``
(``Original Images and manual labels/<modality>/<stem>.jpg``).  Bounding boxes
come from the Pascal-VOC XML files in ``OT_label`` and ``PB_label`` of the
original archive.  Only full frames are used as model inputs.

The build token is the first underscore-delimited token of the filename stem
(for example ``SI383820211201123521``) and the sequence index is the second
token.  Exact duplicate images (identical SHA-256 of the image bytes) are
represented once in the primary cohort: within a duplicate group the member
with the lowest (modality, build token, sequence index, stem) is retained.
Annotation anomalies (empty XML, invalid box) remain in the primary cohort and
are removed only in the strict sensitivity cohort.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

MANIFEST_SORT_KEYS = ["modality", "build_token", "sequence_index"]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def file_digest(path: str | Path, algorithm: str = "sha256", chunk_size: int = 8 << 20) -> str:
    """Return the hexadecimal digest of a file."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archives(data_root: Path, config: dict) -> dict:
    """Check presence and MD5 of both archives; return a status dictionary."""
    status = {}
    for key, spec in config["dataset"]["archives"].items():
        path = Path(data_root) / spec["filename"]
        record = {"path": str(path), "present": path.is_file(), "expected_md5": spec["md5"]}
        if path.is_file():
            record["md5"] = file_digest(path, "md5")
            record["md5_match"] = record["md5"] == spec["md5"]
        else:
            record["md5"] = None
            record["md5_match"] = False
        status[key] = record
    return status


# ---------------------------------------------------------------------------
# Filename and annotation parsing
# ---------------------------------------------------------------------------
def parse_filename(stem: str) -> tuple[str, int | None]:
    """Return (build token, sequence index) from a filename stem."""
    parts = str(stem).split("_")
    build_token = parts[0]
    sequence_index = None
    if len(parts) >= 2:
        try:
            sequence_index = int(parts[1])
        except ValueError:
            sequence_index = None
    return build_token, sequence_index


def parse_xml_annotation(xml_bytes: bytes) -> dict:
    """Parse a Pascal-VOC annotation into image size and object boxes.

    A box is valid when ``0 <= xmin < xmax <= width`` and
    ``0 <= ymin < ymax <= height`` with width and height taken from the XML.
    """
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    width = height = None
    if size is not None:
        try:
            width = int(float(size.findtext("width")))
        except (TypeError, ValueError):
            width = None
        try:
            height = int(float(size.findtext("height")))
        except (TypeError, ValueError):
            height = None
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        node = obj.find("bndbox")
        box = None
        valid = None
        if node is not None:
            try:
                xmin = float(node.findtext("xmin"))
                ymin = float(node.findtext("ymin"))
                xmax = float(node.findtext("xmax"))
                ymax = float(node.findtext("ymax"))
                box = (xmin, ymin, xmax, ymax)
                if width and height:
                    valid = bool(0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height)
            except (TypeError, ValueError):
                box = None
        objects.append({"name": name, "box": box, "valid": valid})
    return {"width": width, "height": height, "objects": objects}


def annotation_status(binary_label: int, object_count, all_bbox_valid) -> str:
    """Classify the annotation state of an image."""
    if int(binary_label) == 0:
        return "GOOD_NO_XML"
    if object_count is None or (isinstance(object_count, float) and np.isnan(object_count)):
        return "DEFECT_XML_METADATA_MISSING"
    if int(object_count) == 0:
        return "DEFECT_EMPTY_XML"
    if all_bbox_valid is not None and not pd.isna(all_bbox_valid) and bool(all_bbox_valid) is False:
        return "DEFECT_INVALID_BBOX"
    return "DEFECT_VALID_LOCALIZATION"


# ---------------------------------------------------------------------------
# Archive readers
# ---------------------------------------------------------------------------
def read_classification_labels(zip_path: Path, extensions, defect_folder: str = "Defects") -> dict:
    """Return {(modality, stem): label record} from the classification archive."""
    labels = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if path.suffix.lower() not in extensions:
                continue
            parts = path.parts
            if len(parts) < 4:
                raise RuntimeError(f"Unexpected classification member: {info.filename}")
            modality, class_folder = parts[-3], parts[-2]
            key = (modality, path.stem)
            if key in labels:
                raise RuntimeError(f"Duplicate classification key: {key}")
            labels[key] = {
                "class_folder": class_folder,
                "binary_label": 1 if class_folder == defect_folder else 0,
                "classification_member": info.filename,
            }
    return labels


def read_original_archive(zip_path: Path, extensions, extract_root: Path | None = None) -> tuple[dict, dict]:
    """Read full frames and XML annotations from the original archive.

    Returns ``(images, annotations)`` keyed by (modality, stem).  When
    ``extract_root`` is given, every full frame and XML file is written below
    it, preserving the archive member path.
    """
    images, annotations = {}, {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            suffix = path.suffix.lower()
            if suffix in extensions and path.parent.name in {"OT", "PB"}:
                payload = archive.read(info)
                key = (path.parent.name, path.stem)
                if key in images:
                    raise RuntimeError(f"Duplicate original image key: {key}")
                images[key] = {
                    "original_member": info.filename,
                    "original_sha256": hashlib.sha256(payload).hexdigest(),
                }
            elif suffix == ".xml":
                payload = archive.read(info)
                parent = path.parent.name
                modality = {"OT_label": "OT", "PB_label": "PB"}.get(parent, "UNKNOWN")
                parsed = parse_xml_annotation(payload)
                annotations[(modality, path.stem)] = {"xml_member": info.filename, **parsed}
            else:
                continue
            if extract_root is not None:
                target = os_path(Path(extract_root) / Path(*path.parts))
                if not os.path.isfile(target) or os.path.getsize(target) != len(payload):
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, "wb") as handle:
                        handle.write(payload)
    return images, annotations


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------
def build_manifest(labels: dict, images: dict, annotations: dict, extracted_prefix: str = "extracted") -> pd.DataFrame:
    """Build the one-row-per-image benchmark manifest with cohort flags."""
    if set(labels) != set(images):
        missing = sorted(set(labels) ^ set(images))[:5]
        raise RuntimeError(f"Classification and original archives do not map one to one: {missing}")
    records = []
    for key in sorted(images):
        modality, stem = key
        label = labels[key]
        image = images[key]
        annotation = annotations.get(key)
        has_xml = annotation is not None
        if has_xml:
            objects = annotation["objects"]
            valid_values = [obj["valid"] for obj in objects if obj["valid"] is not None]
            object_count = len(objects)
            all_bbox_valid = all(valid_values) if valid_values else None
        else:
            object_count = 0
            all_bbox_valid = None
        build_token, sequence_index = parse_filename(stem)
        records.append({
            "modality": modality,
            "stem": stem,
            "build_token": build_token,
            "sequence_index": sequence_index,
            "class_folder": label["class_folder"],
            "binary_label": int(label["binary_label"]),
            "has_xml": has_xml,
            "object_count": object_count,
            "all_bbox_valid": all_bbox_valid,
            "original_sha256": image["original_sha256"],
            "original_member": image["original_member"],
            "image_relpath": f"{extracted_prefix}/{image['original_member']}",
        })
    df = pd.DataFrame.from_records(records)
    if df["sequence_index"].isna().any():
        raise RuntimeError("A filename stem has no integer sequence index.")
    df["sequence_index"] = df["sequence_index"].astype(int)

    mismatched = int((df["binary_label"] != df["has_xml"].astype(int)).sum())
    if mismatched:
        raise RuntimeError(f"Folder label and XML presence disagree for {mismatched} images.")

    df["binary_label_name"] = np.where(df["binary_label"] == 1, "Defect", "Good")
    df["annotation_status"] = [
        annotation_status(lbl, cnt, valid)
        for lbl, cnt, valid in zip(df["binary_label"], df["object_count"], df["all_bbox_valid"])
    ]

    df["duplicate_group_size"] = df.groupby("original_sha256")["original_sha256"].transform("size").astype(int)
    df = df.sort_values(["original_sha256", "modality", "build_token", "sequence_index", "stem"]).reset_index(drop=True)
    df["duplicate_rank"] = df.groupby("original_sha256").cumcount() + 1
    df["cohort_primary_binary"] = df["duplicate_rank"] == 1
    df["cohort_strict_sensitivity"] = df["cohort_primary_binary"] & ~df["annotation_status"].isin(
        ["DEFECT_EMPTY_XML", "DEFECT_INVALID_BBOX", "DEFECT_XML_METADATA_MISSING"]
    )
    df["localization_eligible"] = (
        (df["binary_label"] == 1)
        & (df["object_count"].fillna(0) > 0)
        & (df["annotation_status"] == "DEFECT_VALID_LOCALIZATION")
    )
    df["benchmark_sample_id"] = (
        df["modality"].astype(str) + "__" + df["build_token"].astype(str) + "__"
        + df["sequence_index"].map(lambda value: f"{int(value):06d}")
    )
    if df["benchmark_sample_id"].duplicated().any():
        raise RuntimeError("Benchmark sample identifiers are not unique.")

    columns = [
        "benchmark_sample_id", "modality", "build_token", "sequence_index", "stem",
        "binary_label", "binary_label_name", "class_folder", "annotation_status",
        "has_xml", "object_count", "all_bbox_valid", "original_sha256",
        "duplicate_group_size", "duplicate_rank", "cohort_primary_binary",
        "cohort_strict_sensitivity", "localization_eligible", "original_member", "image_relpath",
    ]
    return df[columns].sort_values(MANIFEST_SORT_KEYS).reset_index(drop=True)


def build_box_table(manifest: pd.DataFrame, annotations: dict) -> pd.DataFrame:
    """Return one row per XML object with native coordinates and validity."""
    lookup = manifest.set_index(["modality", "stem"])["benchmark_sample_id"].to_dict()
    rows = []
    for (modality, stem), annotation in sorted(annotations.items()):
        for index, obj in enumerate(annotation["objects"], start=1):
            box = obj["box"] or (np.nan, np.nan, np.nan, np.nan)
            rows.append({
                "benchmark_sample_id": lookup.get((modality, stem), ""),
                "modality": modality,
                "stem": stem,
                "object_index": index,
                "object_name": obj["name"],
                "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3],
                "native_width": annotation["width"],
                "native_height": annotation["height"],
                "box_valid": bool(obj["valid"]) if obj["valid"] is not None else False,
            })
    return pd.DataFrame.from_records(rows)


def cohort_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return per-modality, per-build class counts of the primary cohort."""
    primary = manifest[manifest["cohort_primary_binary"]]
    table = (
        primary.groupby(["modality", "build_token", "binary_label"]).size()
        .unstack(fill_value=0).rename(columns={0: "good", 1: "defect"}).reset_index()
    )
    table["n"] = table["good"] + table["defect"]
    return table[["modality", "build_token", "n", "good", "defect"]]


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise RuntimeError("Unparseable boolean values in manifest.")
    return mapped.astype(bool)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV written by :func:`build_manifest` with typed columns."""
    df = pd.read_csv(path, dtype={"benchmark_sample_id": str, "build_token": str, "stem": str})
    for column in ("cohort_primary_binary", "cohort_strict_sensitivity", "localization_eligible", "has_xml"):
        df[column] = _to_bool(df[column])
    df["binary_label"] = df["binary_label"].astype(int)
    df["sequence_index"] = df["sequence_index"].astype(int)
    return df


def os_path(path: str | Path) -> str:
    """Absolute path string; on Windows long paths receive the extended-length form."""
    text = str(Path(path).absolute())
    prefix = "\\\\?\\"
    if os.name == "nt" and len(text) >= 240 and not text.startswith(prefix):
        return prefix + text
    return text


def image_path(data_root: Path, relpath: str) -> str:
    """Return the absolute path of a manifest image."""
    return os_path(Path(data_root) / Path(*PurePosixPath(relpath).parts))
