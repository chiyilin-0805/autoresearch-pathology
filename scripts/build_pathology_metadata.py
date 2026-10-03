"""Build a non-image pathology inventory without inventing patient identities.

The source datasets currently present in ``AI病理`` do not expose a reliable
image-to-patient mapping. This script therefore leaves ``patient_id`` and
``slide_id`` empty unless an explicit, anonymized mapping is provided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CRC_DATASETS = {"CRC-VAL-HE-7K", "NCT-CRC-HE-100K"}


def iter_images(source: Path) -> Iterable[Path]:
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def infer_fields(relative_path: Path) -> dict[str, str]:
    parts = relative_path.parts
    dataset = parts[0] if parts else "unknown"
    tissue_class = relative_path.parent.name

    if dataset in CRC_DATASETS:
        organ = "colon"
        source_split = (
            "official_train"
            if dataset == "NCT-CRC-HE-100K"
            else "official_validation"
        )
    elif dataset == "LC25000":
        organ = "lung"
        source_split = next(
            (part for part in parts if part in {"train", "val", "test"}),
            "unknown",
        )
    else:
        organ = "unknown"
        source_split = "unknown"

    relative_name = relative_path.as_posix()
    image_id = "IMG_" + hashlib.sha256(relative_name.encode("utf-8")).hexdigest()[:16]
    return {
        "image_id": image_id,
        "dataset": dataset,
        "filename": relative_name,
        "organ": organ,
        "tissue_class": tissue_class,
        "source_split": source_split,
    }


def load_mapping(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}

    mapping: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "patient_id", "slide_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"mapping is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            filename = (row["filename"] or "").replace("\\", "/").strip()
            patient_id = (row["patient_id"] or "").strip()
            slide_id = (row["slide_id"] or "").strip()
            if not filename or not patient_id or not slide_id:
                raise ValueError(f"mapping line {line_number} contains an empty field")
            if filename in mapping:
                raise ValueError(f"duplicate filename in mapping: {filename}")
            mapping[filename] = (patient_id, slide_id)
    return mapping


def load_lc25000_groups(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    groups: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "group_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"LC25000 group file is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            filename = (row["filename"] or "").strip().lower()
            raw_group_id = (row["group_id"] or "").strip()
            if not filename or not raw_group_id:
                raise ValueError(f"LC25000 group line {line_number} contains an empty field")
            group_id = f"LC25000_TILE_{int(raw_group_id):04d}"
            previous = groups.get(filename)
            if previous is not None and previous != group_id:
                raise ValueError(f"conflicting LC25000 group for {filename}")
            groups[filename] = group_id
    return groups


def build_inventory(
    source: Path,
    mapping_path: Path | None,
    lc25000_groups_path: Path | None,
) -> list[dict[str, str]]:
    mapping = load_mapping(mapping_path)
    lc25000_groups = load_lc25000_groups(lc25000_groups_path)
    rows: list[dict[str, str]] = []
    used_mapping_keys: set[str] = set()
    missing_lc25000_groups: list[str] = []

    for image_path in iter_images(source):
        relative_path = image_path.relative_to(source)
        row = infer_fields(relative_path)
        identity = mapping.get(row["filename"])
        if identity:
            row["patient_id"], row["slide_id"] = identity
            row["identity_status"] = "mapped"
            used_mapping_keys.add(row["filename"])
        else:
            row["patient_id"] = ""
            row["slide_id"] = ""
            row["identity_status"] = "missing_source_mapping"

        if row["dataset"] == "LC25000" and lc25000_groups:
            group_id = lc25000_groups.get(relative_path.name.lower())
            if group_id is None:
                missing_lc25000_groups.append(row["filename"])
                row["group_id"] = ""
                row["identity_type"] = "unavailable"
            else:
                row["group_id"] = group_id
                row["identity_type"] = "source_tile"
        else:
            row["group_id"] = ""
            row["identity_type"] = "patient" if identity else "unavailable"
        rows.append(row)

    if mapping:
        unused = set(mapping).difference(used_mapping_keys)
        missing = [row["filename"] for row in rows if not row["patient_id"]]
        if unused:
            example = sorted(unused)[:3]
            raise ValueError(f"mapping references {len(unused)} unknown files; examples: {example}")
        if missing:
            example = missing[:3]
            raise ValueError(f"mapping omits {len(missing)} images; examples: {example}")

    if missing_lc25000_groups:
        raise ValueError(
            f"LC25000 group metadata omits {len(missing_lc25000_groups)} local images; "
            f"examples: {missing_lc25000_groups[:3]}"
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("AI病理"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/slides.csv"))
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Private, anonymized CSV with filename,patient_id,slide_id columns.",
    )
    parser.add_argument(
        "--lc25000-groups",
        type=Path,
        help="Public LC25000-clean CSV that maps augmented images to source tiles.",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"source directory does not exist: {args.source}")

    rows = build_inventory(args.source, args.mapping, args.lc25000_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "slide_id",
        "filename",
        "organ",
        "image_id",
        "dataset",
        "tissue_class",
        "source_split",
        "identity_status",
        "group_id",
        "identity_type",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mapped = sum(bool(row["patient_id"]) for row in rows)
    grouped = sum(bool(row["group_id"]) for row in rows)
    print(f"wrote {len(rows)} image records to {args.output}")
    print(f"patient-mapped records: {mapped}/{len(rows)}")
    print(f"source-grouped records: {grouped}/{len(rows)}")
    for dataset, count in sorted(Counter(row["dataset"] for row in rows).items()):
        print(f"dataset {dataset}: {count} images")
    for organ, count in sorted(Counter(row["organ"] for row in rows).items()):
        print(f"organ {organ}: {count} images")
    if mapped != len(rows):
        print(
            "strict patient-level split generation is intentionally blocked until a "
            "complete, anonymized patient/slide mapping is supplied"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
