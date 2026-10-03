"""Create a leakage-aware split from the strongest identities currently available.

This is deliberately not called a patient-level split. NCT/CRC patient-to-patch
metadata is unavailable, while LC25000 exposes reconstructed source-tile groups.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
DATASET_SPLITS = {
    "NCT-CRC-HE-100K": "train",
    "CRC-VAL-HE-7K": "test",
}


def allocate_counts(total: int) -> dict[str, int]:
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    counts = {name: int(total * ratio) for name, ratio in ratios.items()}
    if total >= 3:
        for name in SPLITS:
            counts[name] = max(1, counts[name])
    while sum(counts.values()) > total:
        candidates = [name for name in SPLITS if counts[name] > (1 if total >= 3 else 0)]
        counts[max(candidates, key=lambda name: counts[name])] -= 1
    while sum(counts.values()) < total:
        gaps = {name: total * ratios[name] - counts[name] for name in SPLITS}
        counts[max(SPLITS, key=lambda name: gaps[name])] += 1
    return counts


def read_slides(path: Path) -> list[dict[str, str]]:
    required = {
        "patient_id",
        "group_id",
        "identity_type",
        "image_id",
        "filename",
        "dataset",
        "organ",
        "tissue_class",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"slides metadata is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("slides metadata is empty")
    return rows


def assign_lc25000_groups(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    class_groups: dict[str, set[str]] = defaultdict(set)
    group_classes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["dataset"] != "LC25000":
            continue
        group_id = row["group_id"]
        if not group_id or row["identity_type"] != "source_tile":
            raise ValueError(f"LC25000 image lacks a verified source-tile group: {row['filename']}")
        class_groups[row["tissue_class"]].add(group_id)
        group_classes[group_id].add(row["tissue_class"])

    cross_class = sorted(group for group, classes in group_classes.items() if len(classes) > 1)
    if cross_class:
        raise ValueError(f"LC25000 groups cross tissue classes: {cross_class[:3]}")

    assignments: dict[str, str] = {}
    for class_index, tissue_class in enumerate(sorted(class_groups)):
        groups = sorted(class_groups[tissue_class])
        random.Random(seed + class_index).shuffle(groups)
        counts = allocate_counts(len(groups))
        cursor = 0
        for split in SPLITS:
            for group_id in groups[cursor : cursor + counts[split]]:
                assignments[group_id] = split
            cursor += counts[split]
    return assignments


def create_assignments(
    rows: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    lc_assignments = assign_lc25000_groups(rows, seed)
    unit_rows: dict[str, dict[str, str]] = {}
    image_rows: list[dict[str, str]] = []

    for row in rows:
        dataset = row["dataset"]
        if dataset == "LC25000":
            group_id = row["group_id"]
            split = lc_assignments[group_id]
            identity_type = "source_tile"
            unit_id = group_id
        elif dataset in DATASET_SPLITS:
            split = DATASET_SPLITS[dataset]
            identity_type = "official_cohort"
            unit_id = f"{dataset}_COHORT"
            group_id = unit_id
        else:
            raise ValueError(f"no leakage-safe policy exists for dataset: {dataset}")

        unit_rows.setdefault(
            unit_id,
            {
                "patient_id": "",
                "group_id": group_id,
                "split": split,
                "identity_type": identity_type,
                "organ": row["organ"],
                "dataset": dataset,
            },
        )
        image_rows.append(
            {
                "image_id": row["image_id"],
                "filename": row["filename"],
                "patient_id": row["patient_id"],
                "group_id": group_id,
                "split": split,
                "identity_type": identity_type,
                "organ": row["organ"],
                "dataset": dataset,
                "tissue_class": row["tissue_class"],
            }
        )

    observed_splits: dict[str, set[str]] = defaultdict(set)
    for row in image_rows:
        observed_splits[row["group_id"]].add(row["split"])
    leaked = sorted(group for group, splits in observed_splits.items() if len(splits) != 1)
    if leaked:
        raise ValueError(f"identity groups cross splits: {leaked[:3]}")

    return sorted(unit_rows.values(), key=lambda row: row["group_id"]), image_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", type=Path, default=Path("data/metadata/slides.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/splits.csv"))
    parser.add_argument(
        "--image-output", type=Path, default=Path("data/metadata/image_splits.csv")
    )
    parser.add_argument(
        "--summary-output", type=Path, default=Path("data/metadata/split_summary.csv")
    )
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    slides = read_slides(args.slides)
    units, images = create_assignments(slides, args.seed)
    write_csv(
        args.output,
        units,
        ["patient_id", "group_id", "split", "identity_type", "organ", "dataset"],
    )
    write_csv(
        args.image_output,
        images,
        [
            "image_id",
            "filename",
            "patient_id",
            "group_id",
            "split",
            "identity_type",
            "organ",
            "dataset",
            "tissue_class",
        ],
    )

    unit_summary = Counter(
        (row["dataset"], row["organ"], row["split"], row["identity_type"])
        for row in units
    )
    image_summary = Counter(
        (row["dataset"], row["organ"], row["split"], row["identity_type"])
        for row in images
    )
    summary_rows = []
    for key in sorted(image_summary):
        dataset, organ, split, identity_type = key
        summary_rows.append(
            {
                "dataset": dataset,
                "organ": organ,
                "split": split,
                "identity_type": identity_type,
                "units": str(unit_summary[key]),
                "images": str(image_summary[key]),
                "verified_patient_level": "yes" if identity_type == "patient" else "no",
            }
        )
    write_csv(
        args.summary_output,
        summary_rows,
        [
            "dataset",
            "organ",
            "split",
            "identity_type",
            "units",
            "images",
            "verified_patient_level",
        ],
    )

    unit_counts = Counter(row["split"] for row in units)
    image_counts = Counter(row["split"] for row in images)
    patient_count = len({row["patient_id"] for row in images if row["patient_id"]})
    print(f"verified patient identities: {patient_count}")
    for split in SPLITS:
        print(f"{split}: units={unit_counts[split]}, images={image_counts[split]}")
    print("warning: units are source tiles or official cohorts, not verified patients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
