"""Create and validate deterministic patient-level train/val/test splits."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {"patient_id", "slide_id", "filename", "organ"}
SPLIT_ORDER = ("train", "val", "test")


def read_slides(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"slides metadata is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("slides metadata is empty")

    incomplete = [
        index
        for index, row in enumerate(rows, start=2)
        if not all((row[column] or "").strip() for column in REQUIRED_COLUMNS)
    ]
    if incomplete:
        preview = ", ".join(map(str, incomplete[:10]))
        raise ValueError(
            f"{len(incomplete)} rows lack patient/slide identity; "
            f"first affected CSV lines: {preview}. Refusing image-level splitting."
        )

    filename_counts = Counter(row["filename"] for row in rows)
    duplicates = sorted(name for name, count in filename_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate filenames found; examples: {duplicates[:3]}")

    slide_to_patients: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        slide_to_patients[row["slide_id"]].add(row["patient_id"])
    bad_slides = sorted(slide for slide, patients in slide_to_patients.items() if len(patients) > 1)
    if bad_slides:
        raise ValueError(f"slide_id maps to multiple patients; examples: {bad_slides[:3]}")
    return rows


def split_counts(total: int) -> dict[str, int]:
    if total < 3:
        raise ValueError("at least 3 patients are required for train/val/test")

    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    counts = {name: int(total * ratio) for name, ratio in ratios.items()}
    for name in SPLIT_ORDER:
        if counts[name] == 0:
            counts[name] = 1

    while sum(counts.values()) > total:
        candidates = [name for name in SPLIT_ORDER if counts[name] > 1]
        counts[max(candidates, key=lambda name: counts[name])] -= 1
    while sum(counts.values()) < total:
        ideal_gap = {
            name: total * ratios[name] - counts[name]
            for name in SPLIT_ORDER
        }
        counts[max(SPLIT_ORDER, key=lambda name: ideal_gap[name])] += 1
    return counts


def assign_patients(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    patients = sorted({row["patient_id"] for row in rows})
    random.Random(seed).shuffle(patients)
    counts = split_counts(len(patients))

    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        for patient_id in patients[cursor : cursor + counts[split]]:
            assignment[patient_id] = split
        cursor += counts[split]
    return assignment


def validate(rows: list[dict[str, str]], assignment: dict[str, str]) -> None:
    patients = {row["patient_id"] for row in rows}
    if patients != set(assignment):
        missing = sorted(patients.difference(assignment))
        extra = sorted(set(assignment).difference(patients))
        raise ValueError(f"split coverage mismatch; missing={missing[:3]}, extra={extra[:3]}")

    invalid = sorted(set(assignment.values()).difference(SPLIT_ORDER))
    if invalid:
        raise ValueError(f"unknown split labels: {invalid}")

    slide_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        slide_splits[row["slide_id"]].add(assignment[row["patient_id"]])
    leaked = sorted(slide for slide, splits in slide_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"slides span multiple splits; examples: {leaked[:3]}")


def print_summary(rows: list[dict[str, str]], assignment: dict[str, str]) -> None:
    patient_counts = Counter(assignment.values())
    image_counts = Counter(assignment[row["patient_id"]] for row in rows)
    organ_present = all((row["organ"] or "").strip() for row in rows)
    print(f"total patients: {len(assignment)}")
    for split in SPLIT_ORDER:
        print(
            f"{split}: patients={patient_counts[split]}, "
            f"images={image_counts[split]}"
        )
    print(f"organ information complete: {'yes' if organ_present else 'no'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", type=Path, default=Path("data/metadata/slides.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/splits.csv"))
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()

    rows = read_slides(args.slides)
    assignment = assign_patients(rows, args.seed)
    validate(rows, assignment)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "split"])
        writer.writeheader()
        for patient_id in sorted(assignment):
            writer.writerow({"patient_id": patient_id, "split": assignment[patient_id]})

    print_summary(rows, assignment)
    print(f"wrote validated patient split to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
