"""Create leakage-aware train/validation/test manifests for the 4,000-image set."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "colon_cross_organ_dataset_4000"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=20261006)
    return parser.parse_args()


def choose_case_split(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """Assign 28 donor cases as 20/4/4 while matching 70/15/15 image counts."""
    counts = Counter(row["donor_case_id"] for row in rows)
    cases = sorted(counts)
    if len(cases) < 6:
        raise ValueError("too few donor cases for three-way isolation")
    # The current data have 28 cases per organ. Keep the intended 20/4/4 case
    # allocation; the randomized search finds the closest image-count balance.
    val_cases = max(1, round(len(cases) * 0.15))
    test_cases = max(1, round(len(cases) * 0.15))
    train_cases = len(cases) - val_cases - test_cases
    target = {"train": len(rows) * 0.70, "val": len(rows) * 0.15, "test": len(rows) * 0.15}
    rng = random.Random(seed)
    best: tuple[float, list[str]] | None = None
    for _ in range(100_000):
        order = cases.copy()
        rng.shuffle(order)
        groups = {
            "train": order[:train_cases],
            "val": order[train_cases : train_cases + val_cases],
            "test": order[train_cases + val_cases :],
        }
        score = sum(
            (sum(counts[case] for case in group) - target[split]) ** 2
            for split, group in groups.items()
        )
        if best is None or score < best[0]:
            best = (score, order)
            if score == 0:
                break
    assert best is not None
    order = best[1]
    assignment = {}
    for case in order[:train_cases]:
        assignment[case] = "train"
    for case in order[train_cases : train_cases + val_cases]:
        assignment[case] = "val"
    for case in order[train_cases + val_cases :]:
        assignment[case] = "test"
    return assignment


def mask_box(path: Path) -> tuple[str, str, str, str]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    points = cv2.findNonZero((mask > 0).astype("uint8"))
    if points is None:
        return "", "", "", ""
    x, y, width, height = cv2.boundingRect(points)
    return str(x), str(y), str(width), str(height)


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    with (root / "metadata.csv").open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 4000:
        raise ValueError(f"expected 4,000 rows, found {len(rows)}")

    positives = [row for row in rows if row["label"] == "1"]
    negatives = [row for row in rows if row["label"] == "0"]
    assignments: dict[tuple[str, str], str] = {}
    for organ in ("lung", "kidney"):
        organ_rows = [row for row in positives if row["donor_organ"] == organ]
        for case, split in choose_case_split(organ_rows, args.seed + len(organ)) .items():
            assignments[(organ, case)] = split
    for row in positives:
        row["split"] = assignments[(row["donor_organ"], row["donor_case_id"])]

    positive_counts = Counter(row["split"] for row in positives)
    # Match negative counts to positives exactly, with round-robin tissue-class
    # stratification and no recipient reuse.
    rng = random.Random(args.seed)
    negatives_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in negatives:
        negatives_by_class[row["recipient_tissue_class"]].append(row)
    for values in negatives_by_class.values():
        rng.shuffle(values)
    ordered_negatives: list[dict[str, str]] = []
    offsets = Counter()
    while len(ordered_negatives) < len(negatives):
        tissue_class = sorted(negatives_by_class)[len(ordered_negatives) % len(negatives_by_class)]
        ordered_negatives.append(negatives_by_class[tissue_class][offsets[tissue_class]])
        offsets[tissue_class] += 1
    rng.shuffle(ordered_negatives)
    cursor = 0
    for split in SPLITS:
        count = positive_counts[split]
        for row in ordered_negatives[cursor : cursor + count]:
            row["split"] = split
        cursor += count
    if cursor != len(negatives):
        raise RuntimeError("negative allocation did not consume every image")

    rows.sort(key=lambda row: row["image_id"])
    split_dir = root / "splits"
    split_dir.mkdir(exist_ok=True)
    fields = list(rows[0])
    if "split" not in fields:
        fields.append("split")
    for split in SPLITS:
        with (split_dir / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)
    with (root / "splits.csv").open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerows({"image_id": row["image_id"], "split": row["split"]} for row in rows)

    manifest_fields = [
        "sample_id", "cohort", "split", "label", "image_path", "paste_mask_path",
        "anomaly_mask_path", "filename", "variant", "host_organ", "host_case_id",
        "host_slide_id", "host_original_path", "donor_organ", "donor_case_id",
        "donor_slide_id", "donor_original_path", "target_area_fraction",
        "actual_area_fraction", "box_x", "box_y", "box_width", "box_height",
        "mask_filename", "source_split",
    ]
    manifest_rows = []
    for row in rows:
        label_name = "positive" if row["label"] == "1" else "negative"
        image_path = Path("images") / label_name / f"{row['image_id']}.jpg"
        mask_path = Path("masks") / label_name / f"{row['image_id']}.png"
        box = mask_box(root / mask_path) if row["label"] == "1" else ("", "", "", "")
        manifest_rows.append(
            {
                "sample_id": row["image_id"],
                "cohort": "colon_cross_organ_4000",
                "split": "historical_test" if row["split"] == "test" else row["split"],
                "label": row["label"],
                "image_path": image_path.as_posix(),
                "paste_mask_path": mask_path.as_posix() if row["label"] == "1" else "",
                "anomaly_mask_path": mask_path.as_posix() if row["label"] == "1" else "",
                "filename": image_path.name,
                "variant": "cross_organ_other_patient" if row["label"] == "1" else "clean",
                "host_organ": "colon",
                "host_case_id": Path(row["recipient_path"]).stem,
                "host_slide_id": Path(row["recipient_path"]).stem,
                "host_original_path": row["recipient_path"],
                "donor_organ": row["donor_organ"],
                "donor_case_id": row["donor_case_id"],
                "donor_slide_id": row["donor_slide_id"],
                "donor_original_path": row["donor_path"],
                "target_area_fraction": row["target_area_ratio"],
                "actual_area_fraction": row["actual_area_ratio"],
                "box_x": box[0], "box_y": box[1], "box_width": box[2], "box_height": box[3],
                "mask_filename": mask_path.name if row["label"] == "1" else "",
                "source_split": row["split"],
            }
        )
    with (root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "seed": args.seed,
        "policy": "70/15/15 target; donor-case-disjoint positives; unique but not patient-grouped NCT hosts",
        "counts": {
            split: {
                "total": sum(row["split"] == split for row in rows),
                "positive": sum(row["split"] == split and row["label"] == "1" for row in rows),
                "negative": sum(row["split"] == split and row["label"] == "0" for row in rows),
                "lung": sum(row["split"] == split and row["donor_organ"] == "lung" for row in rows),
                "kidney": sum(row["split"] == split and row["donor_organ"] == "kidney" for row in rows),
            }
            for split in SPLITS
        },
        "donor_cases": {
            split: {
                organ: sorted({row["donor_case_id"] for row in positives if row["split"] == split and row["donor_organ"] == organ})
                for organ in ("lung", "kidney")
            }
            for split in SPLITS
        },
        "host_overlap": {
            f"{first}-{second}": len(
                {Path(row["recipient_path"]).name for row in rows if row["split"] == first}
                & {Path(row["recipient_path"]).name for row in rows if row["split"] == second}
            )
            for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "limitation": "NCT-CRC-HE-100K does not provide patient IDs, so recipient patient-level isolation cannot be verified.",
    }
    (root / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
