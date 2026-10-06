"""Build a 4,000-image colon contamination classification/detection dataset.

The output contains 2,000 cross-organ positives (1,000 lung and 1,000 kidney)
and 2,000 clean colon negatives.  The existing reviewed 20-image pilot is
copied unchanged and 1,980 additional positives are synthesized.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from generate_colon_cross_organ_preview import (
    COLON_CLASSES,
    composite,
    contact_sheet,
    load_rgb,
    overlay,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLON_ROOT = PROJECT_ROOT / "AI病理" / "NCT-CRC-HE-100K"
DEFAULT_CPTAC_ROOT = (
    Path.home()
    / "Desktop"
    / "HistoGuard_contamination_20261004"
    / "HistoGuard_contamination_20261004"
)
DEFAULT_PILOT_ROOT = PROJECT_ROOT / "runs" / "colon_cross_organ_preview_20"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "colon_cross_organ_dataset_4000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colon-root", type=Path, default=DEFAULT_COLON_ROOT)
    parser.add_argument("--cptac-root", type=Path, default=DEFAULT_CPTAC_ROOT)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser.parse_args()


def collect_hosts(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for tissue_class in COLON_CLASSES:
        paths = sorted(root.rglob(f"{tissue_class}-*.tif"))
        if not paths:
            raise FileNotFoundError(f"no {tissue_class} colon tiles under {root}")
        result[tissue_class] = paths
    return result


def balanced_hosts(
    by_class: dict[str, list[Path]], count: int, excluded_names: set[str], rng: random.Random
) -> list[Path]:
    pools: dict[str, list[Path]] = {}
    for tissue_class, paths in by_class.items():
        candidates = [path for path in paths if path.name not in excluded_names]
        rng.shuffle(candidates)
        pools[tissue_class] = candidates
    selected: list[Path] = []
    offsets = Counter()
    while len(selected) < count:
        tissue_class = COLON_CLASSES[len(selected) % len(COLON_CLASSES)]
        path = pools[tissue_class][offsets[tissue_class]]
        offsets[tissue_class] += 1
        selected.append(path)
        excluded_names.add(path.name)
    rng.shuffle(selected)
    return selected


def load_donors(root: Path, rng: random.Random) -> dict[str, list[dict[str, str]]]:
    provenance = root / "provenance" / "source_tiles.csv"
    with provenance.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    donors: dict[str, list[dict[str, str]]] = {"lung": [], "kidney": []}
    for source_row in rows:
        organ = source_row["organ"].lower()
        source_path = root / source_row["path"]
        if organ in donors and source_path.is_file():
            row = dict(source_row)
            row["absolute_path"] = str(source_path.resolve())
            donors[organ].append(row)
    for organ, rows_for_organ in donors.items():
        if len(rows_for_organ) < 10:
            raise ValueError(f"insufficient {organ} donors: {len(rows_for_organ)}")
        rng.shuffle(rows_for_organ)
    return donors


def contamination_ratios(count: int, rng: np.random.Generator) -> np.ndarray:
    # Deliberately emphasize hard, low-area contamination.
    bins = ((0.005, 0.010, 0.30), (0.010, 0.030, 0.30), (0.030, 0.050, 0.20), (0.050, 0.100, 0.20))
    counts = [int(round(count * weight)) for _, _, weight in bins]
    counts[-1] += count - sum(counts)
    values = np.concatenate(
        [rng.uniform(low, high, bin_count) for (low, high, _), bin_count in zip(bins, counts)]
    )
    rng.shuffle(values)
    return values


def write_label(path: Path, details: dict[str, float], width: int, height: int) -> None:
    center_x = (details["bbox_x"] + details["bbox_width"] / 2) / width
    center_y = (details["bbox_y"] + details["bbox_height"] / 2) / height
    relative_width = details["bbox_width"] / width
    relative_height = details["bbox_height"] / height
    path.write_text(
        f"0 {center_x:.8f} {center_y:.8f} {relative_width:.8f} {relative_height:.8f}\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    positive_images = output / "images" / "positive"
    negative_images = output / "images" / "negative"
    positive_masks = output / "masks" / "positive"
    negative_masks = output / "masks" / "negative"
    positive_labels = output / "labels" / "positive"
    negative_labels = output / "labels" / "negative"
    for directory in (
        positive_images,
        negative_images,
        positive_masks,
        negative_masks,
        positive_labels,
        negative_labels,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    by_class = collect_hosts(args.colon_root.resolve())
    local_by_name = {path.name: path for paths in by_class.values() for path in paths}
    donors = load_donors(args.cptac_root.resolve(), py_rng)

    with (args.pilot_root / "metadata.csv").open("r", encoding="utf-8-sig", newline="") as source:
        pilot_rows = list(csv.DictReader(source))
    if len(pilot_rows) != 20:
        raise ValueError("the reviewed pilot must contain exactly 20 images")

    excluded_names = {Path(row["recipient_path"]).name for row in pilot_rows}
    new_positive_hosts = balanced_hosts(by_class, 1980, excluded_names, py_rng)
    negative_hosts = balanced_hosts(by_class, 2000, excluded_names, py_rng)
    target_ratios = contamination_ratios(1980, np_rng)
    organ_order = ["lung", "kidney"] * 990
    py_rng.shuffle(organ_order)

    records: list[dict[str, object]] = []
    preview_items: list[tuple[np.ndarray, str]] = []
    overlay_items: list[tuple[np.ndarray, str]] = []

    # Preserve the already-reviewed pilot exactly as positive samples 1-20.
    for index, pilot in enumerate(pilot_rows, start=1):
        image_id = f"positive_{index:04d}"
        source_image = Path(pilot["output_image"])
        source_mask = Path(pilot["output_mask"])
        source_label = Path(pilot["output_label"])
        image_path = positive_images / f"{image_id}.jpg"
        mask_path = positive_masks / f"{image_id}.png"
        label_path = positive_labels / f"{image_id}.txt"
        shutil.copy2(source_image, image_path)
        shutil.copy2(source_mask, mask_path)
        shutil.copy2(source_label, label_path)
        local_host = local_by_name.get(Path(pilot["recipient_path"]).name)
        records.append(
            {
                "image_id": image_id,
                "label": 1,
                "label_name": "contaminated",
                "recipient_organ": "colon",
                "recipient_tissue_class": pilot["recipient_tissue_class"],
                "recipient_path": str(local_host or pilot["recipient_path"]),
                "donor_organ": pilot["donor_organ"],
                "donor_case_id": pilot["donor_case_id"],
                "donor_slide_id": pilot["donor_slide_id"],
                "donor_path": pilot["donor_path"],
                "target_area_ratio": float(pilot["target_area_ratio"]),
                "actual_area_ratio": float(pilot["actual_area_ratio"]),
                "blur_sigma": float(pilot["blur_sigma"]),
                "seed": int(pilot["seed"]),
                "source_set": "reviewed_pilot_20",
                "output_image": str(image_path),
                "output_mask": str(mask_path),
                "output_label": str(label_path),
            }
        )

    for offset, (host_path, organ, target_ratio) in enumerate(
        zip(new_positive_hosts, organ_order, target_ratios), start=21
    ):
        image_id = f"positive_{offset:04d}"
        host = load_rgb(host_path)
        last_error: Exception | None = None
        for retry in range(30):
            donor_index = (offset + retry) % len(donors[organ])
            donor_record = donors[organ][donor_index]
            donor_path = Path(donor_record["absolute_path"])
            sample_seed = args.seed * 100_000 + offset * 100 + retry
            sample_rng = np.random.default_rng(sample_seed)
            blur_sigma = float(sample_rng.uniform(0.55, 1.15))
            try:
                contaminated, mask, details = composite(
                    host, load_rgb(donor_path), float(target_ratio), sample_rng, blur_sigma
                )
                break
            except (RuntimeError, ValueError) as error:
                last_error = error
        else:
            raise RuntimeError(f"failed to generate {image_id}: {last_error}")

        image_path = positive_images / f"{image_id}.jpg"
        mask_path = positive_masks / f"{image_id}.png"
        label_path = positive_labels / f"{image_id}.txt"
        Image.fromarray(contaminated).save(image_path, quality=95, subsampling=0)
        Image.fromarray(mask).save(mask_path)
        write_label(label_path, details, host.shape[1], host.shape[0])
        records.append(
            {
                "image_id": image_id,
                "label": 1,
                "label_name": "contaminated",
                "recipient_organ": "colon",
                "recipient_tissue_class": host_path.parent.name,
                "recipient_path": str(host_path.resolve()),
                "donor_organ": organ,
                "donor_case_id": donor_record["case_id"],
                "donor_slide_id": donor_record["slide_id"],
                "donor_path": str(donor_path),
                "target_area_ratio": float(target_ratio),
                "actual_area_ratio": float(details["actual_area_ratio"]),
                "blur_sigma": blur_sigma,
                "seed": sample_seed,
                "source_set": "generated_expansion_1980",
                "output_image": str(image_path),
                "output_mask": str(mask_path),
                "output_label": str(label_path),
            }
        )
        if offset % 250 == 0:
            print(f"generated {offset}/2000 positives", flush=True)
        if len(preview_items) < 48 and offset % 41 == 0:
            caption = f"{offset} {organ} {details['actual_area_ratio'] * 100:.1f}%"
            preview_items.append((contaminated, caption))
            overlay_items.append((overlay(contaminated, mask), caption))

    for index, host_path in enumerate(negative_hosts, start=1):
        image_id = f"negative_{index:04d}"
        image = load_rgb(host_path)
        image_path = negative_images / f"{image_id}.jpg"
        mask_path = negative_masks / f"{image_id}.png"
        label_path = negative_labels / f"{image_id}.txt"
        Image.fromarray(image).save(image_path, quality=95, subsampling=0)
        Image.fromarray(np.zeros(image.shape[:2], dtype=np.uint8)).save(mask_path)
        label_path.write_text("", encoding="utf-8")
        records.append(
            {
                "image_id": image_id,
                "label": 0,
                "label_name": "clean",
                "recipient_organ": "colon",
                "recipient_tissue_class": host_path.parent.name,
                "recipient_path": str(host_path.resolve()),
                "donor_organ": "",
                "donor_case_id": "",
                "donor_slide_id": "",
                "donor_path": "",
                "target_area_ratio": 0.0,
                "actual_area_ratio": 0.0,
                "blur_sigma": 0.0,
                "seed": args.seed,
                "source_set": "clean_negative_2000",
                "output_image": str(image_path),
                "output_mask": str(mask_path),
                "output_label": str(label_path),
            }
        )

    with (output / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    contact_sheet(preview_items, output / "positive_contact_sheet.jpg", columns=6)
    contact_sheet(overlay_items, output / "positive_overlay_contact_sheet.jpg", columns=6)
    positive_records = [record for record in records if record["label"] == 1]
    summary = {
        "total_images": len(records),
        "positive_images": len(positive_records),
        "negative_images": sum(record["label"] == 0 for record in records),
        "positive_donor_counts": dict(Counter(record["donor_organ"] for record in positive_records)),
        "unique_recipient_images": len({Path(str(record["recipient_path"])).name for record in records}),
        "unique_donor_cases": {
            organ: len({record["donor_case_id"] for record in positive_records if record["donor_organ"] == organ})
            for organ in ("lung", "kidney")
        },
        "actual_contamination_range": [
            min(float(record["actual_area_ratio"]) for record in positive_records),
            max(float(record["actual_area_ratio"]) for record in positive_records),
        ],
        "seed": args.seed,
        "metadata_sha256": sha256(output / "metadata.csv"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Colon cross-organ contamination dataset (4,000 images)\n\n"
        "- 2,000 positives: 1,000 lung-to-colon and 1,000 kidney-to-colon.\n"
        "- 2,000 negatives: untouched original colon tiles, re-encoded with the same JPEG settings.\n"
        "- The reviewed 20-image pilot is preserved; 1,980 positives are newly generated.\n"
        "- Every output uses a different colon recipient image.\n"
        "- Small contamination is deliberately overrepresented.\n"
        "- Research data only; not clinically validated.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
