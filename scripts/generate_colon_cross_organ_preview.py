"""Generate 20 colon-host previews contaminated only by lung/kidney tissue.

The script reads original NCT colon tiles and original CPTAC lung/kidney tiles.
It never uses an already-synthetic image as a host or donor. Outputs are a
small engineering preview, not a training-ready or clinically validated set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLON_ROOT = Path(r"D:\AI病理\NCT-CRC-HE-100K")
DEFAULT_CPTAC_ROOT = Path(
    r"C:\Users\ASUS\Desktop\HistoGuard_contamination_20261004"
    r"\HistoGuard_contamination_20261004"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "colon_cross_organ_preview_20"
COLON_CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colon-root", type=Path, default=DEFAULT_COLON_ROOT)
    parser.add_argument("--cptac-root", type=Path, default=DEFAULT_CPTAC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tissue_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    near_white = np.all(image >= 245, axis=2)
    tissue = ((saturation > 12) | (value < 220)) & (~near_white) & (value > 20)
    mask = tissue.astype(np.uint8) * 255
    scale = max(3, int(round(min(mask.shape) * 0.008)))
    if scale % 2 == 0:
        scale += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 32:
            cleaned[labels == label] = 255
    return cleaned


def crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("empty fragment mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def rotate_expanded(
    image: np.ndarray, mask: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = max(1, int(math.ceil(height * sine + width * cosine)))
    new_height = max(1, int(math.ceil(height * cosine + width * sine)))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return crop_to_mask(rotated_image, rotated_mask)


def irregular_fragment(
    donor: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    donor_mask = tissue_mask(donor)
    ys, xs = np.where(donor_mask > 0)
    if len(xs) < 1000:
        raise ValueError("donor has insufficient tissue")
    height, width = donor_mask.shape
    for _ in range(100):
        anchor_index = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[anchor_index]), int(ys[anchor_index])
        radius_x = float(rng.uniform(35, 90))
        radius_y = float(rng.uniform(35, 90))
        vertices = int(rng.integers(10, 17))
        angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
        angles += rng.uniform(-0.16, 0.16, vertices)
        radial = rng.uniform(0.68, 1.20, vertices)
        points = np.column_stack(
            (cx + np.cos(angles) * radius_x * radial, cy + np.sin(angles) * radius_y * radial)
        )
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        window = np.zeros_like(donor_mask)
        cv2.fillPoly(window, [points.astype(np.int32)], 255)
        candidate = cv2.bitwise_and(window, donor_mask)
        component_count, labels = cv2.connectedComponents(candidate, connectivity=8)
        if component_count <= 1:
            continue
        component = int(labels[cy, cx])
        if component == 0:
            continue
        candidate = (labels == component).astype(np.uint8) * 255
        if np.count_nonzero(candidate) < 500:
            continue
        fragment, fragment_mask = crop_to_mask(donor, candidate)
        angle = float(rng.uniform(0, 360))
        fragment, fragment_mask = rotate_expanded(fragment, fragment_mask, angle)
        if bool(rng.integers(0, 2)):
            fragment = cv2.flip(fragment, 1)
            fragment_mask = cv2.flip(fragment_mask, 1)
        return fragment, fragment_mask, angle
    raise RuntimeError("failed to select a usable donor fragment")


def resize_to_area(
    fragment: np.ndarray, mask: np.ndarray, target_pixels: int
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(5):
        current_pixels = max(1, int(np.count_nonzero(mask)))
        scale = math.sqrt(target_pixels / current_pixels)
        width = max(3, int(round(mask.shape[1] * scale)))
        height = max(3, int(round(mask.shape[0] * scale)))
        fragment = cv2.resize(fragment, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        if abs(np.count_nonzero(mask) - target_pixels) / target_pixels <= 0.015:
            break
    return crop_to_mask(fragment, mask)


def color_match_lab(
    fragment: np.ndarray,
    fragment_mask: np.ndarray,
    destination: np.ndarray,
    host_tissue: np.ndarray,
    strength: float,
) -> np.ndarray:
    source_lab = cv2.cvtColor(fragment, cv2.COLOR_RGB2LAB).astype(np.float32)
    destination_lab = cv2.cvtColor(destination, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_pixels = fragment_mask > 0
    target_pixels = host_tissue > 0
    if np.count_nonzero(target_pixels) < 32:
        target_pixels = np.ones_like(target_pixels, dtype=bool)
    adjusted = source_lab.copy()
    for channel in range(3):
        source_values = source_lab[:, :, channel][source_pixels]
        target_values = destination_lab[:, :, channel][target_pixels]
        source_mean, source_std = float(source_values.mean()), float(source_values.std() + 1e-6)
        target_mean, target_std = float(target_values.mean()), float(target_values.std() + 1e-6)
        matched = (source_lab[:, :, channel] - source_mean) * (target_std / source_std) + target_mean
        adjusted[:, :, channel] = (1.0 - strength) * source_lab[:, :, channel] + strength * matched
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB)


def choose_position(
    host_tissue: np.ndarray,
    fragment_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int, float]:
    host_height, host_width = host_tissue.shape
    height, width = fragment_mask.shape
    if height > host_height or width > host_width:
        raise ValueError("fragment is larger than host")
    best: tuple[int, int, float] | None = None
    selected = fragment_mask > 0
    selected_count = int(np.count_nonzero(selected))
    for _ in range(500):
        x = int(rng.integers(0, host_width - width + 1))
        y = int(rng.integers(0, host_height - height + 1))
        overlap = float(np.count_nonzero(selected & (host_tissue[y : y + height, x : x + width] > 0))) / selected_count
        if best is None or overlap > best[2]:
            best = (x, y, overlap)
        if overlap >= 0.72:
            return x, y, overlap
    if best is None or best[2] < 0.40:
        raise RuntimeError("could not place fragment on host tissue")
    return best


def composite(
    host: np.ndarray,
    donor: np.ndarray,
    target_ratio: float,
    rng: np.random.Generator,
    blur_sigma: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    host_height, host_width = host.shape[:2]
    target_pixels = int(round(host_height * host_width * target_ratio))
    fragment, fragment_mask, rotation = irregular_fragment(donor, rng)
    fragment, fragment_mask = resize_to_area(fragment, fragment_mask, target_pixels)
    x, y, tissue_overlap = choose_position(tissue_mask(host), fragment_mask, rng)
    height, width = fragment_mask.shape
    destination = host[y : y + height, x : x + width]
    destination_tissue = tissue_mask(host)[y : y + height, x : x + width]
    color_strength = float(rng.uniform(0.55, 0.75))
    fragment = color_match_lab(
        fragment, fragment_mask, destination, destination_tissue, color_strength
    )
    feather_sigma = float(rng.uniform(0.80, 1.45))
    fragment = cv2.GaussianBlur(fragment, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    alpha = cv2.GaussianBlur(
        (fragment_mask.astype(np.float32) / 255.0),
        (0, 0),
        sigmaX=feather_sigma,
        sigmaY=feather_sigma,
    )
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    output = host.copy()
    blended = fragment.astype(np.float32) * alpha + destination.astype(np.float32) * (1.0 - alpha)
    output[y : y + height, x : x + width] = np.clip(blended, 0, 255).astype(np.uint8)
    mask = np.zeros((host_height, host_width), dtype=np.uint8)
    mask[y : y + height, x : x + width][fragment_mask > 0] = 255
    actual_pixels = int(np.count_nonzero(mask))
    return output, mask, {
        "target_pixels": target_pixels,
        "actual_pixels": actual_pixels,
        "actual_area_ratio": actual_pixels / (host_height * host_width),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": width,
        "bbox_height": height,
        "rotation_degrees": rotation,
        "blur_sigma": blur_sigma,
        "feather_sigma": feather_sigma,
        "color_match_strength": color_strength,
        "host_tissue_overlap": tissue_overlap,
    }


def collect_colon_hosts(root: Path, rng: random.Random) -> list[Path]:
    by_class: dict[str, list[Path]] = {}
    for class_name in COLON_CLASSES:
        candidates = sorted(root.rglob(f"{class_name}-*.tif"))
        if not candidates:
            candidates = sorted(path for path in root.rglob("*.tif") if path.parent.name == class_name)
        rng.shuffle(candidates)
        by_class[class_name] = candidates
    selected: list[Path] = []
    offsets = {name: 0 for name in COLON_CLASSES}
    for index in range(20):
        class_name = COLON_CLASSES[index % len(COLON_CLASSES)]
        selected.append(by_class[class_name][offsets[class_name]])
        offsets[class_name] += 1
    return selected


def collect_donors(root: Path, rng: random.Random) -> dict[str, list[dict[str, str]]]:
    provenance = root / "provenance" / "source_tiles.csv"
    with provenance.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    donors: dict[str, list[dict[str, str]]] = {"lung": [], "kidney": []}
    for row in rows:
        organ = row["organ"].lower()
        path = root / Path(row["path"])
        if organ in donors and path.is_file():
            row = dict(row)
            row["absolute_path"] = str(path)
            donors[organ].append(row)
    for values in donors.values():
        rng.shuffle(values)
        if len(values) < 10:
            raise ValueError("fewer than 10 source donor tiles are available")
    return donors


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32)
    selected = mask > 0
    result[selected] = 0.62 * result[selected] + 0.38 * np.asarray([255, 35, 35])
    return np.clip(result, 0, 255).astype(np.uint8)


def contact_sheet(
    items: list[tuple[np.ndarray, str]], destination: Path, columns: int = 4
) -> None:
    tile_size = 224
    caption_height = 34
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (array, caption) in enumerate(items):
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + caption_height)
        image = Image.fromarray(array).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + tile_size + 3), caption, fill="black")
    sheet.save(destination, quality=95, subsampling=0)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    labels_dir = output_dir / "labels"
    for directory in (images_dir, masks_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    python_rng = random.Random(args.seed)
    hosts = collect_colon_hosts(args.colon_root.resolve(), python_rng)
    donors = collect_donors(args.cptac_root.resolve(), python_rng)
    records: list[dict[str, Any]] = []
    generated_tiles: list[tuple[np.ndarray, str]] = []
    overlay_tiles: list[tuple[np.ndarray, str]] = []
    previous_ratio = 0.0
    # Use the same five mild blur levels four times, shuffled until their
    # correlation with contamination area is negligible. This prevents blur
    # strength from becoming a label/area shortcut in the small 20-image set.
    blur_rng = np.random.default_rng(args.seed + 991)
    blur_values = np.tile(np.linspace(0.55, 1.15, 5), 4)
    area_steps = np.arange(1, 21, dtype=np.float64)
    for _ in range(10_000):
        blur_rng.shuffle(blur_values)
        if abs(float(np.corrcoef(area_steps, blur_values)[0, 1])) < 0.05:
            break
    else:
        raise RuntimeError("could not decorrelate blur strength from contamination area")

    for index, host_path in enumerate(hosts, start=1):
        organ = "lung" if index % 2 else "kidney"
        donor_record = donors[organ][(index - 1) // 2]
        donor_path = Path(donor_record["absolute_path"])
        target_ratio = index * 0.005
        sample_seed = args.seed * 100 + index
        rng = np.random.default_rng(sample_seed)
        host = load_rgb(host_path)
        donor = load_rgb(donor_path)
        contaminated, mask, details = composite(
            host, donor, target_ratio, rng, float(blur_values[index - 1])
        )
        if details["actual_area_ratio"] <= previous_ratio:
            raise RuntimeError("actual contamination ratios are not strictly increasing")
        previous_ratio = float(details["actual_area_ratio"])
        image_id = f"colon_crossorgan_{index:02d}_{organ}_{int(target_ratio * 10000):04d}bp"
        image_path = images_dir / f"{image_id}.jpg"
        mask_path = masks_dir / f"{image_id}.png"
        label_path = labels_dir / f"{image_id}.txt"
        Image.fromarray(contaminated).save(image_path, quality=95, subsampling=0)
        Image.fromarray(mask).save(mask_path)
        center_x = (details["bbox_x"] + details["bbox_width"] / 2) / host.shape[1]
        center_y = (details["bbox_y"] + details["bbox_height"] / 2) / host.shape[0]
        relative_width = details["bbox_width"] / host.shape[1]
        relative_height = details["bbox_height"] / host.shape[0]
        label_path.write_text(
            f"0 {center_x:.8f} {center_y:.8f} {relative_width:.8f} {relative_height:.8f}\n",
            encoding="utf-8",
        )
        record = {
            "image_id": image_id,
            "label": "contaminated",
            "recipient_organ": "colon",
            "recipient_tissue_class": host_path.parent.name,
            "recipient_path": str(host_path),
            "donor_organ": organ,
            "donor_case_id": donor_record["case_id"],
            "donor_slide_id": donor_record["slide_id"],
            "donor_path": str(donor_path),
            "target_area_ratio": target_ratio,
            "seed": sample_seed,
            **details,
            "output_image": str(image_path),
            "output_mask": str(mask_path),
            "output_label": str(label_path),
        }
        records.append(record)
        caption = f"{index:02d} {organ} {details['actual_area_ratio'] * 100:.2f}%"
        generated_tiles.append((contaminated, caption))
        overlay_tiles.append((overlay(contaminated, mask), caption))

    fieldnames = list(records[0])
    with (output_dir / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    contact_sheet(generated_tiles, output_dir / "contact_sheet.jpg")
    contact_sheet(overlay_tiles, output_dir / "overlay_contact_sheet.jpg")
    summary = {
        "samples": len(records),
        "recipient_organ": "colon",
        "donor_counts": {organ: sum(row["donor_organ"] == organ for row in records) for organ in ("lung", "kidney")},
        "target_area_range": [records[0]["target_area_ratio"], records[-1]["target_area_ratio"]],
        "actual_area_range": [records[0]["actual_area_ratio"], records[-1]["actual_area_ratio"]],
        "strictly_increasing_actual_area": all(
            records[i]["actual_area_ratio"] < records[i + 1]["actual_area_ratio"]
            for i in range(len(records) - 1)
        ),
        "seed": args.seed,
        "output_sha256": {row["image_id"]: sha256(Path(row["output_image"])) for row in records},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Colon cross-organ contamination preview (20 images)\n\n"
        "- Recipients: original NCT-CRC-HE-100K colon tiles.\n"
        "- Donors: original CPTAC lung/kidney tiles, 10 from each organ.\n"
        "- Positives only: cross-organ tissue pasted into colon hosts.\n"
        "- Target areas: 0.5% through 10.0% in 0.5-point increments.\n"
        "- Mild fragment blur, feathered borders, and LAB stain matching are applied.\n"
        "- Research preview only; not clinically validated and not sufficient alone for training.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
