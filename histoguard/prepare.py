"""Fixed HistoGuard data and evaluation harness.

Training experiments may change ``train.py``. Split definitions, label
semantics, metrics, and test access remain fixed here.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as vision_f
from torchvision.transforms.functional import InterpolationMode


MAIN_DATA_DEFAULT = Path(
    r"C:\Users\ASUS\Desktop\HistoGuard_contamination_20261004"
    r"\HistoGuard_contamination_20261004"
)
AUX_DATA_DEFAULT = Path(
    r"C:\Users\ASUS\Desktop\HistoGuard_PanNuke_auxiliary_20261004"
    r"\HistoGuard_PanNuke_auxiliary_20261004"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
POSITIVE_VARIANTS = {"cross_organ_other_patient", "same_organ_other_patient"}


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 384
    batch_size: int = 24
    workers: int = 4
    augmentation: str = "basic"
    paired_sham: bool = False


def read_manifest(root: Path, split: str) -> list[dict[str, str]]:
    if split not in {"train", "val", "historical_test"}:
        raise ValueError(f"unsupported split: {split}")
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == split]
    if not rows:
        raise ValueError(f"no records for split={split}")
    for row in rows:
        label = int(row["label"])
        if label != int(row["variant"] in POSITIVE_VARIANTS):
            raise ValueError(f"inconsistent target: {row['sample_id']}")
    return rows


def _pair_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["cohort"],
        row["host_case_id"],
        row["host_organ"],
        row["target_area_fraction"],
    )


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _load_mask(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None:
        return Image.new("L", size, color=0)
    with Image.open(path) as opened:
        return opened.convert("L")


def _same_image_paste(image: torch.Tensor, rng: random.Random) -> torch.Tensor:
    _, height, width = image.shape
    patch_h = rng.randint(max(12, height // 18), max(16, height // 6))
    patch_w = rng.randint(max(12, width // 18), max(16, width // 6))
    sy, sx = rng.randint(0, height - patch_h), rng.randint(0, width - patch_w)
    ty, tx = rng.randint(0, height - patch_h), rng.randint(0, width - patch_w)
    patch = image[:, sy : sy + patch_h, sx : sx + patch_w].clone()
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, patch_h),
        torch.linspace(-1.0, 1.0, patch_w),
        indexing="ij",
    )
    region = xx.square() + yy.square() <= 1.0
    destination = image[:, ty : ty + patch_h, tx : tx + patch_w]
    destination[:, region] = patch[:, region]
    return image


def _apply_common_pil_augmentation(
    images: list[Image.Image], mask: Image.Image, robust: bool
) -> tuple[list[Image.Image], Image.Image]:
    if random.random() < 0.5:
        images = [vision_f.hflip(image) for image in images]
        mask = vision_f.hflip(mask)
    if random.random() < 0.5:
        images = [vision_f.vflip(image) for image in images]
        mask = vision_f.vflip(mask)
    turns = random.randint(0, 3)
    if turns:
        angle = 90 * turns
        images = [vision_f.rotate(image, angle) for image in images]
        mask = vision_f.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
    if robust:
        brightness = random.uniform(0.80, 1.20)
        contrast = random.uniform(0.80, 1.20)
        saturation = random.uniform(0.85, 1.15)
        images = [
            vision_f.adjust_saturation(
                vision_f.adjust_contrast(vision_f.adjust_brightness(image, brightness), contrast),
                saturation,
            )
            for image in images
        ]
        if random.random() < 0.30:
            sigma = random.uniform(0.25, 1.6)
            images = [vision_f.gaussian_blur(image, [5, 5], [sigma, sigma]) for image in images]
    return images, mask


class HistoGuardDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        split: str,
        config: DataConfig,
        *,
        training: bool,
    ) -> None:
        self.root = root.resolve()
        self.rows = read_manifest(self.root, split)
        self.config = config
        self.training = training
        sham_paths = {
            _pair_key(row): self.root / row["image_path"]
            for row in self.rows
            if row["variant"] == "same_patient_sham"
        }
        self.pair_paths: list[Path | None] = []
        for row in self.rows:
            pair_path = sham_paths.get(_pair_key(row)) if int(row["label"]) else None
            self.pair_paths.append(pair_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = _load_rgb(self.root / row["image_path"])
        mask_path = self.root / row["anomaly_mask_path"] if row["anomaly_mask_path"] else None
        mask = _load_mask(mask_path, image.size)
        pair_path = self.pair_paths[index] if self.config.paired_sham else None
        pair = _load_rgb(pair_path) if pair_path is not None else Image.new("RGB", image.size)
        size = [self.config.image_size, self.config.image_size]
        image = vision_f.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        pair = vision_f.resize(pair, size, interpolation=InterpolationMode.BILINEAR)
        mask = vision_f.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        if self.training:
            robust = self.config.augmentation == "robust"
            images, mask = _apply_common_pil_augmentation([image, pair], mask, robust)
            image, pair = images

        image_tensor = vision_f.pil_to_tensor(image).float().div_(255.0)
        pair_tensor = vision_f.pil_to_tensor(pair).float().div_(255.0)
        mask_tensor = (vision_f.pil_to_tensor(mask).float() > 127).float()
        if self.training and self.config.augmentation == "robust":
            if random.random() < 0.30:
                noise = torch.randn_like(image_tensor) * random.uniform(0.004, 0.020)
                image_tensor = (image_tensor + noise).clamp_(0, 1)
                if pair_path is not None:
                    pair_tensor = (pair_tensor + noise).clamp_(0, 1)
            if random.random() < 0.45:
                image_tensor = _same_image_paste(image_tensor, random)
                if pair_path is not None:
                    pair_tensor = _same_image_paste(pair_tensor, random)
        image_tensor = vision_f.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        pair_tensor = vision_f.normalize(pair_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "pair_image": pair_tensor,
            "has_pair": torch.tensor(pair_path is not None, dtype=torch.bool),
            "sample_id": row["sample_id"],
            "variant": row["variant"],
            "organ": row["host_organ"],
            "area": torch.tensor(float(row["actual_area_fraction"] or 0.0)),
            "target_area": torch.tensor(float(row["target_area_fraction"] or 0.0)),
        }


class PanNukeConsistencyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Unlabeled reference/calibration images; PanNuke test is never used."""

    def __init__(self, root: Path, image_size: int) -> None:
        self.paths = sorted(
            path
            for split in ("reference", "calibration")
            for path in (root / "data" / split).rglob("*.png")
        )
        self.image_size = image_size
        if not self.paths:
            raise ValueError("no PanNuke reference/calibration images")

    def __len__(self) -> int:
        return len(self.paths)

    def _view(self, image: Image.Image) -> torch.Tensor:
        view = vision_f.resize(
            image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR
        )
        if random.random() < 0.5:
            view = vision_f.hflip(view)
        if random.random() < 0.5:
            view = vision_f.vflip(view)
        view = vision_f.adjust_brightness(view, random.uniform(0.70, 1.30))
        view = vision_f.adjust_contrast(view, random.uniform(0.70, 1.30))
        view = vision_f.adjust_saturation(view, random.uniform(0.75, 1.25))
        if random.random() < 0.35:
            view = vision_f.gaussian_blur(view, [5, 5], [0.2, 1.8])
        tensor = vision_f.pil_to_tensor(view).float().div_(255.0)
        return vision_f.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = _load_rgb(self.paths[index])
        return self._view(image), self._view(image)


def build_loaders(
    main_root: Path,
    aux_root: Path,
    config: DataConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": True,
        "persistent_workers": config.workers > 0,
    }
    train = DataLoader(
        HistoGuardDataset(main_root, "train", config, training=True),
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_config = DataConfig(
        image_size=config.image_size,
        batch_size=config.batch_size,
        workers=config.workers,
        augmentation="basic",
        paired_sham=False,
    )
    val = DataLoader(
        HistoGuardDataset(main_root, "val", val_config, training=False),
        shuffle=False,
        drop_last=False,
        **common,
    )
    pan = DataLoader(
        PanNukeConsistencyDataset(aux_root, config.image_size),
        shuffle=True,
        drop_last=True,
        **common,
    )
    return train, val, pan


def build_test_loader(main_root: Path, config: DataConfig) -> DataLoader:
    return DataLoader(
        HistoGuardDataset(main_root, "historical_test", config, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels == 1].sum() / max(1, labels.sum()))


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((labels == 1) & predictions))
    tn = int(np.sum((labels == 0) & (~predictions)))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & (~predictions)))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": 2 * precision * sensitivity / max(1e-12, precision + sensitivity),
    }


def _largest_component_box(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height = [int(value) for value in stats[component, :4]]
    return x, y, x + width, y + height


def _box_iou(first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int] | None) -> float:
    if first is None or second is None:
        return 0.0
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (first[2] - first[0]) * (first[3] - first[1])
    area2 = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1, area1 + area2 - intersection)


@torch.inference_mode()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            output = model(images)
        probabilities = output["logits"].float().sigmoid().cpu().numpy()
        heatmaps = torch_f.interpolate(
            output["mask_logits"].float(),
            size=batch["mask"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).sigmoid().cpu().numpy()
        for index in range(len(probabilities)):
            predictions.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "variant": batch["variant"][index],
                    "organ": batch["organ"][index],
                    "area": float(batch["area"][index]),
                    "target_area": float(batch["target_area"][index]),
                    "label": int(batch["label"][index]),
                    "probability": float(probabilities[index]),
                    "heatmap": heatmaps[index, 0].astype(np.float16),
                    "target": batch["mask"][index, 0].numpy().astype(np.uint8),
                }
            )
    return predictions


def compute_metrics(
    records: list[dict[str, Any]], cls_threshold: float = 0.5, mask_threshold: float = 0.5
) -> dict[str, Any]:
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    scores = np.asarray([record["probability"] for record in records], dtype=np.float64)
    predicted_labels = scores >= cls_threshold
    metrics: dict[str, Any] = {
        "images": len(records),
        "auroc": _binary_auroc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "classification_threshold": cls_threshold,
        "mask_threshold": mask_threshold,
        **_confusion(labels, predicted_labels),
    }
    positive_dice: list[float] = []
    positive_iou: list[float] = []
    box_ious: list[float] = []
    point_hits: list[float] = []
    end_to_end: list[float] = []
    negative_pixel_fraction: list[float] = []
    for record, alarm in zip(records, predicted_labels, strict=True):
        heatmap = record["heatmap"].astype(np.float32)
        target = record["target"].astype(bool)
        binary = heatmap >= mask_threshold
        if record["label"]:
            intersection_soft = float((heatmap * target).sum())
            soft_dice = (2 * intersection_soft + 1.0) / (heatmap.sum() + target.sum() + 1.0)
            intersection = int(np.logical_and(binary, target).sum())
            union = int(np.logical_or(binary, target).sum())
            hard_dice = (2 * intersection) / max(1, int(binary.sum() + target.sum()))
            positive_dice.append(float(soft_dice))
            positive_iou.append(intersection / max(1, union))
            predicted_box = _largest_component_box(binary)
            target_box = _largest_component_box(target)
            box_ious.append(_box_iou(predicted_box, target_box))
            hit = 0.0
            if predicted_box is not None:
                center_x = min(target.shape[1] - 1, (predicted_box[0] + predicted_box[2]) // 2)
                center_y = min(target.shape[0] - 1, (predicted_box[1] + predicted_box[3]) // 2)
                hit = float(target[center_y, center_x])
            point_hits.append(hit)
            end_to_end.append(float(alarm and hit > 0))
        else:
            negative_pixel_fraction.append(float(binary.mean()))
    metrics.update(
        {
            "soft_dice": float(np.mean(positive_dice)),
            "pixel_iou": float(np.mean(positive_iou)),
            "box_iou": float(np.mean(box_ious)),
            "point_hit_rate": float(np.mean(point_hits)),
            "end_to_end_hit_rate": float(np.mean(end_to_end)),
            "negative_mask_fraction": float(np.mean(negative_pixel_fraction)),
        }
    )
    for variant in ("clean", "same_patient_sham"):
        indices = np.asarray([record["variant"] == variant for record in records])
        metrics[f"{variant}_fpr"] = (
            float(predicted_labels[indices].mean()) if indices.any() else 0.0
        )
    for variant in POSITIVE_VARIANTS:
        indices = np.asarray([record["variant"] == variant for record in records])
        metrics[f"{variant}_tpr"] = (
            float(predicted_labels[indices].mean()) if indices.any() else 0.0
        )
    metrics["robust_score"] = (
        0.35 * metrics["auroc"]
        + 0.15 * metrics["balanced_accuracy"]
        + 0.20 * metrics["soft_dice"]
        + 0.10 * metrics["box_iou"]
        + 0.10 * metrics["point_hit_rate"]
        + 0.10 * (1.0 - metrics["same_patient_sham_fpr"])
    )
    return metrics


def calibrate_thresholds(records: list[dict[str, Any]]) -> tuple[float, float, dict[str, Any]]:
    best_cls = max(
        (compute_metrics(records, threshold, 0.5) for threshold in np.linspace(0.05, 0.95, 91)),
        key=lambda item: (item["balanced_accuracy"], item["sensitivity"]),
    )["classification_threshold"]
    candidates = []
    for mask_threshold in np.linspace(0.10, 0.90, 33):
        metrics = compute_metrics(records, best_cls, float(mask_threshold))
        localization_score = (
            0.45 * metrics["soft_dice"]
            + 0.25 * metrics["box_iou"]
            + 0.20 * metrics["point_hit_rate"]
            - 0.10 * metrics["negative_mask_fraction"]
        )
        candidates.append((localization_score, float(mask_threshold), metrics))
    _, best_mask, best_metrics = max(candidates, key=lambda item: item[0])
    return float(best_cls), best_mask, best_metrics


def segmentation_loss(mask_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = torch_f.interpolate(target, size=mask_logits.shape[-2:], mode="nearest")
    bce = torch_f.binary_cross_entropy_with_logits(
        mask_logits, target, pos_weight=torch.tensor([12.0], device=mask_logits.device)
    )
    probabilities = mask_logits.sigmoid().flatten(1)
    flat_target = target.flatten(1)
    intersection = (probabilities * flat_target).sum(1)
    dice = 1.0 - (2 * intersection + 1.0) / (
        probabilities.sum(1) + flat_target.sum(1) + 1.0
    )
    return 0.35 * bce + 0.65 * dice.mean()


def save_prediction_rows(path: Path, records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "sample_id", "variant", "organ", "target_area", "area",
                "label", "probability", "prediction",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: record[key]
                    for key in (
                        "sample_id", "variant", "organ", "target_area", "area",
                        "label", "probability",
                    )
                }
                | {"prediction": int(record["probability"] >= metrics["classification_threshold"])}
            )


__all__ = [
    "AUX_DATA_DEFAULT",
    "MAIN_DATA_DEFAULT",
    "DataConfig",
    "build_loaders",
    "build_test_loader",
    "calibrate_thresholds",
    "collect_predictions",
    "compute_metrics",
    "save_prediction_rows",
    "segmentation_loss",
]
