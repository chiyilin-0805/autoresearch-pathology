"""Calibrate on val, freeze thresholds, evaluate historical_test once, and visualize."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from prepare import (
    MAIN_DATA_DEFAULT,
    DataConfig,
    HistoGuardDataset,
    _largest_component_box,
    calibrate_thresholds,
    collect_predictions,
    compute_metrics,
    save_prediction_rows,
)
from train import HistoGuardNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, default=MAIN_DATA_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=Path("results/histoguard/final"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="calibrate and package a candidate without reading historical_test",
    )
    return parser.parse_args()


def loader(root: Path, split: str, config: DataConfig) -> DataLoader:
    return DataLoader(
        HistoGuardDataset(root, split, config, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HistoGuardNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment = checkpoint["experiment"]
    model = HistoGuardNet(
        fpn=bool(experiment["fpn"]),
        localized_evidence=bool(experiment["localized_evidence"]),
        backbone_name=experiment.get("backbone", "resnet18"),
        classification_mode=experiment.get("classification_mode", "legacy"),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device, memory_format=torch.channels_last).eval(), checkpoint


def positive_strata(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record["label"]:
            continue
        area_name = f"{int(round(record['target_area'] * 100)):02d}pct"
        groups[f"area/{area_name}"].append(record)
        groups[f"organ/{record['organ']}"].append(record)
        groups[f"variant/{record['variant']}"].append(record)
    for name, subset in sorted(groups.items()):
        alarms = np.asarray([record["probability"] >= threshold for record in subset])
        output[name] = {
            "n": len(subset),
            "tpr": float(alarms.mean()),
            "fnr": float(1.0 - alarms.mean()),
        }
    return output


def _safe_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__")


def make_visualizations(
    root: Path,
    records: list[dict[str, Any]],
    mask_threshold: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = {}
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["split"] == "historical_test":
                rows_by_id[row["sample_id"]] = row
    contact_rows = []
    selected_keys: set[tuple[str, str, int]] = set()
    for record in records:
        area = int(round(record["target_area"] * 100))
        key = (record["organ"], record["variant"], area)
        should_select = key not in selected_keys and len(contact_rows) < 16
        if should_select:
            selected_keys.add(key)
        row = rows_by_id[record["sample_id"]]
        with Image.open(root / row["image_path"]) as opened:
            original = opened.convert("RGB")
        width, height = original.size
        heatmap = cv2.resize(
            record["heatmap"].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_u8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        original_array = np.asarray(original)
        overlay = (0.58 * original_array + 0.42 * color).clip(0, 255).astype(np.uint8)
        predicted_box = _largest_component_box(heatmap >= mask_threshold)
        boxed = original.copy()
        draw = ImageDraw.Draw(boxed)
        if predicted_box is not None:
            draw.rectangle(predicted_box, outline=(255, 0, 0), width=4)
        if int(row["label"]):
            gt = (
                int(row["box_x"]),
                int(row["box_y"]),
                int(row["box_x"]) + int(row["box_width"]),
                int(row["box_y"]) + int(row["box_height"]),
            )
            draw.rectangle(gt, outline=(0, 255, 0), width=3)
        label_text = f"p={record['probability']:.3f} {record['variant']}"
        draw.rectangle((0, 0, min(width, 420), 25), fill=(0, 0, 0))
        draw.text((5, 5), label_text, fill=(255, 255, 255))
        stem = _safe_name(record["sample_id"])
        Image.fromarray(color).save(output_dir / f"{stem}_heatmap.png")
        Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.jpg", quality=95)
        boxed.save(output_dir / f"{stem}_boxed.jpg", quality=95)
        if should_select:
            contact_rows.append(
                np.concatenate((np.asarray(boxed), color, overlay), axis=1)
            )
    sheet = np.concatenate(contact_rows, axis=0)
    contact_path = output_dir.parent / "test_box_heatmap_overlay_contact_sheet.jpg"
    Image.fromarray(sheet).save(contact_path, quality=92)
    return contact_path


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint.resolve(), device)
    raw_config = checkpoint["data_config"]
    config = DataConfig(
        image_size=int(raw_config["image_size"]),
        batch_size=int(raw_config["batch_size"]),
        workers=args.workers,
        augmentation="basic",
        paired_sham=False,
    )

    # Calibration uses val only. historical_test is not instantiated until both
    # thresholds have been frozen below.
    val_records = collect_predictions(model, loader(args.main_root, "val", config), device)
    cls_threshold, mask_threshold, calibrated_val = calibrate_thresholds(val_records)
    calibration = {
        "classification_threshold": cls_threshold,
        "mask_threshold": mask_threshold,
        "val_metrics": calibrated_val,
    }
    (output_root / "val_calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    calibrated_checkpoint = dict(checkpoint)
    calibrated_checkpoint["classification_threshold"] = cls_threshold
    calibrated_checkpoint["mask_threshold"] = mask_threshold
    calibrated_checkpoint["thresholds_calibrated"] = True
    calibrated_checkpoint["calibration_split"] = "val"
    if args.validation_only:
        calibrated_checkpoint["selection_status"] = "validation_candidate_not_tested"
        candidate_path = output_root.parent / "best_model_val_calibrated.pt"
        torch.save(calibrated_checkpoint, candidate_path)
        shutil.copy2(args.checkpoint, output_root.parent / "best_model_uncalibrated.pt")
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "candidate_checkpoint": str(candidate_path),
            "classification_threshold": cls_threshold,
            "mask_threshold": mask_threshold,
            "val_metrics_at_calibrated_thresholds": calibrated_val,
            "historical_test_evaluated": False,
        }
        (output_root / "validation_candidate_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    # First and only final test evaluation after model and thresholds are frozen.
    test_records = collect_predictions(
        model, loader(args.main_root, "historical_test", config), device
    )
    test_metrics = compute_metrics(test_records, cls_threshold, mask_threshold)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_frozen": True,
        "thresholds_frozen_before_test": True,
        "classification_threshold": cls_threshold,
        "mask_threshold": mask_threshold,
        "val_metrics_at_calibrated_thresholds": calibrated_val,
        "test_metrics": test_metrics,
        "test_positive_strata": positive_strata(test_records, cls_threshold),
    }
    (output_root / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_prediction_rows(output_root / "test_predictions.csv", test_records, test_metrics)
    contact_path = make_visualizations(
        args.main_root.resolve(),
        test_records,
        mask_threshold,
        output_root / "test_visualizations",
    )
    calibrated_checkpoint["selection_status"] = "frozen_historical_test_evaluated"
    torch.save(calibrated_checkpoint, output_root.parent / "best_model_calibrated.pt")
    shutil.copy2(args.checkpoint, output_root.parent / "best_model_uncalibrated.pt")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"contact_sheet: {contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
