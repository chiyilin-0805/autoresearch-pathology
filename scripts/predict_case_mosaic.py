"""Batch-predict coordinate-named WSI patches and build case-level visuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as vision_f
from torchvision.transforms.functional import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "histoguard"))
from prepare import IMAGENET_MEAN, IMAGENET_STD, _largest_component_box  # noqa: E402
from train import HistoGuardNet  # noqa: E402


class CaseDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: Path, rows: list[dict[str, str]], image_size: int, scale: int) -> None:
        self.root = root
        self.rows = rows
        self.image_size = image_size
        self.tile_size = max(1, round(224 / scale))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self.root / row["image_path"]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        preview = vision_f.resize(
            image, [self.tile_size, self.tile_size], interpolation=InterpolationMode.BILINEAR
        )
        resized = vision_f.resize(
            image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR
        )
        tensor = vision_f.pil_to_tensor(resized).float().div_(255.0)
        tensor = vision_f.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return {
            "image": tensor,
            "preview": vision_f.pil_to_tensor(preview),
            "sample_id": row["sample_id"],
            "path": str(path),
            "x": int(row["x"]),
            "y": int(row["y"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overview-scale", type=int, default=16)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> tuple[HistoGuardNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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


def panel(title: str, image: np.ndarray, header: int = 38) -> Image.Image:
    output = Image.new("RGB", (image.shape[1], image.shape[0] + header), "white")
    output.paste(Image.fromarray(image), (0, header))
    draw = ImageDraw.Draw(output)
    draw.text((12, 12), title, fill="black")
    return output


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    root = manifest.parent
    with manifest.open("r", encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["host_case_id"] == args.case_id]
    if not rows:
        raise ValueError(f"no manifest rows for {args.case_id}")
    width = int(rows[0]["wsi_level0_width"])
    height = int(rows[0]["wsi_level0_height"])
    if any(int(row["wsi_level0_width"]) != width or int(row["wsi_level0_height"]) != height for row in rows):
        raise ValueError("inconsistent WSI dimensions")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint.resolve(), device)
    image_size = int(checkpoint["data_config"]["image_size"])
    cls_threshold = float(checkpoint["classification_threshold"])
    mask_threshold = float(checkpoint["mask_threshold"])
    dataset = CaseDataset(root, rows, image_size, args.overview_scale)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    canvas_width = math.ceil(width / args.overview_scale)
    canvas_height = math.ceil(height / args.overview_scale)
    tile_size = dataset.tile_size
    original = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    heat = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    tissue = np.zeros((canvas_height, canvas_width), dtype=bool)
    predictions: list[dict[str, Any]] = []
    boxes: list[tuple[int, int, int, int, float]] = []
    processed = 0

    for batch in loader:
        images = batch["image"].to(
            device, non_blocking=True, memory_format=torch.channels_last
        )
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            result = model(images)
        probabilities = result["logits"].float().sigmoid().cpu().numpy()
        patch_heat = torch_f.interpolate(
            result["mask_logits"].float(),
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        ).sigmoid().cpu().numpy()[:, 0]
        previews = batch["preview"].permute(0, 2, 3, 1).numpy()
        for index, probability in enumerate(probabilities):
            x = int(batch["x"][index]) // args.overview_scale
            y = int(batch["y"][index]) // args.overview_scale
            x2 = min(canvas_width, x + tile_size)
            y2 = min(canvas_height, y + tile_size)
            preview = previews[index, : y2 - y, : x2 - x]
            current_heat = patch_heat[index, : y2 - y, : x2 - x]
            original[y:y2, x:x2] = preview
            heat[y:y2, x:x2] = np.maximum(heat[y:y2, x:x2], current_heat)
            tissue[y:y2, x:x2] = True
            box = None
            if float(probability) >= cls_threshold:
                local_box = _largest_component_box(current_heat >= mask_threshold)
                if local_box is not None:
                    box = (x + local_box[0], y + local_box[1], x + local_box[2], y + local_box[3])
                    boxes.append((*box, float(probability)))
            predictions.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "image": batch["path"][index],
                    "x": int(batch["x"][index]),
                    "y": int(batch["y"][index]),
                    "prediction": "contaminated" if float(probability) >= cls_threshold else "clean",
                    "score": f"{float(probability):.8f}",
                    "classification_threshold": f"{cls_threshold:.4f}",
                    "mask_threshold": f"{mask_threshold:.4f}",
                    "overview_box_xyxy": "" if box is None else " ".join(map(str, box)),
                }
            )
        processed += len(probabilities)
        if processed % 2048 < len(probabilities):
            print(f"processed {processed}/{len(dataset)} patches", flush=True)

    boxed = original.copy()
    boxed_image = Image.fromarray(boxed)
    draw = ImageDraw.Draw(boxed_image)
    for x1, y1, x2, y2, probability in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
    boxed = np.asarray(boxed_image)
    heat_u8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    color[~tissue] = 255
    overlay = original.copy()
    overlay[tissue] = (0.58 * original[tissue] + 0.42 * color[tissue]).clip(0, 255).astype(np.uint8)

    Image.fromarray(original).save(output_dir / f"{args.case_id}_original_overview.jpg", quality=94)
    Image.fromarray(boxed).save(output_dir / f"{args.case_id}_prediction_boxes.jpg", quality=94)
    Image.fromarray(color).save(output_dir / f"{args.case_id}_heatmap.jpg", quality=94)
    Image.fromarray(overlay).save(output_dir / f"{args.case_id}_heatmap_overlay.jpg", quality=94)
    panels = [
        panel("Original overview", original),
        panel(f"Prediction boxes (red), alerts={len(boxes)}", boxed),
        panel("Heatmap overlay", overlay),
    ]
    combined = Image.new("RGB", (sum(item.width for item in panels), max(item.height for item in panels)), "white")
    offset = 0
    for item in panels:
        combined.paste(item, (offset, 0))
        offset += item.width
    combined.save(output_dir / f"{args.case_id}_original_boxes_heatmap.jpg", quality=94)

    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    scores = np.asarray([float(row["score"]) for row in predictions])
    summary = {
        "case_id": args.case_id,
        "patches": len(predictions),
        "alert_patches": int((scores >= cls_threshold).sum()),
        "alert_fraction": float((scores >= cls_threshold).mean()),
        "classification_threshold": cls_threshold,
        "mask_threshold": mask_threshold,
        "maximum_score": float(scores.max()),
        "score_percentiles": {str(q): float(np.percentile(scores, q)) for q in (50, 90, 95, 99, 99.9)},
        "wsi_level0_size": [width, height],
        "overview_scale": args.overview_scale,
        "overview_size": [canvas_width, canvas_height],
        "checkpoint": str(args.checkpoint.resolve()),
        "interpretation": "Unlabeled real case: alerts and heatmaps only; accuracy cannot be calculated.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
