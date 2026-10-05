"""Render top-scoring case patches as original/box/heatmap/overlay rows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw
from torchvision.transforms import functional as vision_f
from torchvision.transforms.functional import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "histoguard"))
from prepare import IMAGENET_MEAN, IMAGENET_STD, _largest_component_box  # noqa: E402
from train import HistoGuardNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> tuple[HistoGuardNet, dict]:
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
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    with args.predictions.open("r", encoding="utf-8-sig", newline="") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: float(row["score"]), reverse=True)[: args.count]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint.resolve(), device)
    image_size = int(checkpoint["data_config"]["image_size"])
    mask_threshold = float(checkpoint["mask_threshold"])
    tiles = []
    for rank, row in enumerate(rows, start=1):
        with Image.open(row["image"]) as opened:
            original = opened.convert("RGB")
        resized = vision_f.resize(original, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
        tensor = vision_f.pil_to_tensor(resized).float().div_(255.0)
        tensor = vision_f.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            result = model(tensor.unsqueeze(0).to(device))
        heat = torch_f.interpolate(
            result["mask_logits"].float(), size=(original.height, original.width), mode="bilinear", align_corners=False
        ).sigmoid()[0, 0].cpu().numpy()
        box = _largest_component_box(heat >= mask_threshold)
        boxed = original.copy()
        if box is not None:
            ImageDraw.Draw(boxed).rectangle(box, outline=(255, 0, 0), width=4)
        heat_u8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = (0.58 * np.asarray(original) + 0.42 * color).clip(0, 255).astype(np.uint8)
        header = 28
        row_image = Image.new("RGB", (original.width * 4, original.height + header), "white")
        captions = (
            f"#{rank} Original p={float(row['score']):.4f}",
            "Prediction box",
            "Heatmap",
            "Overlay",
        )
        images = (original, boxed, Image.fromarray(color), Image.fromarray(overlay))
        draw = ImageDraw.Draw(row_image)
        for column, (caption, image) in enumerate(zip(captions, images)):
            x = column * original.width
            row_image.paste(image, (x, header))
            draw.text((x + 5, 8), caption, fill="black")
        tiles.append(row_image)
    sheet = Image.new("RGB", (tiles[0].width, sum(tile.height for tile in tiles)), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    print(f"Wrote {len(tiles)} alert rows to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
