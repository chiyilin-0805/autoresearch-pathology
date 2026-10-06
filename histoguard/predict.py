"""Predict contamination, bounding boxes, heatmaps, and overlays for new images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw
from torchvision.transforms import functional as vision_f
from torchvision.transforms.functional import InterpolationMode

from prepare import IMAGENET_MEAN, IMAGENET_STD, _largest_component_box
from train import HistoGuardNet


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/histoguard/best_model_calibrated.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/histoguard/predictions"))
    return parser.parse_args()


def collect_paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for item in inputs:
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(item.resolve())
        elif item.is_dir():
            paths.extend(
                path.resolve()
                for path in item.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            raise FileNotFoundError(item)
    return sorted(set(paths))


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint.resolve(), device)
    image_size = int(checkpoint["data_config"]["image_size"])
    cls_threshold = float(checkpoint["classification_threshold"])
    mask_threshold = float(checkpoint["mask_threshold"])
    rows = []
    for path in collect_paths(args.images):
        with Image.open(path) as opened:
            original = opened.convert("RGB")
        width, height = original.size
        resized = vision_f.resize(
            original, [image_size, image_size], interpolation=InterpolationMode.BILINEAR
        )
        tensor = vision_f.pil_to_tensor(resized).float().div_(255.0)
        tensor = vision_f.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        output = model(tensor.unsqueeze(0).to(device))
        probability = float(output["logits"].sigmoid().item())
        heatmap = torch_f.interpolate(
            output["mask_logits"].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()[0, 0].cpu().numpy()
        binary = heatmap >= mask_threshold
        box = _largest_component_box(binary) if probability >= cls_threshold else None
        heatmap_u8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = (0.58 * np.asarray(original) + 0.42 * color).clip(0, 255).astype(np.uint8)
        boxed = original.copy()
        if box is not None:
            ImageDraw.Draw(boxed).rectangle(box, outline=(255, 0, 0), width=4)
        stem = path.stem
        Image.fromarray(color).save(args.output_dir / f"{stem}_heatmap.png")
        Image.fromarray(overlay).save(args.output_dir / f"{stem}_overlay.jpg", quality=95)
        boxed.save(args.output_dir / f"{stem}_boxed.jpg", quality=95)
        rows.append(
            {
                "image": str(path),
                "prediction": "contaminated" if probability >= cls_threshold else "clean",
                "score": f"{probability:.8f}",
                "classification_threshold": f"{cls_threshold:.4f}",
                "mask_threshold": f"{mask_threshold:.4f}",
                "box_xyxy": "" if box is None else " ".join(map(str, box)),
            }
        )
    output_csv = args.output_dir / "predictions.csv"
    with output_csv.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
