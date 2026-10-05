"""Five-minute HistoGuard classification and localization experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
)

from prepare import (
    AUX_DATA_DEFAULT,
    MAIN_DATA_DEFAULT,
    DataConfig,
    build_loaders,
    collect_predictions,
    compute_metrics,
    segmentation_loss,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    robust_augmentation: bool = False
    fpn: bool = False
    localized_evidence: bool = False
    paired_sham_weight: float = 0.0
    pannuke_consistency_weight: float = 0.0
    backbone: str = "resnet18"
    classification_mode: str = "legacy"
    sham_classification_weight: float = 1.0
    sham_heatmap_weight: float = 0.0
    learning_rate: float = 2e-4


EXPERIMENTS = {
    1: Experiment("baseline", "ResNet18 classifier plus layer-2 anomaly mask"),
    2: Experiment(
        "artifact_robust",
        "add stain, blur, noise, and label-preserving hard-paste augmentation",
        robust_augmentation=True,
    ),
    3: Experiment(
        "multiscale_localized",
        "add FPN high-resolution mask and top-k localized evidence to the kept baseline",
        robust_augmentation=False,
        fpn=True,
        localized_evidence=True,
    ),
    4: Experiment(
        "paired_sham_ranking",
        "rank each positive above its same-host same-area sham control",
        robust_augmentation=False,
        fpn=True,
        localized_evidence=True,
        paired_sham_weight=0.25,
    ),
    5: Experiment(
        "pannuke_consistency",
        "add unlabeled PanNuke reference/calibration consistency to the kept FPN baseline",
        robust_augmentation=False,
        fpn=True,
        localized_evidence=True,
        paired_sham_weight=0.0,
        pannuke_consistency_weight=0.05,
    ),
    6: Experiment(
        "sham_hard_negative",
        "upweight same-patient sham controls as supervised hard negatives",
        fpn=True,
        localized_evidence=True,
        sham_classification_weight=2.0,
    ),
    7: Experiment(
        "mask_coupled",
        "force top local anomaly-mask evidence to contribute to the image score",
        fpn=True,
        classification_mode="mask_coupled",
    ),
    8: Experiment(
        "local_mil",
        "aggregate top local instances in a CLAM/GigaPath-inspired MIL classifier",
        fpn=True,
        classification_mode="local_mil",
    ),
    9: Experiment(
        "sham_heatmap_suppression",
        "add sham heatmap suppression to the kept mask-coupled model",
        fpn=True,
        classification_mode="mask_coupled",
        sham_heatmap_weight=0.20,
    ),
    10: Experiment(
        "convnext_multiscale",
        "replace ResNet18 in the kept mask-coupled model with ConvNeXt-Tiny",
        fpn=True,
        classification_mode="mask_coupled",
        backbone="convnext_tiny",
        learning_rate=1e-4,
    ),
}


class HistoGuardNet(nn.Module):
    def __init__(
        self,
        *,
        fpn: bool,
        localized_evidence: bool,
        backbone_name: str = "resnet18",
        classification_mode: str = "legacy",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet18":
            backbone = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.stem = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
            )
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            channels = (64, 128, 256, 512)
        elif backbone_name == "convnext_tiny":
            backbone = convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = backbone.features
            channels = (96, 192, 384, 768)
        else:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fpn = fpn
        self.localized_evidence = localized_evidence
        self.classification_mode = classification_mode
        if fpn:
            self.lateral1 = nn.Conv2d(channels[0], 64, 1)
            self.lateral2 = nn.Conv2d(channels[1], 64, 1)
            self.lateral3 = nn.Conv2d(channels[2], 64, 1)
            self.lateral4 = nn.Conv2d(channels[3], 64, 1)
            self.mask_head = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
        else:
            self.mask_head = nn.Sequential(
                nn.Conv2d(channels[1], 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
        if classification_mode == "local_mil":
            self.instance_head = nn.Conv2d(channels[3], 1, 1)
        if classification_mode == "mask_coupled":
            self.evidence_scale = nn.Parameter(torch.tensor(1.0))
        classifier_features = channels[3]
        if classification_mode == "local_mil" or (
            classification_mode == "legacy" and localized_evidence
        ):
            classifier_features += 1
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(classifier_features, 1))

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.backbone_name == "resnet18":
            stem = self.stem(image)
            c1 = self.layer1(stem)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)
        else:
            c1 = self.features[1](self.features[0](image))
            c2 = self.features[3](self.features[2](c1))
            c3 = self.features[5](self.features[4](c2))
            c4 = self.features[7](self.features[6](c3))
        return c1, c2, c3, c4

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        c1, c2, c3, c4 = self.encode(image)
        if self.fpn:
            p4 = self.lateral4(c4)
            p3 = self.lateral3(c3) + torch_f.interpolate(p4, size=c3.shape[-2:], mode="nearest")
            p2 = self.lateral2(c2) + torch_f.interpolate(p3, size=c2.shape[-2:], mode="nearest")
            p1 = self.lateral1(c1) + torch_f.interpolate(p2, size=c1.shape[-2:], mode="nearest")
            mask_logits = self.mask_head(p1)
        else:
            mask_logits = self.mask_head(c2)
        embedding = self.pool(c4).flatten(1)
        if self.classification_mode == "mask_coupled":
            flat_mask = mask_logits.flatten(1)
            top_count = max(1, flat_mask.shape[1] // 20)
            evidence = flat_mask.topk(top_count, dim=1).values.mean(1, keepdim=True)
            logits = (
                self.classifier(embedding).squeeze(1)
                + self.evidence_scale * evidence.squeeze(1)
            )
        elif self.classification_mode == "local_mil":
            instance_logits = self.instance_head(c4).flatten(1)
            top_count = max(1, instance_logits.shape[1] // 5)
            evidence = instance_logits.topk(top_count, dim=1).values.mean(1, keepdim=True)
            logits = self.classifier(torch.cat((embedding, evidence), dim=1)).squeeze(1)
        else:
            classifier_input = embedding
            if self.localized_evidence:
                flat_mask = mask_logits.flatten(1)
                top_count = max(1, flat_mask.shape[1] // 20)
                evidence = flat_mask.topk(top_count, dim=1).values.mean(1, keepdim=True)
                classifier_input = torch.cat((embedding, evidence), dim=1)
            logits = self.classifier(classifier_input).squeeze(1)
        return {
            "logits": logits,
            "mask_logits": mask_logits,
            "embedding": embedding,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--main-root", type=Path, default=MAIN_DATA_DEFAULT)
    parser.add_argument("--aux-root", type=Path, default=AUX_DATA_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=Path("results/histoguard"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--eval-interval", type=float, default=60.0)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional compatible checkpoint used to initialize this single training run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Optional learning-rate override, useful for checkpoint fine-tuning.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_learning_rate(optimizer: torch.optim.Optimizer, base: float, progress: float) -> float:
    rate = base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    for group in optimizer.param_groups:
        group["lr"] = rate
    return rate


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    experiment = EXPERIMENTS[args.experiment]
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    config = DataConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        augmentation="robust" if experiment.robust_augmentation else "basic",
        paired_sham=experiment.paired_sham_weight > 0,
    )
    train_loader, val_loader, pan_loader = build_loaders(
        args.main_root.resolve(), args.aux_root.resolve(), config
    )
    model = HistoGuardNet(
        fpn=experiment.fpn,
        localized_evidence=experiment.localized_evidence,
        backbone_name=experiment.backbone,
        classification_mode=experiment.classification_mode,
        pretrained=args.init_checkpoint is None,
    ).to(device, memory_format=torch.channels_last)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint.resolve(), map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model_state"], strict=True)
    learning_rate = args.learning_rate or experiment.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")
    train_rows = train_loader.dataset.rows
    positives = sum(int(row["label"]) for row in train_rows)
    negatives = len(train_rows) - positives
    positive_weight = torch.tensor([negatives / max(1, positives)], device=device)
    train_iterator = iter(train_loader)
    pan_iterator = iter(pan_loader)

    warmup = next(train_iterator)
    warmup_images = warmup["image"][:4].to(
        device, non_blocking=True, memory_format=torch.channels_last
    )
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        model(warmup_images)
    torch.cuda.synchronize()
    model.train()
    pending_batch: dict[str, Any] | None = warmup

    started = time.perf_counter()
    deadline = started + args.seconds
    next_evaluation = args.eval_interval
    best_score = -math.inf
    best_state = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    steps = examples = 0
    loss_sum = 0.0
    last_rate = learning_rate
    torch.cuda.reset_peak_memory_stats(device)

    while True:
        now = time.perf_counter()
        elapsed = now - started
        if now >= deadline:
            break
        try:
            batch = pending_batch if pending_batch is not None else next(train_iterator)
            pending_batch = None
        except StopIteration:
            train_iterator = iter(train_loader)
            continue
        progress = min(1.0, elapsed / args.seconds)
        last_rate = set_learning_rate(optimizer, learning_rate, progress)
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(images)
            classification_losses = torch_f.binary_cross_entropy_with_logits(
                output["logits"], labels, pos_weight=positive_weight, reduction="none"
            )
            if experiment.sham_classification_weight != 1.0:
                sample_weights = torch.ones_like(labels)
                sham_indices = torch.tensor(
                    [variant == "same_patient_sham" for variant in batch["variant"]],
                    device=device,
                )
                sample_weights[sham_indices] = experiment.sham_classification_weight
                classification_losses = classification_losses * sample_weights
                loss = classification_losses.sum() / sample_weights.sum()
            else:
                loss = classification_losses.mean()
            loss = loss + 0.75 * segmentation_loss(output["mask_logits"], masks)
            if experiment.sham_heatmap_weight:
                sham_indices = torch.tensor(
                    [variant == "same_patient_sham" for variant in batch["variant"]],
                    device=device,
                )
                if sham_indices.any():
                    sham_maps = output["mask_logits"][sham_indices].flatten(1).sigmoid()
                    top_count = max(1, sham_maps.shape[1] // 20)
                    strongest = sham_maps.topk(top_count, dim=1).values.mean()
                    loss = loss + experiment.sham_heatmap_weight * strongest
            if experiment.paired_sham_weight:
                paired_cpu = batch["has_pair"]
                if paired_cpu.any():
                    pair_images = batch["pair_image"][paired_cpu].to(
                        device, non_blocking=True, memory_format=torch.channels_last
                    )
                    pair_logits = model(pair_images)["logits"]
                    positive_logits = output["logits"][paired_cpu.to(device)]
                    ranking = torch_f.softplus(0.75 - positive_logits + pair_logits).mean()
                    loss = loss + experiment.paired_sham_weight * ranking
            if experiment.pannuke_consistency_weight:
                try:
                    pan_first, pan_second = next(pan_iterator)
                except StopIteration:
                    pan_iterator = iter(pan_loader)
                    pan_first, pan_second = next(pan_iterator)
                pan_first = pan_first.to(device, non_blocking=True, memory_format=torch.channels_last)
                pan_second = pan_second.to(device, non_blocking=True, memory_format=torch.channels_last)
                first_embedding = model(pan_first)["embedding"]
                second_embedding = model(pan_second)["embedding"]
                consistency = 1.0 - torch_f.cosine_similarity(
                    first_embedding, second_embedding, dim=1
                ).mean()
                loss = loss + experiment.pannuke_consistency_weight * consistency
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        examples += int(labels.numel())
        loss_sum += float(loss.detach())

        elapsed = time.perf_counter() - started
        if elapsed >= next_evaluation:
            predictions = collect_predictions(model, val_loader, device)
            metrics = compute_metrics(predictions)
            metrics["training_elapsed"] = elapsed
            history.append(metrics)
            print(
                f"val elapsed={elapsed:.1f}s auroc={metrics['auroc']:.4f} "
                f"balanced_acc={metrics['balanced_accuracy']:.4f} "
                f"soft_dice={metrics['soft_dice']:.4f} score={metrics['robust_score']:.4f}",
                flush=True,
            )
            if metrics["robust_score"] > best_score:
                best_score = float(metrics["robust_score"])
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
            model.train()
            next_evaluation += args.eval_interval

    if not history or history[-1]["training_elapsed"] < args.seconds - 5:
        predictions = collect_predictions(model, val_loader, device)
        metrics = compute_metrics(predictions)
        metrics["training_elapsed"] = time.perf_counter() - started
        history.append(metrics)
        print(
            f"val elapsed={metrics['training_elapsed']:.1f}s auroc={metrics['auroc']:.4f} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"soft_dice={metrics['soft_dice']:.4f} score={metrics['robust_score']:.4f}",
            flush=True,
        )
        if metrics["robust_score"] > best_score:
            best_score = float(metrics["robust_score"])
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None or best_metrics is None:
        raise RuntimeError("no validation result was produced")
    model.load_state_dict(best_state)
    run_dir = args.output_root.resolve() / f"exp{args.experiment}_{experiment.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    checkpoint = {
        "model_state": model.state_dict(),
        "experiment_number": args.experiment,
        "experiment": asdict(experiment),
        "data_config": asdict(config),
        "val_metrics": best_metrics,
        "thresholds_calibrated": False,
        "training_policy": "train weights; val model selection; historical_test untouched",
        "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "experiment_number": args.experiment,
        "experiment": asdict(experiment),
        "best_val_metrics": best_metrics,
        "validation_history": history,
        "training": {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "examples_seen": examples,
            "mean_loss": loss_sum / max(1, steps),
            "last_learning_rate": last_rate,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        },
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    report = train(args)
    metrics = report["best_val_metrics"]
    print("---")
    print(f"experiment:       {args.experiment}")
    print(f"name:             {report['experiment']['name']}")
    print(f"robust_score:     {metrics['robust_score']:.6f}")
    print(f"auroc:            {metrics['auroc']:.6f}")
    print(f"balanced_acc:     {metrics['balanced_accuracy']:.6f}")
    print(f"soft_dice:        {metrics['soft_dice']:.6f}")
    print(f"box_iou:          {metrics['box_iou']:.6f}")
    print(f"point_hit:        {metrics['point_hit_rate']:.6f}")
    print(f"sham_fpr:         {metrics['same_patient_sham_fpr']:.6f}")
    print(f"training_seconds: {report['training']['seconds']:.1f}")
    print(f"steps:            {report['training']['steps']}")
    print(f"peak_vram_mb:     {report['training']['peak_vram_mb']:.1f}")
    print(f"checkpoint:       {report['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
