"""Package immutable HistoGuard v1/v2 checkpoints with parameters and lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-model", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v2-model", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--v2-training-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def checkpoint_parameters(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    experiment = checkpoint["experiment"]
    data_config = checkpoint["data_config"]
    return {
        "architecture": "ConvNeXt-Tiny + multiscale FPN + mask-coupled evidence",
        "experiment": experiment,
        "image_size": int(data_config["image_size"]),
        "batch_size": int(data_config["batch_size"]),
        "classification_threshold": float(checkpoint["classification_threshold"]),
        "mask_threshold": float(checkpoint["mask_threshold"]),
        "thresholds_calibrated": bool(checkpoint.get("thresholds_calibrated", False)),
        "selection_status": checkpoint.get("selection_status"),
        "initial_checkpoint": checkpoint.get("initial_checkpoint"),
    }


def package_version(
    destination: Path,
    *,
    version: str,
    model: Path,
    report: Path,
    dataset: str,
    parent_version: str | None,
    parent_sha256: str | None,
    training_metrics: Path | None = None,
) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    model_destination = destination / "best_model_calibrated.pt"
    shutil.copy2(model, model_destination)
    shutil.copy2(report, destination / "test_report.json")
    if training_metrics is not None:
        shutil.copy2(training_metrics, destination / "training_metrics.json")
    parameters = checkpoint_parameters(model)
    model_hash = sha256(model_destination)
    metadata = {
        "version": version,
        "date": "2026-10-05",
        "dataset": dataset,
        "model_file": model_destination.name,
        "model_size_bytes": model_destination.stat().st_size,
        "sha256": model_hash,
        "parent_version": parent_version,
        "parent_sha256": parent_sha256,
        "parameters": parameters,
        "research_only": True,
        "clinical_validation": False,
    }
    (destination / "parameters.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "README.txt").write_text(
        f"HistoGuard {version}\n"
        f"Dataset: {dataset}\n"
        f"Parent version: {parent_version or 'none'}\n"
        f"SHA-256: {model_hash}\n"
        f"Classification threshold: {parameters['classification_threshold']}\n"
        f"Mask threshold: {parameters['mask_threshold']}\n\n"
        "Research model only. Not clinically validated or approved for diagnosis.\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing version root: {output}")
    output.mkdir(parents=True)
    v1_hash = sha256(args.v1_model.resolve())
    v1 = package_version(
        output / "v1_colon4000",
        version="v1_colon4000",
        model=args.v1_model.resolve(),
        report=args.v1_report.resolve(),
        dataset="colon_cross_organ_dataset_4000",
        parent_version=None,
        parent_sha256=None,
    )
    v2 = package_version(
        output / "v2_colon_sham4000",
        version="v2_colon_sham4000",
        model=args.v2_model.resolve(),
        report=args.v2_report.resolve(),
        dataset="D:/HistoGuard_colon_sham_4000",
        parent_version="v1_colon4000",
        parent_sha256=v1_hash,
        training_metrics=args.v2_training_metrics.resolve(),
    )
    index = {
        "format_version": 1,
        "created": "2026-10-05",
        "latest_version": "v2_colon_sham4000",
        "versions": [v1, v2],
    }
    (output / "versions.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
