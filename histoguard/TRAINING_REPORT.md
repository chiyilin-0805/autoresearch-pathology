# HistoGuard contamination classification and localization

Run date: 2026-10-04. Research-only synthetic-data experiment.

## Data protocol

- Main data integrity check: PASS; 1,168 checksummed files verified.
- PanNuke auxiliary integrity check: PASS; 504 checksummed files verified.
- Model weights used 390 fixed `train` samples only.
- Five experiment choices used 78 fixed `val` samples only.
- Classification threshold 0.16 and heatmap threshold 0.125 were calibrated on `val`.
- The frozen model and thresholds were then evaluated once on 78
  `historical_test` samples.
- Splits are case-disjoint for both host and donor case IDs. The historical
  test has prior project use and is not described as a never-seen project-wide
  blind test.

PanNuke has no contamination targets. It was tested only as an unlabeled
reference/calibration consistency objective in experiment 5. That experiment
failed validation and was discarded; the selected final model does not treat
PanNuke images as clean or contaminated labels.

## Five autoresearch experiments

Each experiment used the same RTX 5090, initialization, split, 384-pixel input,
and 300-second wall-clock budget. Selection used a fixed combination of AUROC,
balanced accuracy, soft Dice, box IoU, point-hit rate, and sham control FPR.

| Exp | Change | Score | AUROC | Bal. acc. | Soft Dice | Box IoU | Point hit | Sham FPR | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ResNet-18 + layer-2 mask | 0.762180 | 0.915278 | 0.866667 | 0.467297 | 0.412903 | 0.895833 | 0.125000 | keep |
| 2 | Heavy artifact/stain augmentation | 0.611727 | 0.785069 | 0.741667 | 0.245866 | 0.348629 | 0.708333 | 0.291667 | discard |
| 3 | Multiscale FPN + localized evidence | **0.800063** | 0.881250 | 0.818750 | **0.692256** | 0.657788 | 0.812500 | 0.166667 | **keep / best** |
| 4 | Positive-vs-sham paired ranking | 0.711807 | 0.794097 | 0.775000 | 0.490609 | 0.715844 | 0.895833 | 0.416667 | discard |
| 5 | PanNuke unlabeled consistency | 0.612296 | 0.653819 | 0.691667 | 0.328598 | 0.452394 | 0.729167 | 0.041667 | discard |

## Frozen historical-test result

Classification (78 images; 48 positive and 30 negative):

- AUROC: 0.882639
- Average precision: 0.921404
- Accuracy: 0.833333 (65/78)
- Balanced accuracy: 0.820833
- Sensitivity: 0.875000 (42/48)
- Specificity: 0.766667 (23/30)
- F1: 0.865979
- Clean FPR: 0/6 = 0
- Same-patient sham FPR: 7/24 = 0.291667
- Cross-organ contamination TPR: 21/24 = 0.875000
- Same-organ, other-patient contamination TPR: 21/24 = 0.875000

Localization, with all 48 positives retained in the denominator:

- Soft Dice: 0.741209
- Pixel IoU: 0.701319
- Bounding-box IoU: 0.766610
- Predicted-box center point hit: 0.916667
- End-to-end alarm plus point hit: 0.875000

Positive detection by target area:

- 1%: 10/12 = 0.833333
- 3%: 11/12 = 0.916667
- 5%: 11/12 = 0.916667
- 10%: 10/12 = 0.833333

Positive detection by host organ:

- Kidney: 22/24 = 0.916667
- Lung: 20/24 = 0.833333

## Files and inference

The selected calibrated checkpoint is stored outside Git at:

```text
results/histoguard/best_model_calibrated.pt
```

Predict a file or directory and write classification results, red boxes,
heatmaps, and overlays:

```powershell
D:\AI病理\pathology-contamination-model-ready-v0\.venv\Scripts\python.exe `
  histoguard\predict.py path\to\image_or_folder `
  --checkpoint results\histoguard\best_model_calibrated.pt `
  --output-dir results\histoguard\predictions
```

The contact sheet columns are: boxed image (red prediction, green ground
truth), heatmap, and heatmap overlay.

## Limitations

The task is detection of digitally inserted tissue from another case, not
disease diagnosis. Data include only 56 case identities across Lung and Kidney,
and each host base image produces many correlated variants. The test set has
historical project use. Results do not establish performance on naturally
occurring laboratory floaters, other centers, scanners, stains, organs, or
complete whole-slide images. The calibrated sigmoid score is not a clinical
probability.

## Autoresearch continuation: experiments 6-10

Five more controlled 300-second experiments were run on 2026-10-04. They used
the same train/val split, seed, 384-pixel input and fixed robust score as the
first sweep. The historical test was not read during this continuation.

| Exp | Change | Score | AUROC | Bal. acc. | Soft Dice | Box IoU | Point hit | Sham FPR | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | Sham hard-negative weighting | 0.812111 | 0.855556 | 0.825000 | 0.697429 | 0.702637 | 0.916667 | 0.125000 | keep |
| 7 | Classification coupled to top FPN heatmap evidence | 0.840947 | 0.913889 | 0.818750 | 0.739357 | 0.733184 | 0.937500 | 0.166667 | keep |
| 8 | Independent local-instance MIL head | 0.815445 | 0.883333 | 0.827083 | 0.705225 | 0.745037 | 0.916667 | 0.250000 | discard |
| 9 | Exp7 plus sham heatmap suppression | 0.840463 | 0.895833 | 0.854167 | 0.751426 | 0.776773 | 0.916667 | 0.208333 | discard |
| 10 | Exp7 with ConvNeXt-Tiny backbone | **0.855398** | **0.925694** | **0.864583** | 0.716562 | **0.813219** | **0.979167** | 0.208333 | **keep / best validation candidate** |

Validation-only calibration of experiment 10 selected classification threshold
0.05 and heatmap threshold 0.20. At those thresholds, validation balanced
accuracy was 0.875000, sensitivity 0.916667, specificity 0.833333, soft Dice
0.716561, box IoU 0.862946 and point-hit rate 1.000000. This candidate has not
been rerun on historical_test, because that set has already influenced the
project and must not be reused for iterative model selection.

Candidate checkpoint:

```text
results/histoguard/iteration2_candidate/best_model_val_calibrated.pt
```

- Parameters: 27,948,835
- File size: 111,870,267 bytes
- SHA-256: `24AEB1378125E18FE2B8E30853837E9E2BBDCB3C6868DA9534E34C02BA8B33AE`
- Peak training VRAM: approximately 4.95 GB
- Status: validation-selected research candidate; no new independent test claim

The architecture and project review that motivated these experiments is in
`histoguard/OPEN_SOURCE_RESEARCH.md`.
