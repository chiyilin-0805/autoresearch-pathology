# Colon cross-organ 4,000-image single training run

Run date: 2026-10-05. Research-only synthetic-data experiment.

## Protocol

- Dataset: `runs/colon_cross_organ_dataset_4000`.
- Split target: 70% train, 15% validation, 15% test.
- Actual split: 2,800 train, 600 validation, 600 test.
- Every split is balanced 1:1 between contaminated and clean images.
- Every positive split is balanced 1:1 between lung-to-colon and
  kidney-to-colon contamination.
- Lung and kidney donor cases are disjoint across all three splits.
- Colon recipient output images are unique across all three splits.
- NCT-CRC-HE-100K does not publish patient identifiers, so recipient-side
  patient-level isolation cannot be verified.

The previous best ConvNeXt-Tiny + multiscale FPN + mask-coupled evidence model
was used as initialization. One 600-second fine-tuning run was performed; no
architecture or hyperparameter iteration was run. The validation set selected
the best in-run checkpoint and calibrated thresholds. Model weights and
thresholds were frozen before the test set was read once.

## Split counts

| Split | Total | Positive | Clean | Lung donor | Kidney donor |
|---|---:|---:|---:|---:|---:|
| Train | 2,800 | 1,400 | 1,400 | 700 | 700 |
| Validation | 600 | 300 | 300 | 150 | 150 |
| Test | 600 | 300 | 300 | 150 | 150 |

## Frozen test result

- Correct: 595/600
- Accuracy: 0.991667
- AUROC: 0.999433
- Average precision: 0.999493
- Sensitivity: 0.993333
- Miss rate: 0.006667 (2/300 contaminated images)
- Specificity: 0.990000
- False-positive rate: 0.010000 (3/300 clean images)
- F1: 0.991681
- Soft Dice: 0.893228
- Pixel IoU: 0.825117
- Bounding-box IoU: 0.855475
- Point-hit rate: 0.973333

At the frozen threshold, kidney-to-colon detection was 150/150. Lung-to-colon
detection was 148/150; both misses had about 0.6% contaminated area.

## Outputs

- Calibrated checkpoint:
  `results/histoguard_colon4000_single/best_model_calibrated.pt`
- Machine-readable test report:
  `results/histoguard_colon4000_single/final/test_report.json`
- Per-image test predictions:
  `results/histoguard_colon4000_single/final/test_predictions.csv`
- Test visualization contact sheet:
  `results/histoguard_colon4000_single/final/test_box_heatmap_overlay_contact_sheet.jpg`

## Interpretation limits

These numbers measure performance on synthetic images produced by the same
generation pipeline. The clean test samples do not include same-image paste
shams, and there are only 28 donor cases per organ in the full dataset. The
result does not establish performance on natural laboratory floaters,
whole-slide images, other hospitals/scanners, or clinical use.
