# HistoGuard v2: colon sham 4,000-image single training run

Run date: 2026-10-05. Research-only synthetic-data experiment.

## Version lineage

- Parent model: v1 colon cross-organ 4,000 model.
- Parent SHA-256: `1E9F3B2F89873972A3E13FD4A21B80DD7C37218A8EF7C9107F27688B09FDE30D`.
- New model: v2 colon cross-organ + self-sham 4,000 model.
- New SHA-256: `A22F5BF6AF8E830CA92AFD8C16CD66C9B4AC2F464B57EC05C2DE222376663D1C`.
- Architecture: ConvNeXt-Tiny + multiscale FPN + mask-coupled evidence.
- One 600-second fine-tuning run; no autoresearch iteration or model comparison.

## Data protocol

| Split | Total | Cross-organ positive | Self-sham negative | Clean negative |
|---|---:|---:|---:|---:|
| Train | 2,800 | 1,400 | 700 | 700 |
| Validation | 600 | 300 | 150 | 150 |
| Test | 600 | 300 | 150 | 150 |

Cross-organ donor cases are disjoint across splits. Recipient images are unique
across splits and disjoint from the earlier 4,000-image dataset. NCT does not
provide patient IDs, so recipient-side patient-level isolation cannot be
verified.

The validation set selected the best checkpoint within the single run and
calibrated thresholds. The frozen model and thresholds were then evaluated on
the test split once.

## Frozen test result

- Classification threshold: 0.06
- Mask threshold: 0.40
- Correct: 579/600
- Accuracy: 0.965000
- AUROC: 0.994783
- Average precision: 0.994710
- Sensitivity: 0.953333
- Miss rate: 0.046667 (14/300 contaminated images)
- Specificity: 0.976667
- Overall false-positive rate: 0.023333 (7/300 negatives)
- Clean false-positive rate: 0.000000 (0/150)
- Self-sham false-positive rate: 0.046667 (7/150)
- F1: 0.964587
- Soft Dice: 0.865344
- Pixel IoU: 0.805229
- Bounding-box IoU: 0.818510
- Point-hit rate: 0.950000

## Interpretation limits

The v1 and v2 scores come from different held-out datasets and should not be
treated as a direct leaderboard comparison. Both datasets use synthetic
contamination and the same limited CPTAC donor cohort. Neither result
establishes performance on natural laboratory floaters, complete whole-slide
images, other hospitals/scanners, or clinical use.
