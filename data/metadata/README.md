# Dataset identity and split status

Only anonymized metadata belongs in this directory. Source pathology images,
patient names, medical record numbers, admission numbers, pathology accession
numbers, and private identity mappings must remain local and are ignored by Git.

## Current identity audit

The local `AI病理/` directory contains three public image collections:

- `NCT-CRC-HE-100K`: colorectal patches; the local files do not include a
  patch-to-patient or patch-to-slide mapping.
- `CRC-VAL-HE-7K`: independent colorectal validation patches; the local files
  do not include a patch-to-patient or patch-to-slide mapping.
- `LC25000`: lung images in this local subset. The released images are heavily
  augmented and the local filenames do not identify the originating patient or
  source image.

For that reason, `slides.csv` is currently an image inventory, not a valid
patient-level identity table. Its `patient_id` and `slide_id` fields remain
empty by design. Do not fill them with per-image IDs.

For LC25000 only, the public `LC25000-clean` annotations can recover the
original source-tile group behind each augmented image. These groups prevent
augmentation leakage, but they are not patients.

Source: <https://github.com/GeorgeBatch/LC25000-clean/blob/main/kaggle/lc25000_image_groups.csv>

SHA-256 used for the current generated metadata:
`2ff9e992f759f265dfc9aa2363de0dfcc8e6559884d8bb02a4c925214fdf7e23`.

## Generate the inventory

```powershell
python scripts/build_pathology_metadata.py
```

To attach the public LC25000 source-tile groups:

```powershell
python scripts/build_pathology_metadata.py `
  --lc25000-groups path/to/lc25000_image_groups.csv
python scripts/create_best_available_splits.py
```

The best-available split uses the official NCT cohort for training, the
independent CRC-VAL cohort for testing, and deterministic 70/15/15
source-tile-group splits within each LC25000 tissue class. The resulting
`splits.csv`, `image_splits.csv`, and `split_summary.csv` are leakage-aware at
the published identity level, but must not be described as patient-level
splits.

To attach identities, prepare a **private, anonymized** mapping with exactly
these columns:

```text
filename,patient_id,slide_id
```

Every image must be mapped. Keep the private mapping outside Git, then run:

```powershell
python scripts/build_pathology_metadata.py --mapping path/to/private_mapping.csv
python scripts/create_patient_splits.py
```

The split command refuses incomplete identities, duplicate filenames, slides
assigned to multiple patients, and any split that does not cover every patient.
It deterministically assigns patients at approximately 70% train, 15%
validation, and 15% test using seed `20261003`.
