import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "create_best_available_splits.py"
SPEC = importlib.util.spec_from_file_location("create_best_available_splits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BestAvailableSplitTests(unittest.TestCase):
    def test_source_group_never_crosses_splits(self):
        rows = []
        for class_name in ("lung_aca", "lung_n", "lung_scc"):
            for group_number in range(20):
                group_id = f"{class_name}_{group_number}"
                for image_number in range(2):
                    rows.append(
                        {
                            "patient_id": "",
                            "group_id": group_id,
                            "identity_type": "source_tile",
                            "image_id": f"{group_id}_{image_number}",
                            "filename": f"{group_id}_{image_number}.jpeg",
                            "dataset": "LC25000",
                            "organ": "lung",
                            "tissue_class": class_name,
                        }
                    )
        rows.extend(
            [
                {
                    "patient_id": "",
                    "group_id": "",
                    "identity_type": "unavailable",
                    "image_id": "nct_1",
                    "filename": "nct_1.tif",
                    "dataset": "NCT-CRC-HE-100K",
                    "organ": "colon",
                    "tissue_class": "TUM",
                },
                {
                    "patient_id": "",
                    "group_id": "",
                    "identity_type": "unavailable",
                    "image_id": "crc_1",
                    "filename": "crc_1.tif",
                    "dataset": "CRC-VAL-HE-7K",
                    "organ": "colon",
                    "tissue_class": "TUM",
                },
            ]
        )

        units, images = MODULE.create_assignments(rows, seed=42)
        observed = {}
        for row in images:
            observed.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in observed.values()))
        self.assertEqual(len(images), len(rows))
        self.assertTrue(any(row["identity_type"] == "official_cohort" for row in units))


if __name__ == "__main__":
    unittest.main()
