import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "create_patient_splits.py"
SPEC = importlib.util.spec_from_file_location("create_patient_splits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PatientSplitTests(unittest.TestCase):
    def test_split_keeps_all_slides_for_a_patient_together(self):
        rows = []
        for patient_number in range(20):
            patient_id = f"P{patient_number:03d}"
            for slide_number in range(2):
                rows.append(
                    {
                        "patient_id": patient_id,
                        "slide_id": f"{patient_id}_S{slide_number}",
                        "filename": f"{patient_id}_{slide_number}.png",
                        "organ": "prostate",
                    }
                )

        assignment = MODULE.assign_patients(rows, seed=123)
        MODULE.validate(rows, assignment)
        counts = {split: list(assignment.values()).count(split) for split in MODULE.SPLIT_ORDER}
        self.assertEqual(counts, {"train": 14, "val": 3, "test": 3})

    def test_incomplete_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "slides.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["patient_id", "slide_id", "filename", "organ"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patient_id": "",
                        "slide_id": "",
                        "filename": "anonymous.png",
                        "organ": "colon",
                    }
                )

            with self.assertRaisesRegex(ValueError, "Refusing image-level splitting"):
                MODULE.read_slides(path)


if __name__ == "__main__":
    unittest.main()
