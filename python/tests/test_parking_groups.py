"""Prevent sequence grouping and embargo regressions in the public training API."""

from pathlib import Path
import tempfile, unittest
import cv2, numpy as np
from visionlab.parking import sequence_partitions


class ParkingGroupTests(unittest.TestCase):
    def test_numeric_sequence_ranges_and_embargoes_are_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label, name in enumerate(["empty", "not_empty"]):
                (root / name).mkdir()
                for frame in range(50):
                    image = np.full((20, 20, 3), label * 120 + frame, dtype=np.uint8)
                    cv2.imwrite(str(root / name / f"{frame}_0.png"), image)
            arrays, _ = sequence_partitions(root)
            groups = {
                key: {int(g) for g in partition[2]} for key, partition in arrays.items()
            }
            self.assertEqual(max(groups["train"]), 27)
            self.assertEqual(min(groups["val"]), 32)
            self.assertEqual(max(groups["val"]), 37)
            self.assertEqual(min(groups["test"]), 42)
            for a in groups:
                for b in groups:
                    if a != b:
                        self.assertFalse(groups[a] & groups[b])

    def test_unknown_grouping_does_not_fall_back_to_random(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "empty").mkdir()
            (root / "not_empty").mkdir()
            cv2.imwrite(
                str(root / "empty" / "unknown.png"),
                np.zeros((20, 20, 3), dtype=np.uint8),
            )
            with self.assertRaisesRegex(ValueError, "numeric frame_crop"):
                sequence_partitions(root)


if __name__ == "__main__":
    unittest.main()
