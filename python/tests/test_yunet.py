"""Integration checks for the optional, explicitly supplied YuNet backend."""

import json, unittest
from pathlib import Path
import numpy as np
from visionlab.imaging import face_detector, faces, read_image, redact

ROOT = Path(__file__).resolve().parents[2]


class YuNetTests(unittest.TestCase):
    def test_letterboxing_preserves_face_position_and_redaction_bounds(self):
        manifest = json.loads((ROOT / "web/assets/models/manifest.json").read_text())
        model = ROOT / "web/assets/models" / manifest["models"]["face"]["file"]
        detector = face_detector(model)
        original = read_image(ROOT / "web/assets/model-examples/face.png")
        canvas = np.zeros((512, 768, 3), dtype=np.uint8)
        canvas[:, 128:640] = original
        before = faces(original, detector)
        after = faces(canvas, detector)
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        a, b = before[0], after[0]
        self.assertAlmostEqual((b[0] + b[2] / 2) - (a[0] + a[2] / 2), 128, delta=12)
        self.assertTrue(
            all(
                0 <= box[0] < 768
                and 0 <= box[1] < 512
                and box[0] + box[2] <= 768
                and box[1] + box[3] <= 512
                for box in after
            )
        )
        output = redact(canvas, after)
        self.assertFalse(np.array_equal(canvas, output))
        np.testing.assert_array_equal(canvas[:, :100], output[:, :100])
        self.assertEqual(faces(np.zeros_like(canvas), detector), [])

    def test_invalid_explicit_model_reports_an_error(self):
        with self.assertRaisesRegex(ValueError, "Face model does not exist"):
            face_detector("/missing/visionlab-face.onnx")


if __name__ == "__main__":
    unittest.main()
