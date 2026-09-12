import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from visionlab.datasets import classification_dataset, yolo_dataset
from visionlab.imaging import clip_box, color_mask, hue_ranges, read_image, redact, video_loop
from visionlab.parking import features


class ImageTests(unittest.TestCase):
    def test_red_wrap_and_disconnected_regions(self):
        hsv = np.zeros((20, 40, 3), np.uint8)
        hsv[2:8, 2:8] = [179, 255, 255]
        hsv[10:16, 25:31] = [1, 255, 255]
        mask, boxes = color_mask(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (0, 0, 255), min_area=20)
        self.assertEqual(int(np.count_nonzero(mask)), 72)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(hue_ranges((0, 0, 255))), 2)

    def test_clipped_redaction_never_wraps_to_opposite_edge(self):
        image = np.random.default_rng(4).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        output = redact(image, [(-4, -4, 10, 10)], padding=0)
        self.assertFalse(np.array_equal(output[:6, :6], image[:6, :6]))
        self.assertTrue(np.array_equal(output[6:, 6:], image[6:, 6:]))
        self.assertIsNone(clip_box((-20, -20, 3, 3), 20, 20))

    def test_missing_image_has_useful_error(self):
        with self.assertRaisesRegex(ValueError, 'Cannot read image'):
            read_image('/nonexistent/vision-test.png')

    def test_gray_and_rgba_have_consistent_parking_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            for channels in [1, 4]:
                path = Path(folder) / f'{channels}.png'
                cv2.imwrite(str(path), np.full((10, 12, channels), 120, np.uint8))
                self.assertEqual(features(path).shape, (675,))

    def test_failed_camera_releases_without_transform(self):
        with patch('visionlab.imaging.cv2.VideoCapture') as factory:
            capture = factory.return_value
            capture.isOpened.return_value = False
            with self.assertRaisesRegex(ValueError, 'Cannot open'):
                video_loop('0', lambda _: self.fail('Must not transform'), show=False)
            capture.release.assert_called_once()

    def test_empty_capture_releases_without_frame_shape_crash(self):
        with patch('visionlab.imaging.cv2.VideoCapture') as factory:
            capture = factory.return_value
            capture.isOpened.return_value = True
            capture.get.return_value = 30
            capture.read.return_value = (False, None)
            with self.assertRaisesRegex(ValueError, 'no frames'):
                video_loop('0', lambda _: self.fail('Must not transform'), show=False)
            capture.release.assert_called_once()


class DatasetTests(unittest.TestCase):
    def test_test_set_is_not_a_validation_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['a', 'b']:
                path = root / 'train' / name
                path.mkdir(parents=True)
                (path / 'image.png').write_bytes(b'fixture')
            (root / 'test').mkdir()
            with self.assertRaisesRegex(ValueError, 'Missing val split'):
                classification_dataset(root)

    def test_pose_requires_images_and_real_keypoint_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / 'data.yaml'
            config.write_text('path: .\ntrain: images/train\nval: images/val\nnames: [animal]\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n')
            with self.assertRaisesRegex(ValueError, 'No train images'):
                yolo_dataset(config, 'pose')
            for split in ['train', 'val']:
                image = root / 'images' / split / 'a.png'; image.parent.mkdir(parents=True)
                image.write_bytes(b'fixture')
                label = root / 'labels' / split / 'a.txt'; label.parent.mkdir(parents=True)
                label.write_text('0 .5 .5 .2 .2 .4 .4 2 .6 .6 2\n')
            self.assertEqual(yolo_dataset(config, 'pose')[1], {'train': 1, 'val': 1})
            label.write_text('0 .5 .5 .2 .2 .4 .4 9 .6 .6 2\n')
            with self.assertRaisesRegex(ValueError, 'Invalid pose annotation'):
                yolo_dataset(config, 'pose')


if __name__ == '__main__':
    unittest.main()
