# Dataset requirements

The repository contains code and small demonstration assets. Keep downloaded datasets and trained checkpoints outside Git.

## Image classification

Provide a root with matching class directories under `train/` and `val/`. Keep `test/` separate for final evaluation. `vision-lab train --task classify --data /path/to/data --dry-run` checks the layout without importing YOLO or starting training.

Weather classes in the archive: `cloudy`, `rain`, `shine`, `sunrise`. The saved checkpoint was trained at image size 64; use `--imgsz 64` to reproduce its inference configuration. New training defaults to 224 and accepts an explicit override.

The historically named `pneumonia_classifier` directory actually contains retinal OCT classes `CNV`, `DME`, `DRUSEN`, `NORMAL`. It has `train/` and `test/`, but no separate `val/`. The repaired trainer refuses to use the test set as validation. Create a validation split from training data, grouped by patient/source identity rather than random adjacent scans, before training. The directory name is retained in the archive for compatibility. This is an educational experiment without clinical validation.

## Parking

The `empty/` and `not_empty/` folders contain cropped parking-space images. The training pipeline uses a seeded, stratified image split after removing duplicate feature vectors. Scaling is fitted within each cross-validation fold. This is an internal image benchmark; images from the same camera or recording may still appear on both sides. A deployment evaluation should hold out complete cameras or recordings.

## Object detection

`configs/alpaca.yaml` expects:

```text
data/alpaca/
  images/train/  images/val/
  labels/train/  labels/val/
```

Each image needs a corresponding `.txt` label file. Each row contains `class x_center y_center width height`, with normalized coordinates. Empty label files explicitly represent background images. Unknown classes, non-finite values, out-of-range coordinates, and nonpositive box sizes are rejected. Adjust the YAML `path` to your actual dataset root.

## Animal pose

`configs/animal-pose.yaml` preserves the original 39-keypoint layout and left/right permutation. Labels need a class and box, followed by 39 `(x, y, visibility)` triples. Visibility must be 0, 1, or 2. The archive's YOLO `images/train`, `images/val`, `labels/train`, and `labels/val` directories are empty. Its AwA2 attributes and ResNet feature files do not supply the missing keypoint labels. The repaired preflight makes that prerequisite explicit; no completed animal-pose model is claimed.
