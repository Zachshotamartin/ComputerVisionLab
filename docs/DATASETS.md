# Dataset requirements

The repository contains code and small demonstration assets. Keep full downloaded datasets and original training checkpoints outside Git. Verified browser exports and representative examples are versioned with provenance in this repository.

## Image classification

Provide a root with matching class directories under `train/` and `val/`. Keep `test/` separate for final evaluation. `vision-lab train --task classify --data /path/to/data --dry-run` checks the layout without importing YOLO or starting training.

Weather classes in the archive: `cloudy`, `rain`, `shine`, `sunrise`. The saved checkpoint was trained at image size 64; use `--imgsz 64` to reproduce its inference configuration. New training defaults to 224 and accepts an explicit override.

The historically named `pneumonia_classifier` directory actually contains retinal OCT classes `CNV`, `DME`, `DRUSEN`, `NORMAL`. It has `train/` and `test/`, but no separate `val/`. The repaired trainer refuses to use the test set as validation. Create a validation split from training data, grouped by patient/source identity rather than random adjacent scans, before training. The directory name is retained in the archive for compatibility. This is an educational experiment without clinical validation.

## Parking

The `empty/` and `not_empty/` folders contain cropped parking-space images. The default CLI uses numeric filename-prefix sequence blocks with embargo gaps and grouped cross-validation. Scaling is fitted within each fold, and feature duplicates are removed across partitions. Use `--split random` only to reproduce the older internal image benchmark. Without camera/site metadata, even the sequence split does not establish unseen-camera performance.

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

## Evaluated splits and separate tiger prototype

See [EVALUATION.md](EVALUATION.md) for the immutable weather, patient-grouped OCT, parking, and alpaca protocols. The original 39-keypoint configuration stays available. A separate 12-keypoint tiger dataset and chronological split can be prepared with `python scripts/prepare-tiger-pose.py --download`. All frames come from one video; do not describe its later-frame test as unseen-animal generalization. Downloaded frames remain outside Git.
