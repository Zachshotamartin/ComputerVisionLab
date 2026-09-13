# Evaluation and retraining

The original implementation repairs established that the programs ran. This evaluation asks a different question: how well do selected models perform when the data used for fitting and selection are kept apart from the final evaluation?

## Dataset audit

The archive contains 1,124 weather images, 6,090 parking crops, 84,113 retinal OCT scans, and 551 alpaca images. The original animal-pose folders contain no image/keypoint annotations.

- Weather has 24 exact decoded-image duplicate groups, including 10 crossing the old training/validation boundary. Conservative near-duplicate grouping also flags visually similar scenes; these are candidates, not proof of identical photographs.
- OCT has 6,779 exact file duplicate groups, including 595 crossing training/test, and 566 patient IDs on both sides. Only 86 original test scans remain outside both training-patient and exact-image overlap; those have no DRUSEN examples. The historical test accuracy is not an independent four-class evaluation.
- Parking filenames provide 140 prefixes but no camera/location metadata. Sequence separation can be tested; unseen-camera performance cannot be established from this archive.
- Alpaca has no exact duplicates or cross-split dHash candidates at the chosen threshold. Its 72-image test partition remains separate from development.

[Audit evidence](verification/evaluation/data-audit.json) includes counts and examples. All original files remain on the SSD. Split generation writes separate working copies and manifests.

## Evaluation protocol

Weather groups exact decoded duplicates and conservative perceptual-hash components before allocating training, development, and test partitions. Conflicting-label components are excluded. Two training files were GIFs mislabeled as JPEG (`rain141.jpg` and `shine131.jpg`); Ultralytics ignored them, leaving 565 usable training images from 567 listed. The frozen 189-image validation and test sets were unaffected. The candidate starts from ImageNet pretrained YOLOv8n classification weights, not the old weather checkpoint, because the old checkpoint has already seen portions of the regrouped dataset.

OCT joins patient IDs and identical file hashes into indivisible groups, excludes groups overlapping the old test set, and freezes new training/development/test partitions. A balanced experiment uses up to 1,500 training scans and 300 development/test scans per class, capped at 30 scans per patient group. This is a subset experiment, not a full-archive retrain. Neither patient grouping nor model performance constitutes clinical validation.

Parking uses contiguous filename-prefix ranges, with two-prefix gaps on each side of the development and test boundaries. Cross-validation groups prefixes and fits scaling inside each training fold. Exact feature duplicates are removed across partitions. The archived model may have seen some new holdout images; its comparison is descriptive only.

Alpaca training begins from pretrained YOLOv8n detection weights. Development uses a grouped split of the original training/validation pool. The original test partition supplies 72 images and 134 labeled animals. Detection is evaluated with precision, recall, and mAP rather than classifier accuracy. On the same frozen test images, the new candidate improves mAP50 from 76.55% to 86.13% and mAP50–95 from 44.79% to 68.43%. The legacy checkpoint is a historical baseline whose past selection procedure was not independently audited. The six-image negative check improves from four false-positive images to three; false detections remain a limitation.

The new tiger-pose prototype uses Ultralytics' annotated 12-keypoint dataset. Chronological frame blocks and gaps separate training, development, and test. Every frame comes from one video: this demonstrates a runnable pose workflow, not generalization to other animals or scenes. It does not reconstruct the missing 39-keypoint project. Three validation frames (170, 190, 210) are included as browser examples, selected at fixed intervals rather than by prediction quality. They were used for checkpoint selection and are not additional test evidence. Source-video terms remain applicable; the original frozen split report describes the policy at preparation time.

Best epochs are selected using development validation. Original checkpoints are retained. No hyperparameters are selected using the final test scores. Classifier reports include confusion matrices, per-class recall, calibration diagnostics, failure cases, and bootstrap intervals that resample whole patient/duplicate groups.

## Reproduce

Use the archive's trusted Python environment with Python 3.12.5, PyTorch 2.14.0, Ultralytics 8.4.150, scikit-learn 1.9.1, ONNX 1.20.1, and ONNX Runtime 1.24.2, and Node 22 for browser checks. Commands write generated data, models, and detailed predictions under `output/evaluation/`.

```sh
python scripts/audit-model-data.py --archive /path/to/computervision
python scripts/prepare-evaluation-splits.py --archive /path/to/computervision --task weather
python scripts/prepare-evaluation-splits.py --archive /path/to/computervision --task oct
python scripts/prepare-evaluation-splits.py --archive /path/to/computervision --task alpaca
python scripts/evaluate-parking.py --archive /path/to/computervision
python scripts/train-evaluated-model.py --archive /path/to/computervision --task weather --size 224 --epochs 25
python scripts/train-evaluated-model.py --archive /path/to/computervision --task oct --size 128 --epochs 20
python scripts/train-evaluated-model.py --archive /path/to/computervision --task alpaca --size 640 --epochs 30
```

For the separate pose prototype, run `python scripts/prepare-tiger-pose.py --download`, then `python scripts/train-evaluated-model.py --archive /path/to/computervision --task tiger --size 320 --epochs 40`. The preparer verifies the downloaded archive checksum and preserves a chronological split manifest.

Training uses the Mac's MPS GPU and selects the best development epoch with early stopping. Some MPS operations are not bitwise deterministic despite the fixed seed; split membership and evaluation inputs are reproducible. The prepared split manifests are immutable inputs; the scripts refuse to overwrite an existing split directory. Use a separate output workspace when changing the protocol.

To evaluate the selected checkpoints and export them for the browser:

```sh
python scripts/verify-evaluation.py
python scripts/evaluate-classifiers.py --task weather --checkpoint output/evaluation/training/weather-224/weights/best.pt --size 224 --name weather-candidate
python scripts/evaluate-classifiers.py --task oct --checkpoint output/evaluation/training/oct-128/weights/best.pt --size 128 --name oct-candidate
python scripts/evaluate-detectors.py --task alpaca --checkpoint output/evaluation/training/alpaca-640/weights/best.pt --size 640 --name alpaca-candidate
python scripts/evaluate-detectors.py --task tiger --checkpoint output/evaluation/training/tiger-320/weights/best.pt --size 320 --name tiger-candidate
python scripts/evaluate-detector-negatives.py --checkpoint output/evaluation/training/alpaca-640/weights/best.pt --size 640 --name alpaca-candidate
python scripts/evaluate-detector-negatives.py --checkpoint output/evaluation/training/tiger-320/weights/best.pt --size 320 --name tiger-candidate
python scripts/export-evaluated-models.py --selection docs/verification/evaluation/selection.json
npm test
npm run test:browser
```

Scripts require Pillow, NumPy, scikit-learn, OpenCV, PyYAML, PyTorch, Ultralytics, ONNX, and ONNX Runtime in the evaluation environment. The selection file records paths within the generated training workspace. Original source models remain untouched.

## Face redaction and rectangle tracking

The face stress cases use one manually annotated NASA astronaut image with brightness, blur, scale, rotation, grayscale, and occlusion transformations, plus five non-face images. This is a narrow robustness check involving one identity. YuNet can miss a blurred face and falsely detect an animal; the interface offers sensitivity controls and retains manual regions. The Python CLI can use the same model with `--face-model path/to/face.onnx --face-threshold 0.65`; omitting the model retains OpenCV's bundled cascade.

Rectangle fixtures provide known quadrilaterals under contrast, noise, and blur changes. Negative cases include ellipses, triangles, and noise. Development uses seed 42; a separately generated test uses seed 31415. Browser detection smooths noise and uses relative edge strength. Native Vision proposals are checked for sustained contrast along straight sides, which reduces false outlines around curved shapes. These are synthetic geometry tests, not a benchmark of arbitrary real-world scenes.

<!-- recorded-results -->
## Recorded results

These are the selected candidates’ frozen test results, with the scope described above.

| Model | Measure | Result | Test images |
| --- | --- | ---: | ---: |
| Weather | accuracy | 96.30% | 189 |
| Retinal OCT | accuracy | 89.33% | 1,200 |
| Parking | accuracy | 99.92% | 1,278 |
| Alpacas | box mAP50 | 86.13% | 72 |
| Tiger pose | pose mAP50 | 96.23% | 33 |

Accuracy and mAP are different measures and should not be compared as one ranking.
- Weather: group-bootstrap 95% accuracy interval 93.4–98.9%, using 171 held-out groups. This interval covers within-dataset sampling variation, not shifts to new data sources.
- Retinal OCT: group-bootstrap 95% accuracy interval 86.3–91.9%, using 365 held-out groups. This interval covers within-dataset sampling variation, not shifts to new data sources.

### Weather confusion matrix

Rows are true labels; columns are predicted labels.

| True / predicted | cloudy | rain | shine | sunrise |
| --- | ---: | ---: | ---: | ---: |
| cloudy | 53 | 0 | 5 | 0 |
| rain | 0 | 40 | 0 | 0 |
| shine | 0 | 0 | 44 | 1 |
| sunrise | 0 | 0 | 1 | 45 |

Examples of confident mistakes (filenames refer to the supplied archive):
- `cloudy213.jpg`: cloudy → shine (99.1% model score).
- `cloudy116.jpg`: cloudy → shine (96.5% model score).
- `cloudy127.jpg`: cloudy → shine (95.2% model score).
- `sunrise47.jpg`: sunrise → shine (92.2% model score).
- `cloudy113.jpg`: cloudy → shine (81.2% model score).

### Retinal OCT confusion matrix

Rows are true labels; columns are predicted labels.

| True / predicted | CNV | DME | DRUSEN | NORMAL |
| --- | ---: | ---: | ---: | ---: |
| CNV | 281 | 2 | 15 | 2 |
| DME | 8 | 274 | 1 | 17 |
| DRUSEN | 24 | 0 | 238 | 38 |
| NORMAL | 1 | 9 | 11 | 279 |

Examples of confident mistakes (filenames refer to the supplied archive):
- `DME-8253827-1.jpeg`: DME → CNV (100.0% model score).
- `DME-8502722-5.jpeg`: DME → CNV (100.0% model score).
- `DME-8502722-3.jpeg`: DME → CNV (99.8% model score).
- `DRUSEN-1348350-1.jpeg`: DRUSEN → NORMAL (99.8% model score).
- `DRUSEN-2198788-33.jpeg`: DRUSEN → CNV (99.7% model score).

### Face and geometry checks

Face cases contain 27 transformed positives from one identity and five negatives. YuNet found 26 positives, missed one blurred face, and falsely detected the alpaca. Python’s legacy cascade found 22 positives with one false positive. These figures are narrow fixture results.

Rectangle matching uses bounding-box IoU ≥ 0.7; unmatched extra outlines count as false positives. The native detector can still return duplicate outlines.

| Rectangle check | Precision | Recall | False-positive negative images |
| --- | ---: | ---: | ---: |
| browser development | 99.2% | 80.0% | 0/40 |
| browser holdout | 99.3% | 85.0% | 0/40 |
| native development | 82.7% | 92.5% | 0/40 |
| native holdout | 83.7% | 93.1% | 1/40 |

Alpacas mAP50–95, averaged over stricter localization thresholds: 68.43%.

Alpacas produced detections on 3 of six curated images without the target animal at a 35% cutoff. This small negative check does not establish open-world reliability.

Tiger pose mAP50–95, averaged over stricter localization thresholds: 23.71%.

Tiger pose produced detections on 1 of six curated images without the target animal at a 35% cutoff. This small negative check does not establish open-world reliability.
