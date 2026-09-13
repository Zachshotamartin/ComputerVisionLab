# Browser inference

The **Trained models** switcher runs retrained weather and retinal OCT classifiers, a parking-space SVM, an alpaca detector, and a separate tiger-pose prototype. [Evaluation](EVALUATION.md) documents the fixed splits, selected checkpoints, metrics, and limitations. Original exports are retained with their historical manifest. Image tools also offer automatic face redaction using the separately attributed pretrained YuNet detector, and geometric rectangle tracking.

## What each tool supports

| Tool | Inputs and outputs | Important boundary |
| --- | --- | --- |
| Weather | Image → cloudy/rain/shine/sunrise scores | Four known classes; a center crop is used |
| Parking | One parking-space crop → empty/occupied and SVM margin | The margin is not a probability; full parking lots are outside training |
| Retinal OCT | Scan → CNV/DME/DRUSEN/NORMAL scores | Patient-separated research evaluation; no clinical validation |
| Alpaca | Image → boxes, scores, PNG/JSON export | Single-class detector; duplicate boxes are suppressed |
| Face redaction | Image or optional camera → automatically pixelated face regions | Pretrained YuNet can miss faces; manual regions remain available |
| Rectangle tracking | Image or optional camera → convex quadrilateral outlines | High-contrast 2D outlines; no depth, camera pose, or identity tracking |
| Tiger pose | Validation example or uploaded image → boxes and 12 landmarks, PNG/JSON export | Separate one-video prototype; does not recover the missing 39-keypoint model |

Model files download only when requested. ONNX Runtime 1.29.0 and its matching single-thread WebAssembly runtime are self-hosted; no external inference endpoint receives image pixels. Models have immutable hash filenames and are checked against byte/hash manifests. Switching away from the image tools stops camera tracks. Cancelling a trained-model run terminates the worker, including pending inference, and invalidates its response. A two-session LRU bounds model switching memory.

## Reproduce exports

Use the archive's trusted Python environment with PyTorch, Ultralytics, scikit-learn, ONNX 1.20.1, and ONNX Runtime 1.24.2. The exporter reads original artifacts and writes only to this repository's generated `output/` and browser asset directories:

```sh
# First reproduce training/evaluation using docs/EVALUATION.md.
python scripts/export-evaluated-models.py --selection docs/verification/evaluation/selection.json
npm run prepare:support
npm run prepare:runtime
npm test
npx playwright install chromium
npm run test:browser
npm run build
```

YOLO exports use fixed inputs (224×224 for weather, 128×128 for OCT, 640×640 for alpacas, 320×320 for tiger pose) and opset 17. `modelMath.js` implements RGB normalization and antialiased center-crop preprocessing, area-resized SVM features, letterboxing, and bounding-box decoding/NMS and coordinate restoration for pose landmarks. The parking binary stores float64 scaling and coefficients with losslessly packed float32 support vectors. Export asserts that this packing preserves the original values, and browser margin tests verify inference parity. YuNet uses BGR 0–255 input and its stride-based output decoder.

`verification/browser-export.json` records source checkpoint hashes, PyTorch-versus-ONNX numeric checks, and reference predictions on the shipped examples. Browser tests compare all ten classifier examples with the Python results (class-score tolerance 0.02; SVM-margin tolerance 0.002), exercise alpaca detection and exports, face redaction, perspective rectangles, cancellation/retry, uploads, mobile layout, and camera cleanup. These checks establish functioning inference and conversion parity, not independent accuracy or generalization.

## Sources and licenses

- The selected weather, OCT, alpaca, and tiger exports use [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) with pretrained initialization and the documented training runs. The original 2024 exports remain preserved. Ultralytics' AGPL-3.0 text is included in `web/assets/models/YOLO-LICENSE.txt`; original checkpoint hashes and relative locations are recorded in the manifest and export report.
- [YuNet from OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet) is separately pretrained and MIT licensed. It is not presented as the user's trained model. Its license is included.
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/deploy.html) is MIT licensed; runtime notices accompany the matching binaries.
- The face example is NASA's public-domain astronaut photograph distributed by [scikit-image](https://scikit-image.org/docs/0.20.x/api/skimage.data.html#skimage.data.astronaut), from version 0.25.2. Archive example provenance is recorded in `web/assets/models/manifest.json`; only representative reduced images are included, not the full datasets.
- The rectangle fixture is generated geometry. It can be reproduced by the export script.

- Tiger annotations come from [Ultralytics Tiger-Pose](https://docs.ultralytics.com/datasets/pose/tiger-pose/), drawn from one source video. The training download is checksum-pinned by `prepare-tiger-pose.py`. Three unmodified validation frames are included as labeled browser examples; they were used for checkpoint selection and are not independent test evidence. Reproduce them with `python scripts/prepare-tiger-examples.py` after preparing the frozen split. Source hashes and provenance are in `verification/tiger-examples.json`. The annotation and upstream-model AGPL notice is included; source-video terms remain separate.
