# Initial restoration validation

Verified locally on 2026-09-12:

- Seven JavaScript algorithm regressions: red hue wrapping, disconnected regions, image bounds, redaction isolation, blur borders/alpha, edge/threshold behavior, and dimension limits.
- Eight Python regressions: camera failure/empty-stream cleanup, consistent gray/RGBA features, hue wrap, clipped redaction, missing-input errors, validation/test separation, and keypoint annotations.
- Fresh CPU inference from the archived weather and alpaca checkpoints; structured results are in `verification/checkpoint-predictions.json`.
- A new parking run with 300 sampled images per class: 480 training and 120 test images. See `verification/parking.json`; this is an internal split, not a camera-held-out benchmark.
- A one-epoch weather training smoke test with 32 training and 12 validation images, including checkpoint save and final validation on the SSD path containing an apostrophe. This checks the workflow, not model quality.
- Both the repaired original Xcode project and the independent native build script compile successfully. Physical-camera tracking was not exercised by automated tests.
- Portfolio integration against the latest main passed 209 unit tests, lint, and a 44-route production build. Four browser scenarios passed: real processing/export/source link, keyboard redaction/undo, mobile uploads/denied camera, and camera/worker cleanup on navigation. The integration is tracked in [portfolio PR #51](https://github.com/Zachshotamartin/Portfolio/pull/51).

## Remaining prerequisites and limits

The original 39-keypoint project needs annotated images. Subsequent OCT training now uses a patient-separated split and has no clinical validation. The OpenCV frontal-face cascade can miss faces. Browser redaction supports manual regions and automatic YuNet face detection on processed frames; detection can miss faces. Color tracking and the marker overlay identify hue regions, not semantic objects or 3D pose. The archived checkpoints were smoke-tested, not independently benchmarked for generalization.

Historical training records are documented in `REPAIRS.md` and are not displayed as new results or accuracy improvements.

Browser model integration: see [numeric conversion evidence](verification/browser-export.json) and [the browser validation workflow](BROWSER-MODELS.md).

## Evaluation follow-up

See [the current evaluation report](EVALUATION.md) for retrained checkpoints and independent held-out scores. Current regression suites add sequence-group boundaries, optional Python YuNet, dim rectangle detection, binary SVM integrity, and pose coordinate decoding. Browser checks exercise the actual shipped models, uploads, exports, sensitivity switching, cancellation, camera cleanup, and mobile layout. Native compilation and offline synthetic fixtures do not exercise physical-camera permissions.
