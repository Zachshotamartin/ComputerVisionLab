# Computer Vision Lab

Repairs and extensions of Zach Martin's 2024 OpenCV and YOLO exercises. The portfolio adds an interactive image-processing workbench at `/experiments/computer-vision`.

## Run the Python tools

Use Python 3.12 in a virtual environment. Install the repository with `pip install -e '.[models,parking]'`. From the repository root:

```sh
python -m visionlab.cli --help
python -m visionlab.cli image /path/to/photo.jpg --mode edges --output output/edges.png
python -m visionlab.cli track --source 0 --color '#ff0000'
python -m visionlab.cli redact --source /path/to/photo.jpg --output output/redacted.png
python -m visionlab.cli redact --source /path/to/movie.mp4 --headless --output output/redacted.mp4
python -m visionlab.cli train --task classify --data /path/to/weather/data --dry-run
python -m visionlab.cli train --task classify --data /path/to/weather/data --epochs 25 --output output/weather
python -m visionlab.cli predict --model /path/to/best.pt --source /path/to/image.jpg --imgsz 64 --output output/prediction-1
python -m visionlab.cli parking --data /path/to/parking/data --limit 300 --model output/parking.joblib
python -m visionlab.cli parking --source /path/to/parking-space.jpg --model output/parking.joblib
python -m unittest discover -s python/tests -v
```

Camera/video windows stop on Q, Escape, or window close. Cameras are opened only by an explicit command. A missing or empty video source produces a useful error, and camera/writer resources are released even on failure. Use a new output directory for each prediction run. Models and datasets are intentionally absent from this package.

YOLO's path sanitizer removes apostrophes from checkpoint paths. `visionlab.models` preserves verified local paths while loading and during final training validation, restoring the original downloader afterward. Archives on a drive named `Zach's SSD` work without renaming the drive or copying checkpoints. The adapter does not modify installed library files. Loading existing `.pt` or `.joblib` models assumes they are your own trusted artifacts.

## What was fixed

| Project | Repair |
| --- | --- |
| Image I/O and basic lessons | Paths resolve from the script location, failed reads are checked, image exports run without a GUI, crop/resize use actual dimensions, contour lists are passed correctly. |
| Color detection | Integer hue arithmetic and two intervals at the red wrap boundary; independent connected components replace one bounding box around every matching pixel. |
| Face anonymization | One detector per session; clipped padded regions; no negative slicing or empty-region blur crash. The restored implementation uses OpenCV's bundled frontal-face cascade instead of the archive's obsolete MediaPipe Solutions interface. |
| Weather classification | Portable data and checkpoint paths, explicit input argument, deterministic training parameters, validation preflight, structured probability output. |
| Parking classification | Consistent RGB features for grayscale/RGBA inputs, filtered file types, reproducible stratified split, duplicate feature removal, scaling inside cross-validation, balanced scoring, saved preprocessing and class metadata. |
| Alpaca detection | Pretrained initialization, image/annotation checks, explicit best-checkpoint inference, streamed JSON results, no dereference of an unreadable first frame. |
| Retinal OCT classification | Correct identification of CNV/DME/DRUSEN/NORMAL data in the historically named `pneumonia_classifier` folder. The old directory name is retained for compatibility. Training refuses to reuse the test set as validation. |
| Animal pose | Portable YAML and validation of 39-keypoint label shape, coordinates, visibility, and left/right mapping. The archive's YOLO image and label folders are empty: training cannot run until annotated examples are supplied. The AwA2 attribute/feature files are not YOLO keypoint annotations. |
| Native webcam tool | Implemented the missing application entry point and programmatic window, camera permission text and entitlement, start/stop controls, AVFoundation preview, and Vision rectangle outlines. The old storyboard reference was invalid. |

The repaired archive retains the old small entry-point scripts, now forwarding to `visionlab`. Original versions are backed up under `.repair-backup/2026-09-12` on the SSD. None of the datasets, original weights, or historical training runs were removed.

## Native application

The original Xcode project is in `ar_webcam_tool/AR_webcam_tool/AR_webcam_tool.xcodeproj` in the archive. Its updated source is mirrored under `native/macos/`. Build the `AR_webcam_tool` scheme. Start the camera using the button after launch. This is 2D rectangle tracking, not depth reconstruction or world-anchored AR.

The repository can also build the native app independently with `sh native/macos/build.sh /path/to/build`, using the macOS command-line developer tools. Open the resulting `VisionLab.app` to test the camera.

The app was built with code signing disabled for local verification. Camera permission and tracking on a physical camera require an interactive run; the automated checks do not open the user's camera.

## Browser workbench

The React UI, Web Worker, and pure algorithms live in `web/` in this repository and are imported by the portfolio as a commit-pinned package. It supports color tracking, Sobel edges, thresholding, integral-image box blur, manually selected pixelation, and a colored-marker overlay. Images are bounded to 768 × 512; camera processing is bounded to 640 × 480 at up to 10 fps. No image or camera frame is uploaded. Camera tracks and the worker are released on navigation.

Python and browser algorithms are intentionally identified: Python edge detection uses Canny; browser edges expose Sobel magnitude. Python has automatic frontal-face detection; browser redaction uses user-selected regions. The browser marker overlay follows hue regions in 2D and does not infer camera pose. Face detectors can miss faces and selected regions do not follow moving subjects; inspect the result before sharing it.

## Validation and provenance

`docs/verification/` contains fresh checkpoint predictions, the new parking run, and image provenance. The parking run uses a seeded sample of 300 images per class: 480 training images and 120 test images. Its score is an internal image split, not a held-out camera/location benchmark. No accuracy improvement over the old models is claimed.

Historical last-epoch records in the archive reported weather top-1 `0.95833`, OCT top-1 `0.99`, and alpaca mAP50 `0.77691` / mAP50-95 `0.56685`. These are old training-run records, not new evaluation results; the original split methodology has not been independently validated. They are not used as headline portfolio claims. The OCT experiment has no clinical validation.

The archive includes course material from [Computer Vision Engineer's OpenCV course](https://github.com/computervisioneng/opencv-python-course-computer-vision), [color detection exercise](https://github.com/computervisioneng/color-detection-opencv), [face anonymizer exercise](https://github.com/computervisioneng/face-anonymizer-ptyhon), and [YOLO custom dataset guide](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide). The portfolio identifies these as course-derived explorations, with new repair work and a browser implementation. Original upstream source directories and their notices are preserved in the archive.

Reference APIs: [OpenCV cascade detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html), [Ultralytics classification](https://docs.ultralytics.com/tasks/classify/), [prediction](https://docs.ultralytics.com/modes/predict/), and [training](https://docs.ultralytics.com/modes/train/).

Reproduce checkpoint inference and the parking run with `python scripts/verify-archive.py --archive /path/to/computervision`. Example images show actual computed results. Datasets and model weights are not redistributed in this repository.
