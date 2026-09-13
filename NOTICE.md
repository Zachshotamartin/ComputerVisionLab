# Project provenance

The original 2024 archive contains Zach Martin's exercises following Computer Vision Engineer's tutorials:

- [OpenCV course](https://github.com/computervisioneng/opencv-python-course-computer-vision)
- [Color detection](https://github.com/computervisioneng/color-detection-opencv)
- [Face anonymizer](https://github.com/computervisioneng/face-anonymizer-ptyhon)
- [YOLO custom dataset guide](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide)

The original upstream source directories and notices remain in the SSD archive. This repository contains the repaired shared implementation, new browser interface and algorithms, native rectangle tracker, and validation. It does not present the tutorial concepts or pretrained YOLO architecture as original research.

The hummingbird photograph is the image supplied with the archived OpenCV exercises. The color-marker fixture is generated geometry. Other small preview images show actual computations from the archived user-trained checkpoints; their inputs and provenance are recorded in `docs/verification`. Full datasets and original PyTorch checkpoints are not redistributed here. This revision includes browser model exports of the user-trained YOLO checkpoints and the repaired parking SVM, with provenance and upstream notices.

OpenCV, Ultralytics, PyTorch, scikit-learn, React, and the other dependencies retain their own licenses. No blanket license is asserted over upstream course material or dataset photographs.

The browser face detector is pretrained YuNet from OpenCV Zoo (MIT), and the self-hosted inference runtime is ONNX Runtime Web (MIT). Their license texts accompany their artifacts. YOLO-derived model artifacts include the Ultralytics AGPL-3.0 license text. The face example is NASA public-domain imagery distributed by scikit-image. See `docs/BROWSER-MODELS.md` for precise sources and export verification.

The evaluated revision retrains weather, OCT, alpaca, and parking models using documented archive splits. It also trains a separate 12-landmark tiger prototype from the AGPL-annotated Ultralytics Tiger-Pose dataset. All tiger frames come from one source video; downloaded frames remain outside Git and source-video rights remain separate. Selected ONNX weights and reproducible training/export code are included, with the existing YOLO AGPL notice. Original browser artifacts are retained alongside the selected versions.
