"""Image operations with explicit bounds, circular hue, and checked I/O."""
from pathlib import Path
import math

import cv2
import numpy as np


def read_image(path):
    path = Path(path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Cannot write image: {path}")


def hue_ranges(bgr, tolerance=10, saturation=100, brightness=60):
    """Split the 0..179 OpenCV hue ring before converting to uint8."""
    if len(bgr) != 3 or any(not 0 <= int(c) <= 255 for c in bgr):
        raise ValueError("Color must contain three BGR values between 0 and 255")
    if not 0 <= tolerance <= 89:
        raise ValueError("Hue tolerance must be between 0 and 89")
    if not 0 <= saturation <= 255 or not 0 <= brightness <= 255:
        raise ValueError("Saturation and brightness must be between 0 and 255")
    hue = int(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    low, high = hue - int(tolerance), hue + int(tolerance)
    spans = [(low, high)]
    if low < 0:
        spans = [(0, high), (180 + low, 179)]
    elif high > 179:
        spans = [(low, 179), (0, high - 180)]
    return [(np.array([a, saturation, brightness], np.uint8),
             np.array([b, 255, 255], np.uint8)) for a, b in spans]


def color_mask(image, bgr, tolerance=10, min_area=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for low, high in hue_ranges(bgr, tolerance):
        mask |= cv2.inRange(hsv, low, high)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    objects = []
    for index in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area >= min_area:
            objects.append(dict(x=x, y=y, width=width, height=height, area=area))
        else:
            mask[labels == index] = 0
    return mask, sorted(objects, key=lambda item: item['area'], reverse=True)


def clip_box(box, width, height, padding=0):
    x, y, w, h = box
    if not all(math.isfinite(v) for v in box) or w <= 0 or h <= 0:
        return None
    left, top = max(0, math.floor(x - padding)), max(0, math.floor(y - padding))
    right, bottom = min(width, math.ceil(x + w + padding)), min(height, math.ceil(y + h + padding))
    return (left, top, right, bottom) if right > left and bottom > top else None


def redact(image, boxes, block=18, padding=10):
    output = image.copy()
    for box in boxes:
        bounds = clip_box(box, image.shape[1], image.shape[0], padding)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        roi = image[top:bottom, left:right]
        small = cv2.resize(roi, (max(1, roi.shape[1] // max(1, block)), max(1, roi.shape[0] // max(1, block))), interpolation=cv2.INTER_AREA)
        output[top:bottom, left:right] = cv2.resize(small, (right-left, bottom-top), interpolation=cv2.INTER_NEAREST)
    return output


def face_detector():
    """Load once per session; uses the cascade distributed with OpenCV."""
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        raise ValueError("OpenCV's frontal-face cascade could not be loaded")
    return detector


def faces(image, detector):
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))


def effect(image, mode, amount=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == 'gray':
        return gray
    if mode == 'edges':
        return cv2.Canny(gray, amount, min(255, amount * 2))
    if mode == 'threshold':
        return cv2.threshold(gray, amount, 255, cv2.THRESH_BINARY)[1]
    if mode == 'adaptive':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6)
    if mode == 'blur':
        kernel = max(1, min(99, int(amount))) | 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if mode == 'contours':
        output = image.copy()
        mask = cv2.threshold(gray, amount, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            cv2.drawContours(output, [contour], -1, (100, 220, 160), 2)
        return output
    return image.copy()


def video_loop(source, transform, output=None, show=True):
    """Read before transforming, preserve FPS, and release on every exit."""
    source = int(source) if str(source).isdigit() else str(Path(source).expanduser().resolve())
    capture = cv2.VideoCapture(source)
    writer = None
    frames = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and 0 < fps <= 240 else 30
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if frames == 0:
                    raise ValueError("Video source opened but returned no frames")
                break
            result = transform(frame)
            if output and writer is None:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (result.shape[1], result.shape[0]))
                if not writer.isOpened():
                    raise ValueError(f"Cannot open video output: {output}")
            if writer:
                writer.write(result)
            frames += 1
            if show:
                cv2.imshow('Vision lab — Q to stop', result)
                key = cv2.waitKey(max(1, round(1000 / fps))) & 0xff
                if key in (ord('q'), 27) or cv2.getWindowProperty('Vision lab — Q to stop', cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return frames
