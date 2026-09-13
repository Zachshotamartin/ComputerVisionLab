"""Small curated negative-image check, separate from positive-only detector metrics."""

import argparse, json, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from visionlab.models import load_model
from visionlab.imaging import read_image

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--size", type=int, required=True)
p.add_argument("--name", required=True)
args = p.parse_args()
model = load_model(args.checkpoint)
records = []
files = [
    ROOT / "web/assets/model-examples" / f"{name}.png"
    for name in [
        "weather-cloudy",
        "weather-rain",
        "weather-shine",
        "weather-sunrise",
        "face",
    ]
] + [ROOT / "web/assets/bird.webp"]
for path in files:
    result = model.predict(
        read_image(path),
        imgsz=args.size,
        device="cpu",
        verbose=False,
        conf=0.35,
        iou=0.45,
    )[0]
    records.append(
        {
            "file": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "detections": json.loads(result.to_json()),
        }
    )
report = {
    "scope": "Six curated images without the target animal. This is a small negative sanity check, not open-world validation.",
    "threshold": 0.35,
    "nms_iou": 0.45,
    "false_positive_images": sum(bool(r["detections"]) for r in records),
    "records": records,
}
(ROOT / "output/evaluation" / f"{args.name}-negatives.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(report["false_positive_images"], "false-positive images of", len(records))
