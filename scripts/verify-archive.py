"""Reproduce inference and a small parking run from an existing local archive."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from visionlab.models import load_model
from visionlab.parking import train
from visionlab.imaging import read_image
import cv2

cli = argparse.ArgumentParser()
cli.add_argument('--archive', type=Path, required=True)
cli.add_argument('--output', type=Path, default=ROOT / 'output/verification')
args = cli.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
cases = [
    ('weather', 'imageclassification/weather_classifier/runs/classify/train/weights/best.pt',
     'imageclassification/weather_classifier/data/val/sunrise/sunrise33.jpg', 64),
    ('alpaca', 'objectdetection/alpacadetector/train-yolov8-custom-dataset-step-by-step-guide/runs/detect/train3/weights/best.pt',
     'objectdetection/alpacadetector/train-yolov8-custom-dataset-step-by-step-guide/data/images/test/ec522d35b7604195.jpg', 640),
]
records = []
for name, weights, source, size in cases:
    result = load_model(args.archive / weights).predict(read_image(args.archive / source), imgsz=size, device='cpu', verbose=False)[0]
    cv2.imwrite(str(args.output / f'{name}.jpg'), result.plot())
    records.append({'name': name, 'checkpoint': weights, 'input': source, 'result': json.loads(result.to_json())})
(args.output / 'checkpoint-predictions.json').write_text(json.dumps(records, indent=2))
report = train(args.archive / 'imageclassification/parking_classifier/data', args.output / 'parking.joblib', seed=42, limit=300)
print(json.dumps({'checkpoint_cases': len(records), 'parking': report}, indent=2))
