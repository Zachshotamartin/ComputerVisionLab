"""Reproduce portfolio images from the user's archive, without copying datasets."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from visionlab.imaging import color_mask, effect, read_image

cli = argparse.ArgumentParser()
cli.add_argument('--archive', required=True, type=Path)
cli.add_argument('--verification', type=Path, default=ROOT / 'output/verification')
args = cli.parse_args()
target = ROOT / 'web/assets'
target.mkdir(parents=True, exist_ok=True)


def webp(image, name):
    image = ImageOps.contain(image.convert('RGB'), (768, 512))
    image.save(target / name, quality=88, method=6)


bird = read_image(args.archive / 'basics/mycode/data/bird.jpg')
scale = min(768 / bird.shape[1], 512 / bird.shape[0], 1)
bird = cv2.resize(bird, (round(bird.shape[1] * scale), round(bird.shape[0] * scale)), interpolation=cv2.INTER_AREA)
webp(Image.fromarray(cv2.cvtColor(bird, cv2.COLOR_BGR2RGB)), 'bird.webp')
webp(Image.fromarray(effect(bird, 'edges', 100)), 'edges.webp')
mask, boxes = color_mask(bird, (163, 184, 53), tolerance=12, min_area=80)
gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
cover = cv2.cvtColor((gray * .42).astype(np.uint8), cv2.COLOR_GRAY2BGR)
cover[mask > 0] = bird[mask > 0]
for box in boxes:
    x, y, w, h = [box[k] for k in ('x', 'y', 'width', 'height')]
    cv2.rectangle(cover, (x, y), (x+w, y+h), (161, 231, 197), 2)
webp(Image.fromarray(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)), 'cover.webp')

markers = Image.new('RGB', (768, 512), '#ddd8c9')
draw = ImageDraw.Draw(markers)
for x in range(24, 768, 32): draw.line((x, 0, x, 512), fill='#d2cdbc')
for y in range(24, 512, 32): draw.line((0, y, 768, y), fill='#d2cdbc')
draw.rounded_rectangle((82, 94, 286, 298), radius=16, fill='#d95c43')
draw.ellipse((420, 234, 598, 412), fill='#d95c43')
draw.polygon([(414, 82), (560, 108), (515, 230), (370, 204)], fill='#648977')
draw.ellipse((176, 355, 245, 424), fill='#d9b344')
markers.save(target / 'markers.png')

verification = args.verification
webp(Image.open(verification / 'alpaca.jpg'), 'alpaca-prediction.webp')
webp(Image.open(verification / 'weather.jpg'), 'weather-prediction.webp')
reports = ROOT / 'docs/verification'
reports.mkdir(exist_ok=True)
(reports / 'checkpoint-predictions.json').write_text((verification / 'checkpoint-predictions.json').read_text())
(reports / 'parking.json').write_text((verification / 'parking.json').read_text())
provenance = {
    'bird.webp': 'Resized archive basics/mycode/data/bird.jpg; originally used in the OpenCV course exercises.',
    'cover.webp': 'Computed hue mask and connected-region bounds from the repaired Python code.',
    'edges.webp': 'Computed Canny edges from the repaired Python code.',
    'markers.png': 'Deterministic geometric test fixture created by this script.',
    'alpaca-prediction.webp': 'Fresh CPU inference from the user-trained archive checkpoint; source and boxes recorded in verification/checkpoint-predictions.json.',
    'weather-prediction.webp': 'Fresh CPU inference from the user-trained archive checkpoint; source and probabilities recorded in verification/checkpoint-predictions.json.',
}
(reports / 'asset-provenance.json').write_text(json.dumps(provenance, indent=2))
print('Generated', len(provenance), 'small image assets; dataset images and weights remain in the archive.')
