"""Deterministic geometry and single-identity face stress cases, with explicit ground truth."""

from pathlib import Path
import json, math, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

cli = argparse.ArgumentParser()
cli.add_argument("--seed", type=int, default=42)
cli.add_argument("--output", default="stress")
args = cli.parse_args()
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "output/evaluation" / args.output
out.mkdir(parents=True, exist_ok=True)
cases = []
face = Image.open(ROOT / "web/assets/model-examples/face.png").convert("RGB")
box = [176, 65, 96, 114]


def save(image, name, kind, truth, condition):
    image.save(out / (name + ".png"))
    cases.append(
        {"file": name + ".png", "kind": kind, "truth": truth, "condition": condition}
    )


for factor in [0.2, 0.4, 0.6, 1, 1.4, 1.8]:
    save(
        ImageEnhance.Brightness(face).enhance(factor),
        "face-bright-" + str(factor),
        "face",
        [box],
        {"brightness": factor},
    )
for radius in [0, 1, 2, 4, 6, 9]:
    save(
        face.filter(ImageFilter.GaussianBlur(radius)),
        "face-blur-" + str(radius),
        "face",
        [box],
        {"blur": radius},
    )
for size in [64, 96, 128, 192, 256, 384, 512]:
    image = face.resize((size, size))
    save(
        image,
        "face-size-" + str(size),
        "face",
        [[v * size / 512 for v in box]],
        {"size": size},
    )
for angle in [-30, -15, 15, 30]:
    image = face.rotate(angle)
    theta = -angle * math.pi / 180
    corners = []
    for x, y in [(176, 65), (272, 65), (272, 179), (176, 179)]:
        x -= 256
        y -= 256
        corners.append(
            (
                256 + x * math.cos(theta) - y * math.sin(theta),
                256 + x * math.sin(theta) + y * math.cos(theta),
            )
        )
    xs, ys = zip(*corners)
    save(
        image,
        "face-rotate-" + str(angle),
        "face",
        [[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]],
        {"rotation": angle},
    )
for fraction in [0.15, 0.3, 0.5]:
    image = face.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((176, 179 - 114 * fraction, 272, 179), fill="#182c25")
    save(
        image,
        "face-occluded-" + str(fraction),
        "face",
        [box],
        {"lower_face_occlusion": fraction},
    )
save(
    ImageOps.grayscale(face).convert("RGB"),
    "face-gray",
    "face",
    [box],
    {"grayscale": True},
)
for name in [
    "weather-cloudy",
    "weather-rain",
    "weather-shine",
    "weather-sunrise",
    "alpaca",
]:
    save(
        Image.open(ROOT / f"web/assets/model-examples/{name}.png").convert("RGB"),
        "face-negative-" + name,
        "face",
        [],
        {"negative": True},
    )
rng = np.random.default_rng(args.seed)
for i in range(160):
    image = Image.new("RGB", (320, 240), (25, 40, 32))
    draw = ImageDraw.Draw(image)
    x, y = rng.integers(20, 80), rng.integers(20, 65)
    w, h = rng.integers(110, 200), rng.integers(95, 160)
    skew = int(rng.integers(-25, 26))
    points = [
        (int(x), int(y)),
        (int(x + w), int(y + skew)),
        (int(x + w - 12), int(y + h)),
        (int(x + 8), int(y + h - skew)),
    ]
    level = int(rng.choice([45, 65, 100, 150, 220]))
    draw.polygon(points, fill=(level, level, level))
    noise = int(rng.choice([0, 0, 4, 10]))
    blur = float(rng.choice([0, 0, 1, 2]))
    pixels = np.asarray(image, dtype=float)
    if noise:
        pixels += rng.normal(0, noise, pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype("uint8")).filter(
        ImageFilter.GaussianBlur(blur)
    )
    save(
        image,
        f"rectangle-{i:03}",
        "rectangle",
        [points],
        {"level": level, "noise": noise, "blur": blur},
    )
for i in range(40):
    image = Image.new("RGB", (320, 240), (25, 40, 32))
    draw = ImageDraw.Draw(image)
    if i < 20:
        draw.ellipse((45, 30, 270, 215), fill=(190, 190, 190))
    elif i < 30:
        draw.polygon([(160, 25), (280, 215), (30, 210)], fill=(190, 190, 190))
    else:
        image = Image.fromarray(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    save(image, f"geometry-negative-{i:03}", "rectangle", [], {"negative": True})
(out / "manifest.json").write_text(
    json.dumps(
        {
            "seed": args.seed,
            "face_ground_truth": "Manual visible-face rectangle on NASA astronaut image; transformed analytically. One identity only.",
            "geometry_ground_truth": "Generated convex quadrilateral coordinates; negatives include ellipses, triangles, and random noise.",
            "cases": cases,
        },
        indent=2,
    )
    + "\n"
)
print(len(cases), "fixtures")
