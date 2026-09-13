"""Summarize matched geometry detections; keep raw fixture predictions local."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "output/evaluation"


def read(name):
    return json.loads((out / name).read_text())


def match(records):
    totals = {
        "cases": len(records),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "negative_images": 0,
        "false_positive_negative_images": 0,
    }
    failures = []
    for row in records:
        actual = row["actual"]
        matched = set()
        tp = 0
        for points in row["truth"]:
            x = min(p[0] for p in points)
            y = min(p[1] for p in points)
            w = max(p[0] for p in points) - x
            h = max(p[1] for p in points) - y
            best = -1
            score = 0
            for i, b in enumerate(actual):
                inter = max(0, min(x + w, b["x"] + b["width"]) - max(x, b["x"])) * max(
                    0, min(y + h, b["y"] + b["height"]) - max(y, b["y"])
                )
                iou = inter / (w * h + b["width"] * b["height"] - inter)
                if i not in matched and iou > score:
                    best = i
                    score = iou
            if score >= 0.7:
                tp += 1
                matched.add(best)
        fp = len(actual) - tp
        fn = len(row["truth"]) - tp
        for key, val in [("tp", tp), ("fp", fp), ("fn", fn)]:
            totals[key] += val
        if not row["truth"]:
            totals["negative_images"] += 1
            totals["false_positive_negative_images"] += bool(actual)
        if fp or fn:
            failures.append({"file": row["file"], "fp": fp, "fn": fn})
    return {
        **totals,
        "precision": totals["tp"] / max(1, totals["tp"] + totals["fp"]),
        "recall": totals["tp"] / max(1, totals["tp"] + totals["fn"]),
        "failures": failures,
    }


report = {
    "scope": "Synthetic rectangles, axis-aligned bounding-box IoU >= 0.7. Development seed 42, holdout seed 31415. Faces use IoU >= 0.3 on transformed images of one identity plus five negatives; not demographic or scene validation.",
    "rectangle": {},
    "face": {},
}
for key, name in [
    ("native_baseline", "native-rectangle-predictions.json"),
    ("native_development", "native-rectangle-improved.json"),
    ("native_holdout", "native-rectangle-holdout.json"),
]:
    report["rectangle"][key] = match(read(name))
for key, name in [
    ("browser_development", "stress/rectangle-results.json"),
    ("browser_holdout", "stress-holdout/rectangle-results.json"),
]:
    report["rectangle"][key] = match(read(name)["records"])
report["rectangle"]["browser_baseline"] = read("browser-geometry-evaluation.json")[
    "summary"
]["rectangle"]
report["face"]["browser_yunet"] = read("browser-geometry-evaluation.json")["summary"][
    "face"
]
for key, name in [
    ("python_haar", "python-face-evaluation.json"),
    ("python_yunet", "python-yunet-evaluation.json"),
]:
    r = read(name)
    report["face"][key] = {k: v for k, v in r.items() if k != "records"}
(out / "geometry-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            k: {
                n: {a: b for a, b in v.items() if a not in ["failures", "scope"]}
                for n, v in value.items()
            }
            for k, value in report.items()
            if k != "scope"
        },
        indent=2,
    )
)
