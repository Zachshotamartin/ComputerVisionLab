"""Evaluate the native OpenCV face backend on the same bounded browser fixtures."""

import json, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from visionlab.imaging import face_detector, faces, read_image

cli = argparse.ArgumentParser()
cli.add_argument("--model", type=Path)
cli.add_argument("--name", default="python-face")
args = cli.parse_args()
out = ROOT / "output/evaluation"
manifest = json.loads((out / "stress/manifest.json").read_text())
detector = face_detector(args.model)
records = []
for item in manifest["cases"]:
    if item["kind"] != "face":
        continue
    actual = [
        list(map(float, box))
        for box in faces(read_image(out / "stress" / item["file"]), detector)
    ]
    matched = set()
    tp = 0
    for truth in item["truth"]:
        best = -1
        score = 0
        for i, b in enumerate(actual):
            inter = max(
                0, min(truth[0] + truth[2], b[0] + b[2]) - max(truth[0], b[0])
            ) * max(0, min(truth[1] + truth[3], b[1] + b[3]) - max(truth[1], b[1]))
            iou = inter / (truth[2] * truth[3] + b[2] * b[3] - inter)
            if i not in matched and iou > score:
                best = i
                score = iou
        if score >= 0.3:
            tp += 1
            matched.add(best)
    records.append(
        dict(
            item, actual=actual, tp=tp, fn=len(item["truth"]) - tp, fp=len(actual) - tp
        )
    )
counts = {k: sum(r[k] for r in records) for k in ["tp", "fp", "fn"]}
report = {
    "scope": manifest["face_ground_truth"],
    "cases": len(records),
    **counts,
    "records": records,
}
(out / f"{args.name}-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
print(counts)
