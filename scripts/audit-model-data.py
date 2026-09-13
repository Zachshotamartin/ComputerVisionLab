"""Inventory archived datasets without changing them; record overlap and grouping evidence."""

import argparse, hashlib, json, re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cli = argparse.ArgumentParser()
cli.add_argument("--archive", type=Path, required=True)
args = cli.parse_args()
archive = args.archive.resolve()
out = ROOT / "output/evaluation"
out.mkdir(parents=True, exist_ok=True)
roots = {
    "weather": "imageclassification/weather_classifier/data",
    "parking": "imageclassification/parking_classifier/data",
    "oct": "imageclassification/pneumonia_classifier/data",
    "alpaca": "objectdetection/alpacadetector/train-yolov8-custom-dataset-step-by-step-guide/data",
}
report = {}
all_records = {}
for name, relative in roots.items():
    root = archive / relative
    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    def inspect(path):
        rel = path.relative_to(root)
        parts = rel.parts
        split = (
            parts[1]
            if name == "alpaca"
            else parts[0]
            if name != "parking"
            else "unsplit"
        )
        label = "alpaca" if name == "alpaca" else parts[-2]
        row = {
            "path": str(rel),
            "split": split,
            "label": label,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if name == "oct":
            match = re.match(r"^[A-Z]+-(\d+)-", path.name)
            row["group"] = match.group(1) if match else None
        elif name == "parking":
            row["group"] = path.stem.split("_")[0]
        else:
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    row["size"] = list(im.size)
                    row["pixels"] = hashlib.sha256(im.tobytes()).hexdigest()
                    small = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
                    bits = small[:, 1:] > small[:, :-1]
                    row["dhash"] = (
                        f"{int(''.join(map(str, bits.astype(int).ravel())), 2):016x}"
                    )
            except Exception as e:
                row["error"] = str(e)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect, files))
    all_records[name] = rows
    duplicates = defaultdict(list)
    groups = defaultdict(set)
    for row in rows:
        duplicates[row.get("pixels", row["sha256"])].append(row)
        if row.get("group"):
            groups[row["group"]].add(row["split"])
    crossing = [v for v in duplicates.values() if len({r["split"] for r in v}) > 1]
    result = {
        "images": len(rows),
        "counts": dict(Counter(f"{r['split']}/{r['label']}" for r in rows)),
        "duplicate_clusters": sum(len(v) > 1 for v in duplicates.values()),
        "cross_split_duplicate_clusters": len(crossing),
        "cross_split_duplicate_examples": crossing[:10],
        "groups": len(groups),
        "cross_split_groups": sum(len(v) > 1 for v in groups.values()),
        "unreadable": [r for r in rows if r.get("error")],
    }
    if name in {"weather", "alpaca"}:
        pairs = []
        for i, a in enumerate(rows):
            if "dhash" not in a:
                continue
            for b in rows[i + 1 :]:
                if a["split"] == b["split"] or "dhash" not in b:
                    continue
                distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
                if distance <= 4:
                    pairs.append({"a": a["path"], "b": b["path"], "distance": distance})
        result["cross_split_near_duplicate_candidates"] = pairs
    if name == "parking":
        result["group_note"] = (
            "Filename prefix only; camera/site identities are absent. Grouping does not establish unseen-camera generalization."
        )
    report[name] = result
    (out / f"{name}-inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        name,
        json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}),
        flush=True,
    )
report["animal_pose"] = {
    "annotated_images": 0,
    "checkpoint": None,
    "note": "Archive contains attributes and precomputed classification features, not 39-keypoint annotations.",
}
(out / "data-audit.json").write_text(json.dumps(report, indent=2) + "\n")
