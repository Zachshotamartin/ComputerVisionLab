"""Freeze grouped development and held-out splits; never rewrite source datasets."""

import argparse, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/evaluation"
cli = argparse.ArgumentParser()
cli.add_argument("--archive", type=Path, required=True)
cli.add_argument("--task", choices=["weather", "oct", "alpaca"], required=True)
args = cli.parse_args()
name = args.task
roots = {
    "weather": "imageclassification/weather_classifier/data",
    "oct": "imageclassification/pneumonia_classifier/data",
    "alpaca": "objectdetection/alpacadetector/train-yolov8-custom-dataset-step-by-step-guide/data",
}
source = args.archive.resolve() / roots[name]
rows = json.loads((OUT / f"{name}-inventory.json").read_text())
destination = OUT / "splits" / name
if destination.exists():
    raise SystemExit(f"Split already frozen: {destination}")
parents = list(range(len(rows)))


def find(i):
    while parents[i] != i:
        parents[i] = parents[parents[i]]
        i = parents[i]
    return i


def union(a, b):
    parents[find(a)] = find(b)


seen = {}
for i, row in enumerate(rows):
    key = row.get("pixels", row["sha256"])
    if key in seen:
        union(i, seen[key])
    seen[key] = i
    if name == "oct":
        if not row.get("group"):
            raise ValueError("Missing patient identity")
        key = "patient:" + row["group"]
        if key in seen:
            union(i, seen[key])
        seen[key] = i
if name in {"weather", "alpaca"}:
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if (
                "dhash" in a
                and "dhash" in b
                and (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count() <= 4
            ):
                union(i, j)
components = defaultdict(list)
for i, row in enumerate(rows):
    components[find(i)].append(row)
excluded = []
selected = []
for group, members in components.items():
    if name != "oct" and len({r["label"] for r in members}) > 1:
        excluded.extend(
            dict(r, reason="conflicting near-duplicate labels") for r in members
        )
        continue
    for row in members:
        row["component"] = str(group)
    selected.extend(members)
if name == "oct":
    test_groups = {r["component"] for r in selected if r["split"] == "test"}
    holdout = [
        r
        for r in selected
        if r["split"] == "test"
        and not any(m["split"] == "train" for m in components[int(r["component"])])
    ]
    excluded.extend(
        dict(r, reason="training patient or exact-image overlap with archive test")
        for r in selected
        if r["component"] in test_groups and r not in holdout
    )
    pool = [r for r in selected if r["component"] not in test_groups]
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            pool, [r["label"] for r in pool], [r["component"] for r in pool]
        )
    )
    test_idx = set(folds[0][1])
    val_idx = set(folds[1][1])
    sets = {"train": [], "val": [], "test": []}
    for i, row in enumerate(pool):
        sets["test" if i in test_idx else "val" if i in val_idx else "train"].append(
            row
        )
    # The 86 genuinely separate archived test scans lack DRUSEN; record them
    # separately rather than claiming a balanced four-class test.
    (OUT / "oct-archive-clean-test.json").write_text(
        json.dumps(holdout, indent=2) + "\n"
    )
    # A balanced, patient-diverse training experiment; held-out test stays complete.
    for split, cap in [("train", 1500), ("val", 300), ("test", 300)]:
        retained = []
        rng = np.random.default_rng(42)
        for label in sorted({r["label"] for r in sets[split]}):
            candidates = [r for r in sets[split] if r["label"] == label]
            rng.shuffle(candidates)
            counts = Counter()
            for row in candidates:
                if counts[row["component"]] >= 30:
                    continue
                retained.append(row)
                counts[row["component"]] += 1
                if sum(counts.values()) >= cap:
                    break
        sets[split] = retained
elif name == "alpaca":
    # Preserve the archive's unused test split; develop on training/validation.
    pool = [r for r in selected if r["split"] != "test"]
    holdout = [r for r in selected if r["split"] == "test"]
    test_groups = {r["component"] for r in holdout}
    pool = [r for r in pool if r["component"] not in test_groups]
    train_idx, val_idx = next(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            pool, [r["label"] for r in pool], [r["component"] for r in pool]
        )
    )
    sets = {
        "train": [pool[i] for i in train_idx],
        "val": [pool[i] for i in val_idx],
        "test": holdout,
    }
else:
    # Every duplicate cluster stays together; only one exact copy is retained.
    unique = {}
    for row in selected:
        unique.setdefault(row.get("pixels", row["sha256"]), row)
    pool = list(unique.values())
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            pool, [r["label"] for r in pool], [r["component"] for r in pool]
        )
    )
    test_idx = set(folds[0][1])
    val_idx = set(folds[1][1])
    sets = {"train": [], "val": [], "test": []}
    for i, row in enumerate(pool):
        sets["test" if i in test_idx else "val" if i in val_idx else "train"].append(
            row
        )
for a in sets:
    for b in sets:
        if a != b:
            assert not (
                {r["component"] for r in sets[a]} & {r["component"] for r in sets[b]}
            )
for split, items in sets.items():
    for row in items:
        path = source / row["path"]
        target = (
            destination
            / ("images/" + split if name == "alpaca" else split + "/" + row["label"])
            / path.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if name == "alpaca":
            label = source / "labels" / row["split"] / (path.stem + ".txt")
            label_dest = destination / "labels" / split / label.name
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(label, label_dest)
if name == "alpaca":
    import yaml

    (destination / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "alpaca"},
            }
        )
    )
manifest = {
    "seed": 42,
    "task": name,
    "source": roots[name],
    "grouping": "patient ID plus exact hashes"
    if name == "oct"
    else "decoded pixel hashes and conservative dHash distance <=4 components",
    "counts": {k: dict(Counter(r["label"] for r in v)) for k, v in sets.items()},
    "sets": sets,
    "excluded": excluded,
    "legacy_comparison": "Original checkpoints saw training/validation images in these regrouped sets. Their scores are descriptive only; new candidate hold-outs are independent of candidate fitting.",
}
(destination / "split-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["counts"]), flush=True)
