"""Verify persisted evaluation splits before accepting reports or exporting a model."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "output/evaluation"
for name in ["weather", "oct", "alpaca"]:
    manifest = json.loads((base / "splits" / name / "split-manifest.json").read_text())
    sets = manifest["sets"]
    for a in sets:
        for b in sets:
            if a >= b:
                continue
            for key in ["component", "sha256"]:
                assert not ({r[key] for r in sets[a]} & {r[key] for r in sets[b]}), (
                    name,
                    a,
                    b,
                    key,
                )
    for split, rows in sets.items():
        expected = sum(manifest["counts"][split].values())
        assert len(rows) == expected
        if name in {"weather", "oct"}:
            assert len({r["label"] for r in rows}) == 4
    print(name, "split boundaries verified")
parking = json.loads((base / "parking-split.json").read_text())
for a in parking:
    for b in parking:
        if a < b:
            assert not (
                {Path(p).stem.split("_")[0] for p in parking[a]}
                & {Path(p).stem.split("_")[0] for p in parking[b]}
            )
print("parking filename-group boundaries verified (camera identity unavailable)")
pose = json.loads((base / "splits/tiger/split-manifest.json").read_text())
frames = {
    s: [r["frame"] for r in pose["frames"] if r["split"] == s]
    for s in ["train", "val", "test"]
}
assert max(frames["train"]) + 19 < min(frames["val"])
assert max(frames["val"]) + 19 < min(frames["test"])
print("tiger temporal embargoes verified (same source video)")
