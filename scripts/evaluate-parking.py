"""Run the reusable grouped pipeline and preserve the legacy comparison."""

import argparse, json, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from visionlab.parking import train_sequence

cli = argparse.ArgumentParser()
cli.add_argument("--archive", type=Path, required=True)
args = cli.parse_args()
out = ROOT / "output/evaluation"
report = train_sequence(
    args.archive / "imageclassification/parking_classifier/data",
    out / "parking-grouped.joblib",
    legacy_model=args.archive / "output/parking.joblib",
)
(out / "parking-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
shutil.copyfile(out / "parking-grouped-split.json", out / "parking-split.json")
print(report["candidate_test"]["accuracy"])
