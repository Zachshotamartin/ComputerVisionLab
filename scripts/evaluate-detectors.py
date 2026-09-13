"""Evaluate selected detector/pose checkpoints on frozen held-out data."""

import argparse, json, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from visionlab.models import load_model, local_checkpoints

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["alpaca", "tiger"], required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--size", type=int, required=True)
p.add_argument("--name", required=True)
args = p.parse_args()
out = ROOT / "output/evaluation"
data = out / "splits" / args.task
with local_checkpoints():
    model = load_model(args.checkpoint)
    # Device CPU permits evaluation while another model trains on the GPU.
    metrics = model.val(
        data=str(data / "data.yaml"),
        split="test",
        imgsz=args.size,
        batch=8,
        device="cpu",
        workers=0,
        plots=False,
        save_json=True,
        project=str(out / "detection"),
        name=args.name,
        verbose=False,
    )
    report = {
        "task": args.task,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "input_size": args.size,
        "split_manifest_sha256": hashlib.sha256(
            (data / "split-manifest.json").read_bytes()
        ).hexdigest(),
        "metrics": metrics.results_dict,
        "images": len(list((data / "images/test").glob("*.jpg"))),
        "save_dir": str(metrics.save_dir),
        "scope": "Frozen held-out images; tiger frames come from one video and do not demonstrate unseen-animal generalization.",
    }
    (out / f"{args.name}-evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
