"""Train one fresh baseline on frozen splits; select by development data only."""

import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from visionlab.models import load_model, local_checkpoints

cli = argparse.ArgumentParser()
cli.add_argument("--task", choices=["weather", "oct", "alpaca", "tiger"], required=True)
cli.add_argument("--archive", type=Path, required=True)
cli.add_argument("--epochs", type=int, default=20)
cli.add_argument("--size", type=int, required=True)
args = cli.parse_args()
out = ROOT / "output/evaluation"
source = args.archive.resolve()
task = args.task
pretrained = (
    source / "imageclassification/pneumonia_classifier/yolov8n-cls.pt"
    if task in {"weather", "oct"}
    else "yolov8n.pt"
    if task == "alpaca"
    else "yolov8n-pose.pt"
)
data = out / "splits" / task
data = data / "data.yaml" if task in {"alpaca", "tiger"} else data
os.chdir(out)
started = time.time()
with local_checkpoints():
    model = load_model(pretrained)
    result = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.size,
        batch=16,
        device="mps",
        workers=0,
        project=str(out / "training"),
        name=f"{task}-{args.size}",
        seed=42,
        deterministic=True,
        patience=8,
        optimizer="AdamW",
        lr0=0.001,
        amp=False,
        plots=False,
        cache=False,
        exist_ok=False,
        verbose=False,
        **({"fliplr": 0.0, "flipud": 0.0} if task in {"oct", "tiger"} else {}),
    )
    report = {
        "task": task,
        "size": args.size,
        "epochs_requested": args.epochs,
        "duration_seconds": time.time() - started,
        "pretrained": str(pretrained),
        "best_checkpoint": str(model.trainer.best),
        "validation": result.results_dict,
        "selection": "Best epoch by development validation only; frozen test split is not used during fitting.",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": __import__("torch").__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "device": "mps",
        },
        "determinism": "Seeded splits and training; MPS reports some operations are not bitwise deterministic.",
    }
    (out / f"{task}-training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
