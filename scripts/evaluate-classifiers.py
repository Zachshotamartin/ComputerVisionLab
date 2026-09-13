"""Evaluate frozen classifier splits; publish mistakes and uncertainty, not only accuracy."""

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from visionlab.models import load_model
from ultralytics.data.augment import classify_transforms

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["weather", "oct"], required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--size", type=int, required=True)
p.add_argument("--name", required=True)
args = p.parse_args()
out = ROOT / "output/evaluation"
data = out / "splits" / args.task
model = load_model(args.checkpoint)
classes = list(model.names.values())
network = model.model.float().eval()
transform = classify_transforms(args.size)
torch.set_num_threads(4)
report = {
    "task": args.task,
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "input_size": args.size,
    "classes": classes,
    "split_manifest_sha256": hashlib.sha256(
        (data / "split-manifest.json").read_bytes()
    ).hexdigest(),
    "partitions": {},
}
for split in ["val", "test"]:
    records = []
    started = time.time()
    files = sorted(
        p
        for p in (data / split).rglob("*")
        if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )
    for offset in range(0, len(files), 32):
        batch = files[offset : offset + 32]
        tensor = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch])
        with torch.inference_mode():
            output = network(tensor)
            probs = (output[0] if isinstance(output, tuple) else output).cpu().numpy()
        for path, scores in zip(batch, probs):
            records.append(
                {
                    "file": str(path.relative_to(data)),
                    "truth": classes.index(path.parent.name),
                    "prediction": int(scores.argmax()),
                    "scores": [float(x) for x in scores],
                }
            )
    y = [r["truth"] for r in records]
    pred = [r["prediction"] for r in records]
    scores = np.array([r["scores"] for r in records])
    scores = scores / scores.sum(axis=1, keepdims=True)
    confidence = scores.max(1)
    correct = np.array(y) == pred
    ece = 0
    bins = []
    for lo in np.arange(0, 1, 0.1):
        mask = (confidence >= lo) & (
            confidence < (lo + 0.1) if lo < 0.9 else confidence <= 1
        )
        if mask.any():
            accuracy = float(correct[mask].mean())
            mean = float(confidence[mask].mean())
            ece += mask.mean() * abs(accuracy - mean)
            bins.append(
                {
                    "lower": float(lo),
                    "samples": int(mask.sum()),
                    "accuracy": accuracy,
                    "mean_score": mean,
                }
            )
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(classes))), zero_division=0
    )
    report["partitions"][split] = {
        "samples": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, scores, labels=list(range(len(classes))))),
        "expected_calibration_error": float(ece),
        "score_bins": bins,
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(len(classes)))
        ).tolist(),
        "per_class": {
            c: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        "errors": [r for r in records if r["truth"] != r["prediction"]],
        "seconds": time.time() - started,
    }
    (out / f"{args.name}-{split}-predictions.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(args.name, split, report["partitions"][split]["accuracy"], flush=True)
(out / f"{args.name}-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
