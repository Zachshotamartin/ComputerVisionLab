"""Build compact, attributable public reports from completed runs (no training images)."""

from pathlib import Path
from collections import defaultdict
import json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "output/evaluation"
published = ROOT / "docs/verification/evaluation"
published.mkdir(parents=True, exist_ok=True)


def read(name):
    return json.loads((out / name).read_text())


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        for root, replacement in [
            (str(ROOT), "ComputerVisionLab"),
            (str(ROOT.parent / "computervision"), "computervision"),
        ]:
            value = value.replace(root, replacement)
    return value


def publish(name, report):
    (published / name).write_text(json.dumps(portable(report), indent=2) + "\n")


def cluster_interval(task, name):
    manifest = read(f"splits/{task}/split-manifest.json")
    lookup = {Path(r["path"]).name: r["component"] for r in manifest["sets"]["test"]}
    records = read(f"{name}-test-predictions.json")
    groups = defaultdict(lambda: [0, 0])
    for row in records:
        group = lookup[Path(row["file"]).name]
        groups[group][0] += row["truth"] == row["prediction"]
        groups[group][1] += 1
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(2000):
        sample = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        samples.append(float(sample[0] / sample[1]))
    return {
        "method": "2000 seeded bootstrap resamples of whole duplicate/patient groups",
        "groups": len(groups),
        "accuracy_95_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


summary = {
    "date": "2026-09-12",
    "purpose": "Bounded model evaluation and retraining. Dataset scores do not establish broad real-world or clinical performance.",
    "models": {},
}
for task in ["weather", "oct"]:
    path = out / f"{task}-candidate-evaluation.json"
    if not path.exists():
        continue
    report = read(path.name)
    part = report["partitions"]["test"]
    interval = cluster_interval(task, task + "-candidate")
    summary["models"][task] = {
        "metric": "accuracy",
        "value": part["accuracy"],
        "balanced_accuracy": part["balanced_accuracy"],
        "samples": part["samples"],
        **interval,
        "split": "Patient IDs and exact hashes kept separate"
        if task == "oct"
        else "Exact and conservatively grouped near-duplicate images kept separate",
        "limit": "Balanced subset of one OCT collection; no clinical validation. Legacy test was contaminated."
        if task == "oct"
        else "One small weather collection; grouping is conservative and unfamiliar photo sources remain untested.",
        "source_checkpoint_sha256": report["checkpoint_sha256"],
    }
    # Publish detailed metrics and errors, without duplicating every successful prediction.
    report["test_uncertainty"] = interval
    publish(path.name, report)
if (out / "parking-evaluation.json").exists():
    report = read("parking-evaluation.json")
    part = report["candidate_test"]
    summary["models"]["parking"] = {
        "metric": "accuracy",
        "value": part["accuracy"],
        "balanced_accuracy": part["balanced_accuracy"],
        "samples": part["samples"],
        "groups": part["groups"],
        "split": "Contiguous filename-prefix blocks with gaps between partitions",
        "limit": "Camera and site identifiers are unavailable. This is not an unseen-camera benchmark.",
    }
    publish("parking-evaluation.json", report)
for task in ["alpaca", "tiger"]:
    path = out / f"{task}-candidate-evaluation.json"
    if not path.exists():
        continue
    report = read(path.name)
    key = "metrics/mAP50(P)" if task == "tiger" else "metrics/mAP50(B)"
    summary["models"][task] = {
        "metric": "pose mAP50" if task == "tiger" else "box mAP50",
        "value": report["metrics"][key],
        "map50_95": report["metrics"][
            "metrics/mAP50-95(P)" if task == "tiger" else "metrics/mAP50-95(B)"
        ],
        "samples": report["images"],
        "split": "Later frames with temporal gaps"
        if task == "tiger"
        else "Original held-out test images; development uses a separate grouped split",
        "limit": "A new 12-keypoint tiger prototype from one video. Landmark precision is limited (see stricter mAP50–95); this is not the missing 39-keypoint model or a general animal model."
        if task == "tiger"
        else "One small single-class collection; generalization and negative-image coverage remain limited.",
        "source_checkpoint_sha256": report["checkpoint_sha256"],
    }
    publish(path.name, report)
audit = read("data-audit.json")
(published / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
for name in ["weather", "oct", "alpaca", "tiger"]:
    path = out / "splits" / name / "split-manifest.json"
    manifest = json.loads(path.read_text())
    compact = {
        k: v for k, v in manifest.items() if k not in {"sets", "excluded", "frames"}
    }
    compact["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if name != "tiger":
        compact["groups"] = {
            s: len({r["component"] for r in rows})
            for s, rows in manifest["sets"].items()
        }
        compact["excluded_count"] = len(manifest["excluded"])
    (published / f"{name}-split.json").write_text(json.dumps(compact, indent=2) + "\n")
for pattern in [
    "*-training.json",
    "*-negatives.json",
    "alpaca-legacy-evaluation.json",
    "geometry-evaluation.json",
    "selection.json",
]:
    for path in out.glob(pattern):
        publish(path.name, read(path.name))
(public := ROOT / "web/assets/evaluation.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print("Summarized", list(summary["models"]))
# Human-readable results are generated from the same JSON shown by the tools.
labels = {
    "weather": "Weather",
    "parking": "Parking",
    "oct": "Retinal OCT",
    "alpaca": "Alpacas",
    "tiger": "Tiger pose",
}
lines = [
    "## Recorded results",
    "",
    "These are the selected candidates’ frozen test results, with the scope described above.",
    "",
    "| Model | Measure | Result | Test images |",
    "| --- | --- | ---: | ---: |",
]
for key, item in summary["models"].items():
    lines.append(
        f"| {labels[key]} | {item['metric']} | {item['value'] * 100:.2f}% | {item['samples']:,} |"
    )
lines.extend(
    [
        "",
        "Accuracy and mAP are different measures and should not be compared as one ranking.",
    ]
)
for key, item in summary["models"].items():
    if "accuracy_95_interval" in item:
        lo, hi = item["accuracy_95_interval"]
        lines.append(
            f"- {labels[key]}: group-bootstrap 95% accuracy interval {lo * 100:.1f}–{hi * 100:.1f}%, using {item['groups']} held-out groups. This interval covers within-dataset sampling variation, not shifts to new data sources."
        )
for key in ["weather", "oct"]:
    path = published / f"{key}-candidate-evaluation.json"
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    part = report["partitions"]["test"]
    classes = report["classes"]
    lines.extend(
        [
            "",
            f"### {labels[key]} confusion matrix",
            "",
            "Rows are true labels; columns are predicted labels.",
            "",
            "| True / predicted | " + " | ".join(classes) + " |",
            "| --- | " + " | ".join(["---:"] * len(classes)) + " |",
        ]
    )
    for name, row in zip(classes, part["confusion_matrix"]):
        lines.append("| " + name + " | " + " | ".join(map(str, row)) + " |")
    errors = sorted(part["errors"], key=lambda r: max(r["scores"]), reverse=True)[:5]
    if errors:
        lines.extend(
            [
                "",
                "Examples of confident mistakes (filenames refer to the supplied archive):",
            ]
        )
        for error in errors:
            lines.append(
                f"- `{Path(error['file']).name}`: {classes[error['truth']]} → {classes[error['prediction']]} ({max(error['scores']) * 100:.1f}% model score)."
            )
geometry = read("geometry-evaluation.json")
lines.extend(
    [
        "",
        "### Face and geometry checks",
        "",
        "Face cases contain 27 transformed positives from one identity and five negatives. YuNet found 26 positives, missed one blurred face, and falsely detected the alpaca. Python’s legacy cascade found 22 positives with one false positive. These figures are narrow fixture results.",
        "",
        "Rectangle matching uses bounding-box IoU ≥ 0.7; unmatched extra outlines count as false positives. The native detector can still return duplicate outlines.",
        "",
        "| Rectangle check | Precision | Recall | False-positive negative images |",
        "| --- | ---: | ---: | ---: |",
    ]
)
for key in [
    "browser_development",
    "browser_holdout",
    "native_development",
    "native_holdout",
]:
    r = geometry["rectangle"][key]
    lines.append(
        f"| {key.replace('_', ' ')} | {r['precision'] * 100:.1f}% | {r['recall'] * 100:.1f}% | {r['false_positive_negative_images']}/{r['negative_images']} |"
    )
for task in ["alpaca", "tiger"]:
    if task in summary["models"]:
        item = summary["models"][task]
        lines.extend(
            [
                "",
                f"{labels[task]} mAP50–95, averaged over stricter localization thresholds: {item['map50_95'] * 100:.2f}%.",
            ]
        )
    path = out / f"{task}-candidate-negatives.json"
    if path.exists():
        r = read(path.name)
        lines.extend(
            [
                "",
                f"{labels[task]} produced detections on {r['false_positive_images']} of six curated images without the target animal at a 35% cutoff. This small negative check does not establish open-world reliability.",
            ]
        )
page = ROOT / "docs/EVALUATION.md"
text = page.read_text().split("\n<!-- recorded-results -->")[0]
page.write_text(text + "\n<!-- recorded-results -->\n" + "\n".join(lines) + "\n")
