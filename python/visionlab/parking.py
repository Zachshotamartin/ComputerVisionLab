"""Deterministic parking-space classifier with shared RGB preprocessing."""

from pathlib import Path
import hashlib
import json

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .datasets import images
from .imaging import read_image


def features(path):
    # imread(IMREAD_COLOR) makes gray and RGBA inputs consistently three-channel.
    image = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB)
    return (
        cv2.resize(image, (15, 15), interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        .reshape(-1)
        / 255
    )


def train(data, output, seed=42, limit=0):
    names = ["empty", "not_empty"]
    samples, labels, seen, skipped = [], [], {}, []
    for label, name in enumerate(names):
        paths = images(Path(data) / name)
        if limit:
            order = np.random.default_rng(seed).permutation(len(paths))[:limit]
            paths = [paths[index] for index in order]
        for path in paths:
            try:
                vector = features(path)
            except ValueError:
                skipped.append(path.name)
                continue
            digest = hashlib.sha256(vector.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(f"Identical image has conflicting labels: {path}")
                continue
            seen[digest] = label
            samples.append(vector)
            labels.append(label)
    counts = np.bincount(labels, minlength=2)
    if min(counts) < 10:
        raise ValueError("At least 10 distinct readable images per class are required")
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(samples), labels, test_size=0.2, stratify=labels, random_state=seed
    )
    pipeline = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    search = GridSearchCV(
        pipeline,
        {"svc__C": [1, 10], "svc__gamma": ["scale", 0.001]},
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    predicted = search.predict(x_test)
    report = {
        "seed": seed,
        "classes": names,
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "best_parameters": search.best_params_,
        "skipped_files": skipped,
        "evaluation": "Internal stratified image split with duplicate feature vectors removed. Not a held-out camera or location benchmark.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": search.best_estimator_,
            "classes": names,
            "image_size": 15,
            "report": report,
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def predict(model, source):
    payload = joblib.load(model)
    vector = features(source).reshape(1, -1)
    label = int(payload["pipeline"].predict(vector)[0])
    return {
        "source": Path(source).name,
        "label": payload["classes"][label],
        "decision_margin": float(payload["pipeline"].decision_function(vector)[0]),
    }


def sequence_partitions(data, seed=42, limit=0):
    """Separate filename sequences with gaps; never infer nonexistent camera IDs."""
    import re

    data = Path(data).resolve()
    rows = []
    for label, name in enumerate(["empty", "not_empty"]):
        paths = images(data / name)
        if limit:
            indexes = np.random.default_rng(seed).permutation(len(paths))[:limit]
            paths = sorted(paths[i] for i in indexes)
        for path in paths:
            match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
            if not match:
                raise ValueError(
                    "Sequence splitting needs numeric frame_crop filenames. Supply grouped data or explicitly choose --split random for an internal image split."
                )
            rows.append((path, label, match.group(1)))
    groups = sorted({group for _, _, group in rows}, key=int)
    n = len(groups)
    if n < 30:
        raise ValueError(
            "Sequence splitting needs at least 30 filename groups to retain gaps and evaluation partitions"
        )
    assignment = {
        g: "train" if i < int(n * 0.6) else "val" if i < int(n * 0.8) else "test"
        for i, g in enumerate(groups)
    }
    for cut in [int(n * 0.6), int(n * 0.8)]:
        for group in groups[cut - 2 : cut + 2]:
            assignment[group] = "embargo"
    arrays = {}
    seen = {}
    removed = 0
    for split in ["test", "val", "train"]:
        x = []
        y = []
        group_ids = []
        paths = []
        for path, label, group in rows:
            if assignment[group] != split:
                continue
            vector = features(path)
            digest = hashlib.sha256(vector.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(
                        f"Identical features have conflicting labels: {path}"
                    )
                removed += 1
                continue
            seen[digest] = label
            x.append(vector)
            y.append(label)
            group_ids.append(group)
            paths.append(str(path.relative_to(data)))
        if min(np.bincount(y, minlength=2)) < 5:
            raise ValueError(
                f"{split} needs at least five distinct examples of each class after grouping"
            )
        arrays[split] = (np.asarray(x), np.asarray(y), np.asarray(group_ids), paths)
    return arrays, removed


def evaluate_partition(model, partition):
    x, y, groups, paths = partition
    predicted = model.predict(x)
    margins = model.decision_function(x)
    return {
        "samples": len(y),
        "groups": len(set(groups)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "errors": [
            {"path": p, "truth": int(t), "prediction": int(a), "margin": float(m)}
            for p, t, a, m in zip(paths, y, predicted, margins)
            if t != a
        ],
    }


def train_sequence(data, output, seed=42, limit=0, legacy_model=None):
    """Fit scaling inside grouped CV and keep sequence-separated evaluation data."""
    from sklearn.model_selection import GroupKFold
    import time

    arrays, removed = sequence_partitions(data, seed, limit)
    x, y, groups, _ = arrays["train"]
    search = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.001]},
        cv=GroupKFold(3),
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    started = time.time()
    search.fit(x, y, groups=groups)
    report = {
        "seed": seed,
        "grouping": "sorted numeric filename prefixes, contiguous 60/20/20 ranges, two-prefix embargo on each side; camera identity unknown",
        "duplicate_feature_vectors_removed": removed,
        "counts": {
            k: {
                "images": len(v[1]),
                "classes": np.bincount(v[1], minlength=2).tolist(),
                "groups": len(set(v[2])),
            }
            for k, v in arrays.items()
        },
        "best_parameters": search.best_params_,
        "cross_validation": float(search.best_score_),
        "candidate_validation": evaluate_partition(
            search.best_estimator_, arrays["val"]
        ),
        "candidate_test": evaluate_partition(search.best_estimator_, arrays["test"]),
        "duration_seconds": time.time() - started,
    }
    if legacy_model:
        report.update(
            legacy_on_same_test=evaluate_partition(
                joblib.load(legacy_model)["pipeline"], arrays["test"]
            ),
            legacy_limit="Legacy model used a random sample from the full archive and may have seen this holdout. Its score is descriptive, not an independent benchmark.",
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": search.best_estimator_,
            "classes": ["empty", "not_empty"],
            "image_size": 15,
            "report": report,
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    output.with_name(output.stem + "-split.json").write_text(
        json.dumps({k: v[3] for k, v in arrays.items()}, indent=2) + "\n"
    )
    return report
