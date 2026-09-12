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
    return cv2.resize(image, (15, 15), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255


def train(data, output, seed=42, limit=0):
    names = ['empty', 'not_empty']
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
                    raise ValueError(f'Identical image has conflicting labels: {path}')
                continue
            seen[digest] = label
            samples.append(vector)
            labels.append(label)
    counts = np.bincount(labels, minlength=2)
    if min(counts) < 10:
        raise ValueError('At least 10 distinct readable images per class are required')
    x_train, x_test, y_train, y_test = train_test_split(np.asarray(samples), labels, test_size=.2, stratify=labels, random_state=seed)
    pipeline = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))
    search = GridSearchCV(pipeline, {'svc__C': [1, 10], 'svc__gamma': ['scale', .001]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring='balanced_accuracy', n_jobs=1)
    search.fit(x_train, y_train)
    predicted = search.predict(x_test)
    report = {'seed': seed, 'classes': names, 'training_samples': len(y_train), 'test_samples': len(y_test),
              'accuracy': float(accuracy_score(y_test, predicted)), 'balanced_accuracy': float(balanced_accuracy_score(y_test, predicted)),
              'confusion_matrix': confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(), 'best_parameters': search.best_params_,
              'skipped_files': skipped, 'evaluation': 'Internal stratified image split with duplicate feature vectors removed. Not a held-out camera or location benchmark.'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': search.best_estimator_, 'classes': names, 'image_size': 15, 'report': report}, output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report


def predict(model, source):
    payload = joblib.load(model)
    vector = features(source).reshape(1, -1)
    label = int(payload['pipeline'].predict(vector)[0])
    return {'source': Path(source).name, 'label': payload['classes'][label],
            'decision_margin': float(payload['pipeline'].decision_function(vector)[0])}
