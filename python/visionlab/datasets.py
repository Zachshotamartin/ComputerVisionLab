"""Lightweight dataset preflight. Never downloads or starts training."""
from pathlib import Path
import math

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def classification_dataset(folder, splits=('train', 'val')):
    root = Path(folder).expanduser().resolve()
    report = {}
    for split in splits:
        directory = root / split
        if not directory.is_dir():
            raise ValueError(f"Missing {split} split: {directory}. Supply an explicit validation split; the test set is not reused for training.")
        classes = {p.name: len(images(p)) for p in sorted(directory.iterdir()) if p.is_dir() and not p.name.startswith('.')}
        if len(classes) < 2 or any(count == 0 for count in classes.values()):
            raise ValueError(f"{directory}: at least two nonempty class folders are required")
        if report and set(classes) != set(next(iter(report.values()))):
            raise ValueError(f"Class folders differ between splits in {root}")
        report[split] = classes
    return report


def yolo_dataset(config, task='detect'):
    import yaml
    config = Path(config).expanduser().resolve()
    document = yaml.safe_load(config.read_text())
    if not isinstance(document, dict) or not document.get('names'):
        raise ValueError("Dataset YAML must define class names")
    root = (config.parent / document.get('path', '.')).resolve()
    class_count = len(document['names'])
    points, dimensions = document.get('kpt_shape', [0, 0])
    if task == 'pose':
        if points <= 0 or dimensions not in (2, 3):
            raise ValueError("Pose datasets need kpt_shape: [points, 2 or 3]")
        flip = document.get('flip_idx', list(range(points)))
        if sorted(flip) != list(range(points)) or any(flip[flip[i]] != i for i in range(points)):
            raise ValueError("flip_idx must be a reversible left/right permutation")
    counts = {}
    for split in ('train', 'val'):
        directory = root / document.get(split, f'images/{split}')
        files = images(directory)
        if not files:
            raise ValueError(f"No {split} images in {directory}. The animal-pose archive has no YOLO keypoint annotations; add images and labeled keypoints before training.")
        counts[split] = len(files)
        for image in files:
            relative = image.relative_to(root)
            parts = list(relative.parts)
            if 'images' not in parts:
                raise ValueError("Use images/<split> and labels/<split> directories")
            parts[parts.index('images')] = 'labels'
            label = root.joinpath(*parts).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f"Missing label file: {label}")
            for line_number, line in enumerate(label.read_text().splitlines(), 1):
                values = [float(value) for value in line.split()]
                expected = 5 + (points * dimensions if task == 'pose' else 0)
                valid = len(values) == expected and all(math.isfinite(v) for v in values)
                if valid:
                    valid = values[0].is_integer() and 0 <= values[0] < class_count and all(0 <= v <= 1 for v in values[1:5]) and values[3] > 0 and values[4] > 0
                if valid and task == 'pose':
                    for offset in range(5, expected, dimensions):
                        valid &= all(0 <= value <= 1 for value in values[offset:offset+2])
                        if dimensions == 3:
                            valid &= values[offset+2] in (0, 1, 2)
                if not valid:
                    raise ValueError(f"Invalid {task} annotation: {label}:{line_number}")
    document['path'] = str(root)
    return document, counts
