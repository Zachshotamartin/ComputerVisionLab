"""Load archived weights without Ultralytics stripping apostrophes in paths."""
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def local_checkpoints():
    """Preserve verified local paths during load and final training validation.

    Ultralytics 8.4.150 strips apostrophes before checking a weight path. Adapt
    only its downloader entry point for this call; never modify installed files.
    Missing/pretrained names keep the library's normal download behavior.
    """
    from ultralytics.utils import downloads
    original = downloads.attempt_download_asset

    def resolve(file, *args, **kwargs):
        path = Path(file).expanduser()
        return str(path) if path.is_file() else original(file, *args, **kwargs)

    downloads.attempt_download_asset = resolve
    try:
        yield
    finally:
        downloads.attempt_download_asset = original


def load_model(checkpoint):
    from ultralytics import YOLO
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        # Explicit pretrained names are left to the library's downloader.
        if path.parent == Path('.') and str(path) in {'yolov8n.pt', 'yolov8n-cls.pt', 'yolov8n-pose.pt'}:
            return YOLO(str(path))
        raise ValueError(f'Model does not exist: {path}')
    with local_checkpoints():
        return YOLO(str(path.resolve()))
