"""Keep the small OpenCV lessons runnable from any working directory."""
import argparse
from pathlib import Path


def run(mode, script):
    from . import imaging
    import cv2
    root = next(parent for parent in Path(script).resolve().parents if (parent / 'visionlab').is_dir())
    cli = argparse.ArgumentParser(description=f'OpenCV lesson: {mode}')
    cli.add_argument('--input', type=Path, default=root / 'basics/mycode/data' / ('birds.jpeg' if mode == 'contours' else 'gym.mp4' if mode == 'video' else 'bird.jpg'))
    cli.add_argument('--output', type=Path, default=root / 'output' / f'{mode}.png')
    cli.add_argument('--show', action='store_true')
    args = cli.parse_args()
    try:
        if mode in ('video', 'webcam'):
            imaging.video_loop('0' if mode == 'webcam' else args.input, lambda frame: frame, show=True)
            return
        image = imaging.read_image(args.input)
        if mode == 'crop':
            h, w = image.shape[:2]
            result = image[h//4:max(h//4+1, 3*h//4), w//4:max(w//4+1, 3*w//4)]
        elif mode == 'resize':
            result = cv2.resize(image, (500, max(1, round(image.shape[0] * 500 / image.shape[1]))), interpolation=cv2.INTER_AREA)
        elif mode == 'drawing':
            result = image.copy()
            h, w = result.shape[:2]
            cv2.rectangle(result, (w//4, h//4), (3*w//4, 3*h//4), (100, 220, 160), max(1, w//200))
        else:
            result = imaging.effect(image, mode, 9 if mode == 'blur' else 100)
        imaging.write_image(args.output, result)
        if args.show:
            cv2.imshow(mode, result)
            cv2.waitKey(0)
    except (ValueError, OSError) as error:
        cli.exit(2, f'{error}\n')
    finally:
        cv2.destroyAllWindows()
