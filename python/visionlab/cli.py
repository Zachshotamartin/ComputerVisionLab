"""Run with python -m visionlab.cli. Imports do not open cameras or train."""

import argparse
import json
from pathlib import Path

from .datasets import classification_dataset, yolo_dataset


def parser():
    cli = argparse.ArgumentParser(
        description="Computer vision experiments: portable inputs, reproducible runs, explicit validation."
    )
    commands = cli.add_subparsers(dest="command", required=True)
    image = commands.add_parser("image", help="Process a still image")
    image.add_argument("source", type=Path)
    image.add_argument(
        "--mode",
        choices=[
            "original",
            "gray",
            "edges",
            "threshold",
            "adaptive",
            "blur",
            "contours",
        ],
        default="edges",
    )
    image.add_argument("--amount", type=int, default=100)
    image.add_argument("--output", type=Path, default=Path("output/image.png"))
    image.add_argument("--show", action="store_true")
    for name in ("track", "redact"):
        camera = commands.add_parser(
            name, help="Process an image, video, or camera; Q exits"
        )
        camera.add_argument("--source", default="0")
        camera.add_argument("--output", type=Path)
        camera.add_argument("--headless", action="store_true")
        camera.add_argument("--color", default="#e5bc36")
        camera.add_argument("--tolerance", type=int, default=10)
        camera.add_argument("--min-area", type=int, default=40)
        if name == "redact":
            camera.add_argument(
                "--face-model",
                type=Path,
                help="Optional YuNet ONNX path; omit for the bundled Haar cascade",
            )
            camera.add_argument("--face-threshold", type=float, default=0.65)
    train = commands.add_parser(
        "train", help="Validate before starting a YOLO training run"
    )
    train.add_argument("--task", choices=["classify", "detect", "pose"], required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--imgsz", type=int, default=224)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.add_argument("--output", type=Path, default=Path("output/training"))
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data without importing the model or training",
    )
    for name in ("predict", "evaluate"):
        model = commands.add_parser(name)
        model.add_argument("--model", type=Path, required=True)
        model.add_argument(
            "--source" if name == "predict" else "--data", type=Path, required=True
        )
        model.add_argument("--device", default="cpu")
        model.add_argument("--output", type=Path, default=Path(f"output/{name}"))
        model.add_argument("--imgsz", type=int, default=224)
        model.add_argument("--confidence", type=float, default=0.25)
    parking = commands.add_parser(
        "parking", help="Train or apply the parking-space SVM"
    )
    parking.add_argument("--data", type=Path)
    parking.add_argument("--source", type=Path)
    parking.add_argument("--model", type=Path, default=Path("output/parking.joblib"))
    parking.add_argument("--seed", type=int, default=42)
    parking.add_argument("--split", choices=["sequence", "random"], default="sequence")
    parking.add_argument(
        "--limit", type=int, default=0, help="Optional images per class; 0 uses all"
    )
    return cli


def run(args):
    if args.command == "train":
        if min(args.epochs, args.imgsz, args.batch) < 1:
            raise ValueError("Epochs, image size, and batch size must be positive")
        if args.task == "classify":
            report = classification_dataset(args.data)
            data = str(args.data.resolve())
        else:
            data, report = yolo_dataset(args.data, args.task)
        print(
            json.dumps(
                {"task": args.task, "dataset": report, "seed": args.seed}, indent=2
            )
        )
        if args.dry_run:
            return
        from .models import load_model, local_checkpoints

        if isinstance(data, dict):
            import yaml

            args.output.mkdir(parents=True, exist_ok=True)
            config = args.output.resolve() / "dataset.yaml"
            config.write_text(yaml.safe_dump(data, sort_keys=False))
            data = str(config)
        pretrained = {
            "classify": "yolov8n-cls.pt",
            "detect": "yolov8n.pt",
            "pose": "yolov8n-pose.pt",
        }
        model = load_model(args.model or pretrained[args.task])
        if model.task != args.task:
            raise ValueError(
                f"The checkpoint task is {model.task}, but --task is {args.task}"
            )
        with local_checkpoints():
            model.train(
                data=data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                seed=args.seed,
                deterministic=True,
                workers=0,
                device=args.device,
                project=str(args.output.resolve()),
                name="run",
                exist_ok=False,
            )
        return
    if args.command in ("predict", "evaluate"):
        if not args.model.is_file():
            raise ValueError(f"Model does not exist: {args.model}")
        from .models import load_model

        model = load_model(args.model)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == "evaluate":
            if model.task == "classify":
                classification_dataset(args.data, ("test",))
                data = str(args.data.resolve())
            else:
                raise ValueError(
                    "Use the task-specific YOLO validation command with a separately annotated test split for detection/pose."
                )
            metrics = model.val(
                data=data, split="test", imgsz=args.imgsz, device=args.device, workers=0
            )
            (args.output / "metrics.json").write_text(
                json.dumps(metrics.results_dict, indent=2)
            )
            return
        if not args.source.exists():
            raise ValueError(f"Input does not exist: {args.source}")
        destination = args.output / "predictions.jsonl"
        if destination.exists():
            raise ValueError(
                f"Prediction output already exists: {destination}. Choose a new --output directory."
            )
        written = 0
        for result in model.predict(
            source=str(args.source.resolve()),
            imgsz=args.imgsz,
            conf=args.confidence,
            device=args.device,
            stream=True,
            verbose=False,
        ):
            row = {"source": Path(result.path).name}
            if result.probs is not None:
                scores = result.probs.data.cpu().tolist()
                row["predictions"] = sorted(
                    [
                        {"label": result.names[i], "score": float(p)}
                        for i, p in enumerate(scores)
                    ],
                    key=lambda p: p["score"],
                    reverse=True,
                )
            if result.boxes is not None:
                row["boxes"] = [
                    {
                        "xyxy": box[:4],
                        "score": box[4],
                        "label": result.names[int(box[5])],
                    }
                    for box in result.boxes.data.cpu().tolist()
                ]
            if result.keypoints is not None:
                row["keypoints"] = result.keypoints.data.cpu().tolist()
            # Bound memory for video inference: save one JSON line per frame.
            with destination.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if (
                written == 0
                and args.source.is_file()
                and args.source.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            ):
                result.save(filename=str(args.output / "prediction.jpg"))
            written += 1
        print(f"Predictions written to {args.output.resolve()}")
        return
    if args.command == "parking":
        from .parking import train, train_sequence, predict

        if args.source:
            print(json.dumps(predict(args.model, args.source), indent=2))
        elif args.data:
            trainer = train_sequence if args.split == "sequence" else train
            print(
                json.dumps(
                    trainer(args.data, args.model, args.seed, args.limit), indent=2
                )
            )
        else:
            raise ValueError("Supply --data to train or --source to predict")
        return
    from . import imaging
    import cv2

    if args.command == "image":
        result = imaging.effect(imaging.read_image(args.source), args.mode, args.amount)
        imaging.write_image(args.output, result)
        if args.show:
            try:
                cv2.imshow("Result", result)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()
        return
    detector = (
        imaging.face_detector(args.face_model, args.face_threshold)
        if args.command == "redact"
        else None
    )
    color = args.color.lstrip("#")
    if len(color) != 6:
        raise ValueError("Color must be a six-digit hex color, such as #e5bc36")
    bgr = tuple(int(color[i : i + 2], 16) for i in (4, 2, 0))

    def transform(frame):
        if detector is not None:
            return imaging.redact(frame, imaging.faces(frame, detector))
        _, boxes = imaging.color_mask(frame, bgr, args.tolerance, args.min_area)
        result = frame.copy()
        for box in boxes:
            x, y, w, h = [box[key] for key in ("x", "y", "width", "height")]
            cv2.rectangle(result, (x, y), (x + w, y + h), (120, 220, 160), 2)
        return result

    if Path(args.source).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        imaging.write_image(
            args.output or Path(f"output/{args.command}.png"),
            transform(imaging.read_image(args.source)),
        )
    else:
        if args.headless and not args.output:
            raise ValueError("Headless video processing needs --output")
        print(
            f"Processed {imaging.video_loop(args.source, transform, args.output, not args.headless)} frames"
        )


def main(argv=None):
    cli = parser()
    try:
        run(cli.parse_args(argv))
    except (ValueError, OSError, ImportError) as error:
        cli.exit(2, f"visionlab: {error}\n")


if __name__ == "__main__":
    main()
