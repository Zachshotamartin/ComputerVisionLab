"""Prepare a bounded animal-pose demonstration from attributed public annotations."""

import hashlib, json, re, shutil, argparse, urllib.request, zipfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/evaluation"
source = OUT / "downloads"
dest = OUT / "splits/tiger"
cli = argparse.ArgumentParser()
cli.add_argument("--download", action="store_true")
args = cli.parse_args()
expected = "834dd66fa14897d563d35d67ca7988af9c2331edaa8c8c96f47cd8847fe4a405"
if args.download:
    source.mkdir(parents=True, exist_ok=True)
    archive = source / "tiger-pose.zip"
    if not archive.exists():
        temporary = source / "tiger-pose.zip.partial"
        with (
            urllib.request.urlopen(
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/tiger-pose.zip",
                timeout=30,
            ) as response,
            temporary.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError(
                "Downloaded tiger dataset does not match the pinned checksum"
            )
        temporary.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Tiger dataset checksum mismatch")
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if (
                not (source / member.filename)
                .resolve()
                .is_relative_to(source.resolve())
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe archive entry")
        package.extractall(source)
if dest.exists():
    raise SystemExit("Tiger split already frozen")
config = yaml.safe_load((source / "tiger-pose.yaml").read_text())
rows = []
for path in sorted((source / "images").rglob("*.jpg")):
    number = int(re.search(r"(\d+)", path.stem).group(1))
    label = source / "labels" / path.parent.name / (path.stem + ".txt")
    split = (
        "train"
        if number <= 150
        else "val"
        if 170 <= number <= 210
        else "test"
        if number >= 230
        else "embargo"
    )
    row = {
        "path": str(path.relative_to(source)),
        "frame": number,
        "split": split,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    rows.append(row)
    if split == "embargo":
        continue
    for file, folder in [(path, "images"), (label, "labels")]:
        target = dest / folder / split / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
config.update(
    path=str(dest), train="images/train", val="images/val", test="images/test"
)
config.pop("download", None)
(dest / "data.yaml").write_text(yaml.safe_dump(config))
report = {
    "source": "https://github.com/ultralytics/assets/releases/download/v0.0.0/tiger-pose.zip",
    "archive_sha256": hashlib.sha256(
        (source / "tiger-pose.zip").read_bytes()
    ).hexdigest(),
    "annotation_license": "AGPL-3.0",
    "source_video": "https://youtu.be/Gc6K5eKrTNQ",
    "redistribution": "Original video terms apply; no downloaded frames are shipped in the public repository.",
    "scope": "New 12-keypoint tiger prototype; not restoration of the missing 39-keypoint quadruped model. All images are frames of the same source video, so held-out frames do not establish unseen-animal or unseen-scene performance.",
    "grouping": "Chronological frame blocks with 19-frame embargoes",
    "counts": {
        s: sum(r["split"] == s for r in rows)
        for s in ["train", "val", "test", "embargo"]
    },
    "frames": rows,
}
(dest / "split-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(report["counts"])
