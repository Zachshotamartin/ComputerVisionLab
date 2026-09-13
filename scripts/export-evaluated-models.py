"""Export explicitly selected candidates, preserve older artifacts, and check tensor parity."""

import argparse, hashlib, json, shutil, sys, struct
from pathlib import Path
import numpy as np, torch, onnx, onnxruntime as ort, joblib, cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from visionlab.models import load_model
from ultralytics.data.augment import classify_transforms

p = argparse.ArgumentParser()
p.add_argument("--selection", type=Path, required=True)
args = p.parse_args()
selection = json.loads(args.selection.read_text())
output = ROOT / "web/assets/models"
manifest_path = output / "manifest.json"
manifest = json.loads(manifest_path.read_text())
evidence = json.loads((ROOT / "docs/verification/browser-export.json").read_text())
evidence.update(selection=selection, examples={})
# Keep the original manifest/evidence for reproducibility, as well as its model files.
archive = ROOT / "docs/verification/original-browser"
archive.mkdir(parents=True, exist_ok=True)
for source in [manifest_path, ROOT / "docs/verification/browser-export.json"]:
    destination = archive / source.name
    if not destination.exists():
        shutil.copyfile(source, destination)


def record(path, key, source):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    file = f"{key}-{digest[:12]}{path.suffix}"
    shutil.copyfile(path, output / file)
    return {
        "file": file,
        "bytes": len(raw),
        "sha256": digest,
        "source": source,
        "revision": "grouped-evaluation-2026-09-12",
    }


for key, entry in selection["models"].items():
    path = (ROOT / entry["checkpoint"]).resolve()
    size = entry["size"]
    kind = entry["kind"]
    training_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        manifest["models"].get(key, {}).get("trainingSha") == training_sha
        and manifest["models"][key]["size"] == size
        and (kind != "svm" or manifest["models"][key]["file"].endswith(".bin"))
    ):
        print("Keeping verified export", key, flush=True)
        continue
    if kind == "svm":
        payload = joblib.load(path)
        scaler, svm = payload["pipeline"].steps[0][1], payload["pipeline"].steps[1][1]
        # SVC support vectors originate from float32 preprocessing. Assert that
        # packing them as float32 is lossless rather than silently quantizing weights.
        support = svm.support_vectors_.astype("<f4")
        if not np.array_equal(support.astype(np.float64), svm.support_vectors_):
            raise ValueError("Support vectors require higher precision")
        header = {
            "classes": payload["classes"],
            "size": 15,
            "features": len(scaler.mean_),
            "supportCount": len(support),
            "intercept": float(svm.intercept_[0]),
            "gamma": float(svm._gamma),
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        exported = ROOT / "output/evaluation/parking-export.bin"
        exported.write_bytes(
            b"VLS1"
            + struct.pack("<I", len(encoded))
            + encoded
            + scaler.mean_.astype("<f8").tobytes()
            + scaler.scale_.astype("<f8").tobytes()
            + support.tobytes()
            + svm.dual_coef_[0].astype("<f8").tobytes()
        )
        meta = record(exported, key, entry["checkpoint"])
        meta.update(
            trainingSha=training_sha,
            kind="svm",
            size=15,
            classes=payload["classes"],
            preprocess="area-rgb-0-1",
        )
        manifest["models"][key] = meta
        evidence[key] = {
            "support_vectors": len(svm.support_vectors_),
            "features": len(scaler.mean_),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        continue
    model = load_model(path)
    model.ckpt_path = str(ROOT / "output/evaluation" / f"{key}-browser.pt")
    model.model.pt_path = model.ckpt_path
    exported = Path(
        model.export(
            format="onnx",
            imgsz=size,
            opset=17,
            simplify=False,
            dynamic=False,
            device="cpu",
            batch=1,
        )
    )
    onnx.checker.check_model(onnx.load(exported))
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    tensor = np.random.default_rng(42).random((1, 3, size, size)).astype(np.float32)
    with torch.no_grad():
        reference = model.model(torch.from_numpy(tensor))
        reference = (
            reference[0] if isinstance(reference, tuple) else reference
        ).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    delta = float(np.max(np.abs(reference - actual)))
    if not np.allclose(reference, actual, rtol=1e-3, atol=1e-3):
        raise RuntimeError(f"{key}: parity failed {delta}")
    meta = record(exported, key, entry["checkpoint"])
    meta.update(
        trainingSha=training_sha,
        kind=kind,
        size=size,
        classes=list(model.names.values()),
        preprocess="center-crop-rgb-0-1" if kind == "classify" else "letterbox-rgb-0-1",
    )
    if kind == "pose":
        meta.update(
            keypoints=selection["pose"]["keypoints"],
            skeleton=selection["pose"]["skeleton"],
        )
    manifest["models"][key] = meta
    evidence[key] = {
        "max_absolute_error": delta,
        "input_shape": list(tensor.shape),
        "output_shape": list(actual.shape),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    del session, model
# Recheck the actual shipped examples independently in Python.
for key in ["weather", "oct", "parking"]:
    meta = manifest["models"][key]
    if key != "parking":
        session = ort.InferenceSession(
            str(output / meta["file"]), providers=["CPUExecutionProvider"]
        )
    else:
        payload = joblib.load(ROOT / selection["models"][key]["checkpoint"])
    for item in manifest["examples"][key]:
        image = Image.open(ROOT / "web/assets/model-examples" / item["file"]).convert(
            "RGB"
        )
        if key == "parking":
            vector = (
                cv2.resize(np.asarray(image), (15, 15), interpolation=cv2.INTER_AREA)
                .astype(np.float32)
                .reshape(1, -1)
                / 255
            )
            evidence["examples"][item["file"]] = {
                "decision_margin": float(
                    payload["pipeline"].decision_function(vector)[0]
                ),
                "class": payload["classes"][
                    int(payload["pipeline"].predict(vector)[0])
                ],
            }
        else:
            tensor = classify_transforms(meta["size"])(image).unsqueeze(0).numpy()
            probs = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
            evidence["examples"][item["file"]] = {
                "probabilities": probs.tolist(),
                "top_class": meta["classes"][int(probs.argmax())],
            }
if "tiger" in manifest["models"]:
    manifest["examples"]["tiger"] = []
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
(ROOT / "docs/verification/browser-export.json").write_text(
    json.dumps(evidence, indent=2) + "\n"
)
print("Verified exports:", list(selection["models"]), flush=True)
