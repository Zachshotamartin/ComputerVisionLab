"""Export the user's trusted archive checkpoints and verify numeric parity.
Run with the archive's Python environment and --archive /path/to/computervision.
Original checkpoints and datasets are read only.
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from PIL import Image, ImageOps, ImageDraw
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from visionlab.models import load_model
from ultralytics.data.augment import classify_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=Path, required=True)
args = parser.parse_args()
archive = args.archive.resolve()
output = ROOT / 'web/assets/models'
examples = ROOT / 'web/assets/model-examples'
output.mkdir(parents=True, exist_ok=True)
examples.mkdir(parents=True, exist_ok=True)
manifest = {'version': 1, 'models': {}, 'sources': {}}
evidence = {}
paths = {
 'weather': 'imageclassification/weather_classifier/runs/classify/train/weights/best.pt',
 'oct': 'imageclassification/pneumonia_classifier/runs/classify/train/weights/best.pt',
 'alpaca': 'objectdetection/alpacadetector/train-yolov8-custom-dataset-step-by-step-guide/runs/detect/train3/weights/best.pt',
}

def record(path, prefix, source):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    filename = f'{prefix}-{digest[:12]}{path.suffix}'
    shutil.copyfile(path, output / filename)
    return {'file': filename, 'bytes': len(raw), 'sha256': digest, 'source': source}

for key, relative in paths.items():
    model = load_model(archive / relative)
    size = 640 if key == 'alpaca' else 64
    # Export to the generated output directory, never into the original archive.
    model.ckpt_path = str(ROOT / 'output' / f'{key}.pt')
    model.model.pt_path = model.ckpt_path
    exported = Path(model.export(format='onnx', imgsz=size, opset=17, simplify=False, dynamic=False, device='cpu', batch=1))
    graph = onnx.load(exported)
    onnx.checker.check_model(graph)
    meta = record(exported, key, relative)
    meta.update({'kind': 'detect' if key == 'alpaca' else 'classify', 'size': size, 'classes': list(model.names.values()), 'preprocess': 'letterbox-rgb-0-1' if key == 'alpaca' else 'center-crop-rgb-0-1'})
    session = ort.InferenceSession(str(exported), providers=['CPUExecutionProvider'])
    # Same input tensor in PyTorch and ONNX checks the actual exported weights.
    tensor = np.random.default_rng(42).random((1, 3, size, size)).astype(np.float32)
    with torch.no_grad():
        reference = model.model(torch.from_numpy(tensor))
        if isinstance(reference, tuple): reference = reference[0]
        reference = reference.numpy()
    actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    delta = float(np.max(np.abs(reference - actual)))
    if not np.allclose(reference, actual, rtol=1e-3, atol=1e-3):
        raise RuntimeError(f'{key}: export parity failed, max error {delta}')
    evidence[key] = {'max_absolute_error': delta, 'input_shape': list(tensor.shape), 'output_shape': list(actual.shape), 'source_sha256': hashlib.sha256((archive / relative).read_bytes()).hexdigest()}
    manifest['models'][key] = meta

payload = joblib.load(archive / 'output/parking.joblib')
scaler, svm = payload['pipeline'].steps[0][1], payload['pipeline'].steps[1][1]
parking = {'classes': payload['classes'], 'size': payload['image_size'], 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'support': svm.support_vectors_.tolist(), 'coefficients': svm.dual_coef_[0].tolist(), 'intercept': float(svm.intercept_[0]), 'gamma': float(svm._gamma)}
parking_file = ROOT / 'output/parking.json'
parking_file.write_text(json.dumps(parking, separators=(',', ':')))
manifest['models']['parking'] = {**record(parking_file, 'parking', 'output/parking.joblib'), 'kind': 'svm', 'size': 15, 'classes': payload['classes'], 'preprocess': 'area-rgb-0-1'}
evidence['parking'] = {'support_vectors': len(svm.support_vectors_), 'features': len(scaler.mean_), 'evaluation': payload['report']}

# Small representative inputs retain original relative-path provenance.
fixtures = {}
for key, folder, classes in [
 ('weather', archive / 'imageclassification/weather_classifier/data/val', manifest['models']['weather']['classes']),
 ('oct', archive / 'imageclassification/pneumonia_classifier/data/test', manifest['models']['oct']['classes']),
 ('parking', archive / 'imageclassification/parking_classifier/data', ['empty', 'not_empty']),
]:
    fixtures[key] = []
    for label in classes:
        choices = sorted(p for p in (folder / label).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not choices: raise RuntimeError(f'No sample for {key}/{label}')
        source = choices[0]
        image = ImageOps.contain(Image.open(source).convert('RGB'), (768, 512))
        name = f'{key}-{label.lower()}.png'
        image.save(examples / name)
        fixtures[key].append({'file': name, 'label': label, 'source': str(source.relative_to(archive))})
        if key in ['weather', 'oct']:
            tensor = classify_transforms(64)(image).unsqueeze(0).numpy()
            session = ort.InferenceSession(str(output / manifest['models'][key]['file']), providers=['CPUExecutionProvider'])
            probs = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
            evidence.setdefault('examples', {})[name] = {'probabilities': probs.tolist(), 'top_class': manifest['models'][key]['classes'][int(probs.argmax())]}
            # Golden preprocessing pixels verify the JS center-crop pipeline.
            np.round(tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8).tofile(ROOT / 'output' / f'{name}.rgb')
        else:
            import cv2
            rgb = np.asarray(image)
            vector = cv2.resize(rgb, (15, 15), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(1, -1) / 255
            evidence.setdefault('examples', {})[name] = {'decision_margin': float(payload['pipeline'].decision_function(vector)[0]), 'class': payload['classes'][int(payload['pipeline'].predict(vector)[0])]}

alpaca_source = archive / 'objectdetection/alpacadetector/train-yolov8-custom-dataset-step-by-step-guide/data/images/test/ec522d35b7604195.jpg'
ImageOps.contain(Image.open(alpaca_source).convert('RGB'), (768, 512)).save(examples / 'alpaca.png')
fixtures['alpaca'] = [{'file': 'alpaca.png', 'label': 'Alpacas', 'source': str(alpaca_source.relative_to(archive))}]
manifest['examples'] = fixtures
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(ROOT / 'docs/verification/browser-export.json').write_text(json.dumps(evidence, indent=2) + '\n')
print('Exported models and verified PyTorch/ONNX parity:', {k: v['bytes'] for k, v in manifest['models'].items()})

rectangle = Image.new('RGB', (768, 512), '#182c25')
draw = ImageDraw.Draw(rectangle)
draw.polygon([(80,120),(340,75),(360,360),(110,395)], fill='#d0c29e')
draw.polygon([(445,95),(705,145),(670,370),(415,320)], fill='#dca486')
rectangle.save(ROOT / 'web/assets/rectangles.png')
