"""Publish three explicitly identified validation frames without changing the split."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
split = ROOT / 'output/evaluation/splits/tiger'
source_manifest = json.loads((split / 'split-manifest.json').read_text())
manifest_path = ROOT / 'web/assets/models/manifest.json'
manifest = json.loads(manifest_path.read_text())
examples = []
# Fixed, evenly spaced validation frames; no selection by prediction quality.
for frame in [170, 190, 210]:
    record = next(row for row in source_manifest['frames'] if row['frame'] == frame)
    if record['split'] != 'val':
        raise ValueError(f'Frame {frame} is not in validation')
    source = split / 'images/val' / Path(record['path']).name
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != record['sha256']:
        raise ValueError(f'Frame {frame} does not match the frozen source')
    filename = f'tiger-validation-{frame}.jpg'
    shutil.copyfile(source, ROOT / 'web/assets/model-examples' / filename)
    examples.append({'file': filename, 'label': f'Frame {frame}', 'partition': 'validation',
                     'source': record['path'], 'sha256': digest, 'bytes': source.stat().st_size})
manifest['examples']['tiger'] = [{key: item[key] for key in ['file', 'label', 'source']} for item in examples]
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
provenance = {'dataset': source_manifest['source'], 'source_video': source_manifest['source_video'],
              'split_manifest_sha256': hashlib.sha256((split / 'split-manifest.json').read_bytes()).hexdigest(),
              'selection': 'Frames 170, 190, 210: evenly spaced within the validation partition, used for checkpoint selection. These are demonstration inputs, not new test evidence.',
              'rights': 'Dataset annotations are AGPL-3.0; source-video rights remain with the original owner. No blanket license is asserted over the photographs.',
              'examples': examples}
(ROOT / 'docs/verification/tiger-examples.json').write_text(json.dumps(provenance, indent=2) + '\n')
print('Prepared', len(examples), 'validation examples')
