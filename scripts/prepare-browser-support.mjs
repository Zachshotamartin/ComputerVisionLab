// Reproduce the explicitly pinned third-party face model and public-domain example.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const revision='47534e27c9851bb1128ccc0102f1145e27f23f98';
const manifest=JSON.parse(await readFile('web/assets/models/manifest.json','utf8'));
async function download(url,file){const response=await fetch(url);if(!response.ok)throw new Error(`Could not download ${url}`);const bytes=Buffer.from(await response.arrayBuffer());await writeFile(file,bytes);return bytes;}
await mkdir('web/assets/model-examples',{recursive:true});
const url=`https://media.githubusercontent.com/media/opencv/opencv_zoo/${revision}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`;
const response=await fetch(url);if(!response.ok)throw new Error('Could not download pinned YuNet model');
const bytes=Buffer.from(await response.arrayBuffer()),sha256=createHash('sha256').update(bytes).digest('hex'),file=`face-${sha256.slice(0,12)}.onnx`;
await writeFile('web/assets/models/'+file,bytes);
manifest.models.face={file,bytes:bytes.length,sha256,kind:'face',size:640,classes:['face'],preprocess:'letterbox-bgr-0-255',source:`https://github.com/opencv/opencv_zoo/tree/${revision}/models/face_detection_yunet`};
await writeFile('web/assets/models/manifest.json',JSON.stringify(manifest,null,2)+'\n');
await download(`https://raw.githubusercontent.com/opencv/opencv_zoo/${revision}/models/face_detection_yunet/LICENSE`,'web/assets/models/YUNET-LICENSE.txt');
await download('https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.150/LICENSE','web/assets/models/YOLO-LICENSE.txt');
await download('https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png','web/assets/model-examples/face.png');
console.log('Prepared pinned face model, licenses, and NASA example.');
