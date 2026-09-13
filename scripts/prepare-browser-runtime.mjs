import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const source = 'node_modules/onnxruntime-web';
const {version} = JSON.parse(await readFile(`${source}/package.json`, 'utf8'));
const destination = `web/assets/runtime/onnx-${version}`;
await mkdir(destination, {recursive:true});
const files=[];
for (const file of ['ort.wasm.min.mjs','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm']) {
 const bytes=await readFile(`${source}/dist/${file}`);
 await cp(`${source}/dist/${file}`,`${destination}/${file}`);
 files.push({file:`onnx-${version}/${file}`,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
}
for (const [remote,local] of [['LICENSE','ONNX-RUNTIME-LICENSE'],['ThirdPartyNotices.txt','ONNX-THIRD-PARTY-NOTICES.txt']]) {
 const response=await fetch(`https://raw.githubusercontent.com/microsoft/onnxruntime/v${version}/${remote}`);
 if(!response.ok)throw new Error('Unable to fetch pinned runtime license');
 await writeFile(`web/assets/runtime/${local}`,await response.text());
}
await writeFile('web/assets/runtime/manifest.json',JSON.stringify({version,files},null,2)+'\n');
console.log(`Prepared pinned ONNX Runtime ${version}`);
