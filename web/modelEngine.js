import manifest from './assets/models/manifest.json' with { type: 'json' };
import runtime from './assets/runtime/manifest.json' with { type: 'json' };
import {validateImage,classificationTensor,letterboxTensor,parkingPrediction,decodeAlpacas,decodeFaces,decodePose} from './modelMath.js';

import {decodeParkingModel} from './parkingFormat.js';

const sessions=new Map();
let runtimePromise;
async function readModel(base,key,progress) {
 const model=manifest.models[key];
 if(!model)throw new Error('Unknown model.');
 progress?.(`Loading ${key==='oct'?'retinal OCT':key} model (${(model.bytes/1048576).toFixed(1)} MB)…`);
 const response=await fetch(new URL('models/'+model.file,base),{cache:'force-cache'});
 if(!response.ok)throw new Error('The model could not download. Check your connection and try again.');
 const buffer=await response.arrayBuffer();
 if(buffer.byteLength!==model.bytes)throw new Error('The model download was incomplete. Try again.');
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
 if(hash!==model.sha256)throw new Error('The downloaded model did not pass its integrity check. Reload and try again.');
 return buffer;
}
async function getRuntime(base) {
 if(!runtimePromise)runtimePromise=(async()=>{
  const directory=new URL(`runtime/onnx-${runtime.version}/`,base).href;
  const ort=await import(/* @vite-ignore */ directory+'ort.wasm.min.mjs');
  ort.env.wasm.numThreads=1;
  ort.env.wasm.wasmPaths=directory;
  ort.env.wasm.proxy=false;
  return ort;
 })().catch(error=>{runtimePromise=null;throw error;});
 return runtimePromise;
}
async function sessionFor(base,key,progress) {
 const cacheKey=base+key;
 if(sessions.has(cacheKey)){const cached=sessions.get(cacheKey);sessions.delete(cacheKey);sessions.set(cacheKey,cached);return cached;}
 // A small LRU bounds the memory used by switching between models.
 if(sessions.size>=2){const oldest=sessions.keys().next().value;const previous=await sessions.get(oldest);await previous.session?.release();sessions.delete(oldest);}
 const task=(async()=>{
  const bytes=await readModel(base,key,progress);
  if(key==='parking')return {model:manifest.models.parking.file.endsWith('.bin')?decodeParkingModel(bytes):JSON.parse(new TextDecoder().decode(bytes))};
  progress?.('Preparing local inference…');
  const ort=await getRuntime(base);
  const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  return {session,ort};
 })();
 sessions.set(cacheKey,task);
 try{return await task;}catch(error){sessions.delete(cacheKey);throw error;}
}
export async function inferModel(image,key,{assetBase,threshold=.25,progress}={}) {
 validateImage(image);
 const base=new URL(assetBase||'/assets/computer-vision/',self.location.origin).href;
 const meta=manifest.models[key];
 if(!meta)throw new Error('Unknown model.');
 const loaded=await sessionFor(base,key,progress);
 progress?.('Running on this image…');
 const start=performance.now();
 if(key==='parking')return {...parkingPrediction(image,loaded.model),kind:'svm',milliseconds:performance.now()-start};
 const {session,ort}=loaded;
 const geometry=meta.kind==='classify'?null:letterboxTensor(image,meta.size,{bgr:key==='face'});
 const data=geometry?geometry.tensor:classificationTensor(image,meta.size);
 const tensor=new ort.Tensor('float32',data,[1,3,meta.size,meta.size]);
 let outputs;
 try{
  outputs=await session.run({[session.inputNames[0]]:tensor});
  if(meta.kind==='classify'){
   const values=outputs[session.outputNames[0]].data;
   const scores=meta.classes.map((label,index)=>({label,score:values[index]})).sort((a,b)=>b.score-a.score);
   return {kind:'classify',label:scores[0].label,scores,milliseconds:performance.now()-start};
  }
  const boxes=key==='face'?decodeFaces(outputs,geometry,meta.size,threshold):meta.kind==='pose'?decodePose(outputs[session.outputNames[0]].data,geometry,meta.keypoints,threshold):decodeAlpacas(outputs[session.outputNames[0]].data,geometry,threshold);
  return {kind:meta.kind,boxes,milliseconds:performance.now()-start};
 }finally{tensor.dispose();if(outputs)Object.values(outputs).forEach(output=>output.dispose());}
}
