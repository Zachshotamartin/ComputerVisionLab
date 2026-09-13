import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {PNG} from 'pngjs';
import {classificationTensor,parkingPrediction,decodeAlpacas,decodeFaces,suppress,validateImage} from '../modelMath.js';
import {findRectangles} from '../rectangles.js';
const asset=new URL('../assets/',import.meta.url);
const manifest=JSON.parse(readFileSync(new URL('models/manifest.json',asset)));
const evidence=JSON.parse(readFileSync(new URL('../../docs/verification/browser-export.json',import.meta.url)));
const png=name=>PNG.sync.read(readFileSync(new URL('model-examples/'+name,asset)));

test('every shipped model and runtime matches its pinned byte/hash manifest',()=>{
 for(const model of Object.values(manifest.models)){
  const bytes=readFileSync(new URL('models/'+model.file,asset));assert.equal(bytes.length,model.bytes);assert.equal(createHash('sha256').update(bytes).digest('hex'),model.sha256);
 }
 const runtime=JSON.parse(readFileSync(new URL('runtime/manifest.json',asset)));
 for(const file of runtime.files){const bytes=readFileSync(new URL('runtime/'+file.file,asset));assert.equal(bytes.length,file.bytes);assert.equal(createHash('sha256').update(bytes).digest('hex'),file.sha256);}
});

test('parking decision margins match the repaired Python pipeline on both classes',()=>{
 const model=JSON.parse(readFileSync(new URL('models/'+manifest.models.parking.file,asset)));
 for(const name of ['parking-empty.png','parking-not_empty.png']){const actual=parkingPrediction(png(name),model),expected=evidence.examples[name];assert.equal(actual.label,expected.class);assert.ok(Math.abs(actual.margin-expected.decision_margin)<.002,`${name}: ${actual.margin}`);}
});

test('classification preprocessing center-crops and writes normalized channel-first input',()=>{
 const data=new Uint8ClampedArray(8*4*4);
 for(let y=0;y<4;y++)for(let x=0;x<8;x++){const p=(y*8+x)*4;data[p]=x<2||x>5?255:0;data[p+1]=100;data[p+2]=200;data[p+3]=255;}
 const tensor=classificationTensor({data,width:8,height:4},4);assert.equal(tensor.length,48);assert.equal(tensor[0],0);assert.ok(Math.abs(tensor[16]-100/255)<1e-7);assert.ok(Math.abs(tensor[32]-200/255)<1e-7);
 assert.throws(()=>validateImage({data:[],width:0,height:4}));
});

test('detection decoding undoes padding and NMS removes duplicate boxes',()=>{
 const geometry={scale:2,left:0,top:100,width:200,height:100};
 const decoded=decodeAlpacas(new Float32Array([100,102,180,182,80,80,40,40,.9,.8]),geometry);
 assert.equal(decoded.length,1);assert.deepEqual({...decoded[0],score:1},{x:30,y:30,width:40,height:20,score:1,label:'alpaca'});
 assert.equal(suppress([{x:0,y:0,width:20,height:20,score:.9},{x:50,y:0,width:20,height:20,score:.8}]).length,2);
});

test('face decoding uses the YuNet stride and clamps boxes to the image',()=>{
 const outputs={};for(const stride of [8,16,32]){const n=(640/stride)**2;outputs['cls_'+stride]={data:new Float32Array(n)};outputs['obj_'+stride]={data:new Float32Array(n)};outputs['bbox_'+stride]={data:new Float32Array(n*4)};}
 outputs.cls_8.data[0]=1;outputs.obj_8.data[0]=1;outputs.bbox_8.data.set([1,1,Math.log(4),Math.log(4)]);
 const faces=decodeFaces(outputs,{scale:1,left:0,top:0,width:640,height:640});assert.equal(faces.length,1);assert.equal(faces[0].x,0);assert.ok(Math.abs(faces[0].width-24)<1e-5);
});

test('rectangle tracking finds perspective corners and rejects blank images',()=>{
 const image=PNG.sync.read(readFileSync(new URL('rectangles.png',asset)));
 const boxes=findRectangles(image.data,image.width,image.height,100);assert.equal(boxes.length,2);assert.ok(boxes.every(box=>box.corners.length===4));
 assert.deepEqual(findRectangles(new Uint8Array(128*128*4),128,128,100),[]);
});
