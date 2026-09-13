import {chromium} from '@playwright/test';
import {readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const root=new URL('../',import.meta.url),directory=new URL('output/evaluation/stress/',root),manifest=JSON.parse(await readFile(new URL('manifest.json',directory)));
const browser=await chromium.launch();const page=await browser.newPage();await page.goto(process.env.VISION_EVALUATION_URL||'http://127.0.0.1:4190/');
const records=[];
for(const item of manifest.cases){
 const png=await readFile(new URL(item.file,directory));
 const actual=await page.evaluate(async({png,kind,moduleBase})=>{
  const picture=new Image();picture.src='data:image/png;base64,'+png;await picture.decode();
  const canvas=document.createElement('canvas');canvas.width=picture.naturalWidth;canvas.height=picture.naturalHeight;const ctx=canvas.getContext('2d');ctx.drawImage(picture,0,0);const {data,width,height}=ctx.getImageData(0,0,canvas.width,canvas.height);
  if(kind==='face'){const {inferModel}=await import(moduleBase+'modelEngine.js');return (await inferModel({data:Uint8ClampedArray.from(data),width,height},'face',{assetBase:'/',threshold:.65})).boxes;}
  const {findRectangles}=await import(moduleBase+'rectangles.js');return findRectangles(Uint8ClampedArray.from(data),width,height);
 },{png:png.toString('base64'),kind:item.kind,moduleBase:'/@fs'+fileURLToPath(new URL('web/',root))});
 const boxes=item.truth.map(truth=>item.kind==='face'?truth:[Math.min(...truth.map(p=>p[0])),Math.min(...truth.map(p=>p[1])),Math.max(...truth.map(p=>p[0]))-Math.min(...truth.map(p=>p[0])),Math.max(...truth.map(p=>p[1]))-Math.min(...truth.map(p=>p[1]))]);
 function iou(a,b){const inter=Math.max(0,Math.min(a[0]+a[2],b.x+b.width)-Math.max(a[0],b.x))*Math.max(0,Math.min(a[1]+a[3],b.y+b.height)-Math.max(a[1],b.y));return inter/(a[2]*a[3]+b.width*b.height-inter);}
 const matched=new Set();let tp=0;
 for(const truth of boxes){let best=-1,score=0;actual.forEach((box,i)=>{const overlap=iou(truth,box);if(!matched.has(i)&&overlap>score){best=i;score=overlap;}});if(score>=(item.kind==='face'?.3:.7)){tp++;matched.add(best);}}
 records.push({...item,actual,tp,fn:boxes.length-tp,fp:actual.length-tp});
 if(records.length%20===0)console.log('Evaluated',records.length,'/',manifest.cases.length);
}
await browser.close();
const summary={};for(const kind of ['face','rectangle']){const cases=records.filter(r=>r.kind===kind),tp=cases.reduce((s,r)=>s+r.tp,0),fp=cases.reduce((s,r)=>s+r.fp,0),fn=cases.reduce((s,r)=>s+r.fn,0);summary[kind]={cases:cases.length,tp,fp,fn,precision:tp/(tp+fp),recall:tp/(tp+fn),failures:cases.filter(r=>r.fn||r.fp).map(r=>({file:r.file,condition:r.condition,fn:r.fn,fp:r.fp}))};}
await writeFile(new URL('output/evaluation/browser-geometry-evaluation.json',root),JSON.stringify({scope:manifest.face_ground_truth+' '+manifest.geometry_ground_truth,summary,records},null,2)+'\n');console.log(JSON.stringify(summary,null,2));
