import {readFile,writeFile} from 'node:fs/promises';import {PNG} from 'pngjs';
import {findRectangles} from '../web/rectangles.js';
const directory=new URL('../output/evaluation/'+(process.argv[2]||'stress')+'/',import.meta.url),manifest=JSON.parse(await readFile(new URL('manifest.json',directory))),records=[];
for(const item of manifest.cases.filter(c=>c.kind==='rectangle')){
 const png=PNG.sync.read(await readFile(new URL(item.file,directory))),actual=findRectangles(png.data,png.width,png.height);const matched=new Set();let tp=0;
 for(const points of item.truth){const x=Math.min(...points.map(p=>p[0])),y=Math.min(...points.map(p=>p[1])),w=Math.max(...points.map(p=>p[0]))-x,h=Math.max(...points.map(p=>p[1]))-y;let best=-1,score=0;
  actual.forEach((b,i)=>{const inter=Math.max(0,Math.min(x+w,b.x+b.width)-Math.max(x,b.x))*Math.max(0,Math.min(y+h,b.y+b.height)-Math.max(y,b.y));const iou=inter/(w*h+b.width*b.height-inter);if(!matched.has(i)&&iou>score){score=iou;best=i;}});
  if(score>=.7){tp++;matched.add(best);}
 }
 records.push({...item,actual,tp,fp:actual.length-tp,fn:item.truth.length-tp});
}
const counts=records.reduce((s,r)=>({tp:s.tp+r.tp,fp:s.fp+r.fp,fn:s.fn+r.fn}),{tp:0,fp:0,fn:0});const report={cases:records.length,...counts,precision:counts.tp/(counts.tp+counts.fp),recall:counts.tp/(counts.tp+counts.fn),records};await writeFile(new URL('rectangle-results.json',directory),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({...report,records:undefined}));
