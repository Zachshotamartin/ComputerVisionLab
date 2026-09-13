import {test,expect} from '@playwright/test';
import {readFileSync} from 'node:fs';
const evidence=JSON.parse(readFileSync(new URL('../docs/verification/browser-export.json',import.meta.url)));
async function models(page){await page.goto('/');await page.getByRole('button',{name:'Trained models',exact:true}).click();}
async function run(page){await page.getByRole('button',{name:'Run model',exact:true}).click();await expect(page.locator('.vision-model-lab')).toHaveAttribute('data-state','complete');}
async function result(page){const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save results',exact:true}).click();return JSON.parse(readFileSync(await(await download).path()));}

test('all saved classifiers run in the browser and match Python example predictions',async({page})=>{
 await models(page);
 for(const [model,label,names] of [['weather','Weather',['Cloudy','Rain','Shine','Sunrise']],['parking','Parking',['Empty','Occupied']],['oct','Retinal OCT',['CNV','DME','DRUSEN','Normal']]]){
  await page.getByRole('group',{name:'Trained model',exact:true}).getByRole('button',{name:label,exact:true}).click();
  for(const name of names){
   await page.getByRole('group',{name:'Model examples',exact:true}).getByRole('button',{name,exact:true}).click();await run(page);const actual=await result(page);
   const key=`${model}-${name==='Occupied'?'not_empty':name.toLowerCase()}.png`,expected=evidence.examples[key];
   expect(actual.model).toBe(model);expect(actual.label).toBe(expected.top_class||expected.class);
   if(expected.probabilities){const labels=model==='weather'?['cloudy','rain','shine','sunrise']:['CNV','DME','DRUSEN','NORMAL'];for(const score of actual.scores)expect(Math.abs(score.score-expected.probabilities[labels.indexOf(score.label)])).toBeLessThan(.02);}
   else expect(Math.abs(actual.margin-expected.decision_margin)).toBeLessThan(.002);
  }
 }
});

test('alpaca boxes, cutoff controls, uploaded images and PNG export work',async({page})=>{
 await models(page);await page.getByRole('group',{name:'Trained model',exact:true}).getByRole('button',{name:'Alpacas',exact:true}).click();await run(page);
 const first=await result(page);expect(first.boxes.length).toBeGreaterThan(0);expect(first.boxes.every(box=>box.x>=0&&box.y>=0&&box.width>0)).toBe(true);
 await page.getByRole('button',{name:'Strict',exact:true}).click();expect((await result(page)).cutoff).toBe(.6);
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save annotated PNG',exact:true}).click();expect((await download).suggestedFilename()).toBe('alpaca-detections.png');
 await page.getByLabel('Upload model image',{exact:true}).setInputFiles('web/assets/model-examples/alpaca.png');await expect(page.getByRole('button',{name:'Save results',exact:true})).toHaveCount(0);await run(page);expect((await result(page)).source).toBe('alpaca.png');
});

test('face redaction and perspective rectangle tracking run on real pixels',async({page})=>{
 await page.goto('/');await page.getByRole('button',{name:'Region redaction',exact:true}).click();await page.getByRole('button',{name:'Detect faces',exact:true}).click();await page.getByRole('button',{name:'Try face example',exact:true}).click();
 await expect(page.locator('.vision-readout')).toContainText('1 region');
 const canvases=page.locator('.vision-images canvas');expect(await canvases.first().evaluate(c=>c.toDataURL())).not.toBe(await canvases.last().evaluate(c=>c.toDataURL()));
 await page.getByRole('button',{name:'Rectangle tracking',exact:true}).click();await page.getByRole('button',{name:'Try rectangle example',exact:true}).click();await expect(page.locator('.vision-readout')).toContainText('2 regions');
});

test('model cancellation prevents stale results and download errors can be retried',async({page})=>{
 await models(page);
 let release;const pending=new Promise(resolve=>{release=resolve;});await page.route('**/models/weather-*.onnx',async route=>{await pending;await route.continue();});
 const downloading=page.waitForRequest('**/models/weather-*.onnx');await page.getByRole('button',{name:'Run model',exact:true}).click();await downloading;await expect(page.getByRole('button',{name:'Cancel',exact:true})).toBeVisible();await page.getByRole('button',{name:'Cancel',exact:true}).click();await page.unrouteAll({behavior:'ignoreErrors'});release();
 await expect(page.locator('.vision-model-lab')).toHaveAttribute('data-state','idle');await expect(page.getByRole('button',{name:'Save results',exact:true})).toHaveCount(0);
 await page.route('**/models/weather-*.onnx',route=>route.fulfill({status:503,body:'Unavailable'}));await page.getByRole('button',{name:'Run model',exact:true}).click();await expect(page.getByRole('alert')).toContainText('could not download');await page.unroute('**/models/weather-*.onnx');await run(page);
});

test('phone model workspace fits and never sends image pixels to a server',async({page})=>{
 await page.setViewportSize({width:390,height:844});const uploads=[];page.on('request',req=>{if(req.method()!=='GET')uploads.push(req.url());});await models(page);await run(page);
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);await page.getByLabel('Upload model image',{exact:true}).setInputFiles('web/assets/model-examples/weather-rain.png');await run(page);expect(uploads).toEqual([]);
});

test('switching to models releases an explicitly started camera',async({page})=>{
 await page.addInitScript(()=>{
  navigator.mediaDevices.getUserMedia=async()=>{
   const canvas=document.createElement('canvas');canvas.width=320;canvas.height=240;
   const ctx=canvas.getContext('2d');ctx.fillStyle='#26bc96';ctx.fillRect(0,0,320,240);
   const stream=canvas.captureStream(2);window.testVisionStream=stream;return stream;
  };
 });
 await page.goto('/');await page.getByRole('button',{name:'Use camera',exact:true}).click();await expect(page.getByRole('button',{name:'Stop camera',exact:true})).toBeVisible();await page.getByRole('button',{name:'Trained models',exact:true}).click();
 await expect.poll(()=>page.evaluate(()=>window.testVisionStream.getTracks().every(track=>track.readyState==='ended'))).toBe(true);
 await page.getByRole('button',{name:'Image tools & tracking',exact:true}).click();await expect(page.getByRole('button',{name:'Use camera',exact:true})).toBeVisible();
});

test('face sensitivity updates existing redaction without stale regions',async({page})=>{
 await page.goto('/');await page.getByRole('button',{name:'Region redaction',exact:true}).click();await page.getByRole('button',{name:'Detect faces',exact:true}).click();
 await page.getByLabel('Upload image',{exact:true}).setInputFiles('web/assets/model-examples/alpaca.png');
 await expect(page.locator('.vision-readout')).toContainText('1 region');
 await page.getByRole('group',{name:'Face sensitivity',exact:true}).getByRole('button',{name:'Strict',exact:true}).click();await expect(page.locator('.vision-readout')).toContainText('0 regions');
 await page.getByRole('button',{name:'Try face example',exact:true}).click();await expect(page.locator('.vision-readout')).toContainText('1 region');
});

test('tiger upload-only mode clears the previous source and exports pose results',async({page})=>{
 await models(page);await run(page);
 await page.getByRole('group',{name:'Trained model',exact:true}).getByRole('button',{name:'Tiger pose',exact:true}).click();
 await expect(page.getByRole('button',{name:'Run model',exact:true})).toBeDisabled();
 await expect(page.getByRole('button',{name:'Save results',exact:true})).toHaveCount(0);
 await expect(page.locator('.vision-model-image figcaption')).toHaveText('Upload a tiger photo');
 await expect(page.locator('.vision-evaluation')).toContainText('one video');
 // A repository-owned negative image exercises transport and inference without
 // redistributing frames from the external tiger video.
 await page.getByLabel('Upload model image',{exact:true}).setInputFiles('web/assets/model-examples/alpaca.png');await run(page);
 const actual=await result(page);expect(actual.kind).toBe('pose');expect(actual.model).toBe('tiger');expect(actual.source).toBe('alpaca.png');
 for(const box of actual.boxes){expect(box.keypoints).toHaveLength(12);expect(box.keypoints.every(point=>Number.isFinite(point.x)&&Number.isFinite(point.y))).toBe(true);}
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save annotated PNG',exact:true}).click();expect((await download).suggestedFilename()).toBe('tiger-keypoints.png');
 await page.getByRole('group',{name:'Trained model',exact:true}).getByRole('button',{name:'Weather',exact:true}).click();await expect(page.getByRole('button',{name:'Run model',exact:true})).toBeEnabled();
});
