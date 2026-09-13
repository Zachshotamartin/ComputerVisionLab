import {useEffect,useRef,useState} from 'react';
import {modelChoices,modelExamples,classLabel} from './modelCatalog.js';
import './models.css';

export default function ModelLab({assetBase='/assets/computer-vision/'}) {
 const [modelId,setModelId]=useState('weather'),[source,setSource]=useState(null),[pixels,setPixels]=useState(null);
 const [result,setResult]=useState(null),[busy,setBusy]=useState(false),[status,setStatus]=useState('Choose an example or upload an image, then run the model.'),[error,setError]=useState('');
 const [cutoff,setCutoff]=useState(.35);
 const canvas=useRef(null),engine=useRef(null),request=useRef(0),sourceGeneration=useRef(0);
 const model=modelChoices.find(item=>item.id===modelId);
 const base=new URL(assetBase,typeof location==='undefined'?'https://zachsm.com':location.origin).href;
 const examples=modelExamples[modelId];
 const boxes=result?.boxes?.filter(box=>box.score>=cutoff)||[];
 function cancel(message='Stopped. You can run the model again.') {
  request.current++;engine.current?.terminate();engine.current=null;setBusy(false);setStatus(message);
 }
 useEffect(()=>()=>{request.current++;engine.current?.terminate();},[]);
 useEffect(()=>{
  const example=modelExamples[modelId][0];
  setSource({title:classLabel(example.label)+' example',src:base+'model-examples/'+example.file,id:example.file});
 },[modelId,base]);
 useEffect(()=>{
  if(!source)return;
  const generation=++sourceGeneration.current,image=new Image();
  setPixels(null);setResult(null);setError('');
  image.onload=()=>{
   if(generation!==sourceGeneration.current)return;
   const scale=Math.min(1,768/image.naturalWidth,512/image.naturalHeight),c=canvas.current;
   c.width=Math.max(1,Math.round(image.naturalWidth*scale));c.height=Math.max(1,Math.round(image.naturalHeight*scale));
   const ctx=c.getContext('2d',{willReadFrequently:true});ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);ctx.drawImage(image,0,0,c.width,c.height);
   setPixels(ctx.getImageData(0,0,c.width,c.height));
  };
  image.onerror=()=>{if(generation===sourceGeneration.current)setError('This image could not be opened. Choose a JPEG, PNG, or WebP.');};
  image.src=source.src;
  return()=>{sourceGeneration.current++;if(source.src.startsWith('blob:'))URL.revokeObjectURL(source.src);};
 },[source]);
 useEffect(()=>{
  if(!pixels)return;
  const c=canvas.current,ctx=c.getContext('2d');ctx.putImageData(pixels,0,0);
  for(const box of result?.boxes||[]){
   if(box.score<cutoff)continue;
   ctx.strokeStyle='#dbecac';ctx.lineWidth=Math.max(2,c.width/300);ctx.strokeRect(box.x,box.y,box.width,box.height);
   const text=`Alpaca · ${Math.round(box.score*100)}%`;ctx.font='bold 14px monospace';
   const y=Math.max(20,box.y),w=ctx.measureText(text).width+12,x=Math.min(box.x,Math.max(0,c.width-w));
   ctx.fillStyle='#172b25';ctx.fillRect(x,y-20,w,20);ctx.fillStyle='#e7eddf';ctx.fillText(text,x+6,y-5);
  }
 },[pixels,result,cutoff]);
 function chooseModel(id){if(id===modelId)return;cancel('Choose an example or upload an image, then run the model.');setPixels(null);setResult(null);setError('');setModelId(id);}
 function chooseSource(next){cancel('Ready for a new prediction.');setPixels(null);setResult(null);setSource(next);}
 function upload(event){
  const file=event.target.files?.[0];event.target.value='';if(!file)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)||file.size>20*1024*1024){setError('Choose a JPEG, PNG, or WebP smaller than 20 MB.');return;}
  chooseSource({id:'upload',title:file.name,src:URL.createObjectURL(file)});
 }
 function run(){
  if(!pixels||busy)return;
  setBusy(true);setError('');setResult(null);setStatus('Preparing this model…');
  const id=++request.current;
  try {
  if(!engine.current)engine.current=new Worker(new URL('./model.worker.js',import.meta.url),{type:'module'});
  engine.current.onmessage=({data})=>{
   if(data.id!==request.current)return;
   if(data.progress){setStatus(data.progress);return;}
   setBusy(false);
   if(data.error){setError(data.error);setStatus('Prediction failed. You can try again.');return;}
   setResult(data.result);setStatus(`Complete · ${data.result.milliseconds.toFixed(0)} ms inference`);
  };
  engine.current.onerror=()=>{if(id===request.current){cancel('Prediction stopped.');setError('The local model engine could not start. Try again or reload this page.');}};
  const data=pixels.data.slice();
  engine.current.postMessage({id,model:modelId,image:{data,width:pixels.width,height:pixels.height},options:{assetBase:base,threshold:.05}},[data.buffer]);
  } catch {cancel('Prediction stopped.');setError('The local model engine could not start. Try again or reload this page.');}
 }
 function save(){
  if(!result||busy)return;
  const output={model:modelId,checkpoint:model.sha256,source:source.title,...result,...(result.boxes?{boxes,cutoff}:{})};
  const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`vision-${modelId}-result.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }
 function saveImage(){canvas.current.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='alpaca-detections.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});}
 return <section className="vision-lab vision-model-lab" aria-label="Trained image models" data-model={modelId} data-state={busy?'running':result?'complete':'idle'}>
  <div className="vision-model-selector" role="group" aria-label="Trained model">{modelChoices.map(item=><button key={item.id} type="button" aria-pressed={modelId===item.id} onClick={()=>chooseModel(item.id)}>{item.title}</button>)}</div>
  <div className="vision-model-intro"><h3>{model.description}</h3><p>{model.hint}</p></div>
  <div className="vision-toolbar"><div className="vision-sources" role="group" aria-label="Model examples">{examples.map(example=><button type="button" key={example.file} aria-pressed={source?.id===example.file} onClick={()=>chooseSource({id:example.file,title:classLabel(example.label)+' example',src:base+'model-examples/'+example.file})}>{classLabel(example.label)}</button>)}<label className="vision-upload">Upload image<input aria-label="Upload model image" type="file" accept="image/jpeg,image/png,image/webp" onChange={upload}/></label></div></div>
  <div className="vision-model-workspace">
   <figure className="vision-model-image"><figcaption>{source?.title||'Opening example…'}</figcaption><canvas ref={canvas} width="768" height="512" aria-label={modelId==='alpaca'?'Image with alpaca detections':'Image to classify'}/></figure>
   <aside className="vision-model-results" aria-label="Prediction results">
    <div className="vision-model-run"><button className="vision-model-primary" type="button" disabled={!pixels||busy} onClick={run}>{busy?'Running…':'Run model'}</button>{busy&&<button type="button" onClick={()=>cancel()}>Cancel</button>}</div>
    <p className="vision-model-status" role="status">{status}</p>
    {error&&<p className="vision-error" role="alert">{error}</p>}
    {result?.kind==='classify'&&<><h4>{classLabel(result.label)}</h4><ol className="vision-scores">{result.scores.map(item=><li key={item.label}><div><span>{classLabel(item.label)}</span><strong>{item.score<.001?'<0.1':(item.score*100).toFixed(1)}%</strong></div><meter min="0" max="1" value={item.score} aria-label={`${classLabel(item.label)} model score`}/></li>)}</ol><p className="vision-model-note">Scores compare the model’s known classes; they are not a guarantee of correctness.</p></>}
    {result?.kind==='svm'&&<><h4>{classLabel(result.label)}</h4><p className="vision-margin">Decision margin <strong>{result.margin.toFixed(3)}</strong></p><p className="vision-model-note">Positive means occupied; negative means empty. This SVM does not output calibrated probabilities.</p></>}
    {modelId==='alpaca'&&<fieldset className="vision-cutoff"><legend>Detection cutoff</legend><div role="group" aria-label="Detection cutoff">{[[.6,'Strict'],[.35,'Balanced'],[.15,'Explore']].map(([value,label])=><button type="button" key={value} aria-pressed={cutoff===value} onClick={()=>setCutoff(value)}>{label}</button>)}</div>{result&&<p>{boxes.length} alpaca{boxes.length===1?'':'s'} above {Math.round(cutoff*100)}%{!boxes.length?' · Try a clearer image or a lower cutoff.':''}</p>}</fieldset>}
    {result&&<div className="vision-model-exports"><button type="button" onClick={save}>Save results</button>{modelId==='alpaca'&&<button type="button" onClick={saveImage}>Save annotated PNG</button>}</div>}
    <p className="vision-model-note">Your saved {model.kind==='svm'?'SVM':'YOLOv8'} model · {(model.bytes/1048576).toFixed(1)} MB{model.kind!=='svm'?' + shared inference runtime':''}. Downloads only when you run it.</p>
   </aside>
  </div>
  <p className="vision-privacy">Images stay on your device. Only the selected model downloads; inference runs in a browser worker. Archived checkpoints are demonstrations, not independently validated benchmarks.</p>
 </section>;
}
