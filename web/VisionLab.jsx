import { useCallback, useEffect, useRef, useState } from 'react';
import { visionModes } from './catalog.js';
import ColorPicker from './ColorPicker.jsx';
import './vision.css';

const samples = [
  { id: 'bird', title: 'Hummingbird', src: 'bird.webp', color: '#35b8a3' },
  { id: 'shapes', title: 'Color markers', src: 'markers.png', color: '#d95c43' },
];

function drawOverlay(context, boxes, mode) {
  context.lineWidth = 2;
  context.strokeStyle = '#c5e7a1';
  context.fillStyle = '#c5e7a1';
  for (const [index, box] of boxes.entries()) {
    if (mode === 'redact') continue;
    const { x, y, width, height } = box;
    if(box.corners){context.beginPath();box.corners.forEach(([cx,cy],i)=>i?context.lineTo(cx,cy):context.moveTo(cx,cy));context.closePath();context.stroke();continue;}
    context.strokeRect(x, y, width, height);
    context.font = '13px monospace';
    context.fillText(`${index + 1}`, x + 5, Math.max(15, y - 7));
    if (mode === 'overlay' && index === 0) {
      const dx = width * .25, dy = -height * .25;
      context.strokeRect(x + dx, y + dy, width, height);
      for (const [cx, cy] of [[x, y], [x + width, y], [x, y + height], [x + width, y + height]]) {
        context.beginPath(); context.moveTo(cx, cy); context.lineTo(cx + dx, cy + dy); context.stroke();
      }
    }
  }
}

export default function VisionLab({ assetBase = '/assets/computer-vision/' }) {
  const root=useRef(null);
  const [picking,setPicking]=useState(false),[preview,setPreview]=useState('compare');
  const [automatic,setAutomatic]=useState(false),[faceThreshold,setFaceThreshold]=useState(.65),[progress,setProgress]=useState('');
  const inputs = samples.map(sample => ({ ...sample, src: `${assetBase.replace(/\/?$/, '/')}${sample.src}` }));
  const [mode, setMode] = useState('color'), [amount, setAmount] = useState(100), [color, setColor] = useState(samples[0].color);
  const [tolerance, setTolerance] = useState(24), [regions, setRegions] = useState([]), [source, setSource] = useState(inputs[0]);
  const [pixels, setPixels] = useState(null), [metrics, setMetrics] = useState(null), [error, setError] = useState('');
  const [camera, setCamera] = useState(false), [requestingCamera, setRequestingCamera] = useState(false), [loading, setLoading] = useState(true);
  const before = useRef(null), after = useRef(null), video = useRef(null), worker = useRef(null), stream = useRef(null);
  const nextId = useRef(0), busy = useRef(false), pending = useRef(null), lastImage = useRef(null), currentOptions = useRef(null);
  const pointer = useRef(null), live = useRef(true), cameraGeneration = useRef({ value: 0 });

  const submit = useCallback((image, options) => {
    if (!worker.current || !image || !options) return;
    const job = { id: ++nextId.current, image: { data: image.data.slice(), width: image.width, height: image.height }, options };
    if (busy.current) { pending.current = job; return; }
    busy.current = true;
    worker.current.postMessage(job, [job.image.data.buffer]);
  }, []);

  const stopCamera = useCallback(() => {
    cameraGeneration.current.value++;
    stream.current?.getTracks().forEach(track => track.stop());
    stream.current = null;
    if (video.current) { video.current.pause(); video.current.srcObject = null; }
    setCamera(false); setRequestingCamera(false);
  }, []);

  useEffect(() => {
    live.current = true;
    const cameraToken = cameraGeneration.current;
    const engine = new Worker(new URL('./vision.worker.js', import.meta.url), { type: 'module' });
    worker.current = engine;
    engine.onmessage = ({ data }) => {
      if(data.progress){if(data.id===nextId.current)setProgress(data.progress);return;}
      busy.current = false;
      if (data.id === nextId.current) {
        if (data.error) { setError(data.error); setMetrics(null); setLoading(false); setProgress(''); }
        else {
          const canvas = after.current, result = data.result;
          canvas.width = result.width; canvas.height = result.height;
          const context = canvas.getContext('2d');
          context.putImageData(new ImageData(result.data, result.width, result.height), 0, 0);
          drawOverlay(context, result.boxes, currentOptions.current.mode);
          setMetrics({ regions: result.boxes.length, milliseconds: data.milliseconds, width: result.width, height: result.height });
          setLoading(false);setProgress('');setError('');
        }
      }
      if (pending.current) {
        const job = pending.current; pending.current = null; busy.current = true;
        engine.postMessage(job, [job.image.data.buffer]);
      }
    };
    engine.onerror = () => { busy.current = false; setError('The image processor could not start. Reload this page to try again.'); setLoading(false); };
    return () => {
      live.current = false; cameraToken.value++;
      stream.current?.getTracks().forEach(track => track.stop());
      engine.terminate(); worker.current = null; busy.current = false; pending.current = null;
    };
  }, []);

  useEffect(() => {
    const options = { mode, amount, color, tolerance, regions, automatic, faceThreshold, assetBase, minArea: 50 };
    setLoading(true);setProgress('');
    currentOptions.current = options;
    submit(lastImage.current || pixels, options);
  }, [mode, amount, color, tolerance, regions, automatic, faceThreshold, assetBase, pixels, submit]);

  useEffect(() => {
    let canceled = false;
    const image = new Image();
    image.onload = () => {
      if (canceled) return;
      const scale = Math.min(1, 768 / image.naturalWidth, 512 / image.naturalHeight), canvas = before.current;
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale)); canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const next = context.getImageData(0, 0, canvas.width, canvas.height);
      lastImage.current = next;
      setPixels(next); setError('');
    };
    image.onerror = () => { if (!canceled) { setError('This image could not be opened. Choose a PNG, JPEG, or WebP image.'); setLoading(false); } };
    image.src = source.src;
    return () => { canceled = true; if (source.src.startsWith('blob:')) URL.revokeObjectURL(source.src); };
  }, [source]);

  useEffect(() => {
    if (!camera) return;
    let frame, last = 0;
    function tick(now) {
      frame = requestAnimationFrame(tick);
      if (now - last < 100 || document.hidden || video.current.readyState < 2 || busy.current) return;
      last = now;
      const element = video.current, canvas = before.current;
      const scale = Math.min(1, 640 / element.videoWidth, 480 / element.videoHeight);
      canvas.width = Math.max(1, Math.round(element.videoWidth * scale)); canvas.height = Math.max(1, Math.round(element.videoHeight * scale));
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.drawImage(element, 0, 0, canvas.width, canvas.height);
      lastImage.current = context.getImageData(0, 0, canvas.width, canvas.height);
      submit(lastImage.current, currentOptions.current);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [camera, submit]);

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) { setError('Camera access needs a supported browser on HTTPS or localhost. You can still upload an image.'); return; }
    setError(''); setRequestingCamera(true);
    const generation = ++cameraGeneration.current.value;
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      if (!live.current || generation !== cameraGeneration.current.value) { media.getTracks().forEach(track => track.stop()); return; }
      stream.current = media; video.current.srcObject = media;
      await video.current.play();
      if (!live.current || generation !== cameraGeneration.current.value) { media.getTracks().forEach(track => track.stop()); return; }
      setRegions([]); setCamera(true);
    } catch (failure) {
      stream.current?.getTracks().forEach(track => track.stop()); stream.current = null;
      if (live.current && generation === cameraGeneration.current.value) setError(failure.name === 'NotAllowedError' ? 'Camera access was declined. Choose an image or allow the camera in your browser settings.' : 'The camera could not start. Check whether another application is using it, or upload an image.');
    } finally { if (live.current && generation === cameraGeneration.current.value) setRequestingCamera(false); }
  }

  function chooseSource(next) {
    stopCamera(); setRegions([]); setLoading(true); setMetrics(null); setError('');
    if (next.color) setColor(next.color);
    setSource({ ...next });
  }

  function upload(event) {
    const file = event.target.files?.[0]; event.target.value = '';
    if (!file) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { setError('Choose a JPEG, PNG, or WebP image.'); return; }
    if (file.size > 20 * 1024 * 1024) { setError('Choose an image smaller than 20 MB.'); return; }
    chooseSource({ id: 'upload', title: file.name, src: URL.createObjectURL(file) });
  }

  function point(event, canvas = after.current) {
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
    const left = rect.left + (rect.width - canvas.width * scale) / 2;
    const top = rect.top + (rect.height - canvas.height * scale) / 2;
    return {
      x: Math.max(0, Math.min(canvas.width - 1, (event.clientX - left) / scale)),
      y: Math.max(0, Math.min(canvas.height - 1, (event.clientY - top) / scale)),
    };
  }

  function finishRegion(event) {
    if (!pointer.current) return;
    const start = pointer.current, end = point(event); pointer.current = null;
    if (Math.abs(end.x - start.x) >= 3 && Math.abs(end.y - start.y) >= 3) setRegions(previous => [...previous, { x: start.x, y: start.y, width: end.x - start.x, height: end.y - start.y }]);
  }

  function exportImage() {
    after.current.toBlob(blob => {
      if (!blob) { setError('The image could not be exported. Try again.'); return; }
      const url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = `vision-lab-${mode}.png`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, 'image/png');
  }

  const selected = visionModes.find(item => item.id === mode), tracking = ['color', 'overlay'].includes(mode);
  return <section ref={root} className="vision-lab lab-editor" aria-label="Interactive computer vision lab" data-ready={Boolean(metrics)}>
    <div className="vision-toolbar">
      <div><span className="vision-kicker">Input</span><div className="vision-sources">{inputs.map(sample => <button type="button" key={sample.id} aria-pressed={!camera && source.id === sample.id} onClick={() => chooseSource(sample)}>{sample.title}</button>)}<label className="vision-upload">Upload image<input type="file" accept="image/jpeg,image/png,image/webp" onChange={upload} /></label></div></div>
      <button type="button" className="vision-camera" disabled={requestingCamera} onClick={camera ? stopCamera : startCamera}>{requestingCamera ? 'Opening camera…' : camera ? 'Stop camera' : 'Use camera'}</button>
    </div>
    <div className="vision-workspace">
      <aside className="vision-controls">
        <div className="lab-switcher vision-operation-switcher" role="group" aria-label="Vision operation">{visionModes.map(item=><button type="button" key={item.id} aria-pressed={mode===item.id} onClick={()=>{setMode(item.id);setPicking(false);setError('');}}>{item.label}</button>)}</div>
        <p>{selected.description}</p>{mode==="overlay"&&<button type="button" className="lab-text-button" onClick={()=>chooseSource(inputs[1])}>Try the marker example</button>}
        {tracking ? <><ColorPicker value={color} onChange={setColor} picking={picking} onPick={()=>{setPicking(!picking);setPreview("input");}} /><label>Color tolerance <output>{tolerance}°</output><input aria-label="Color tolerance" type="range" min="2" max="90" value={tolerance} onChange={event => setTolerance(Number(event.target.value))} /></label></> : <label>{mode === 'redact' ? 'Pixel size' : mode === 'blur' ? 'Blur strength' : mode === 'rectangles' ? 'Edge strength' : 'Threshold'} <output>{amount}</output><input aria-label={mode === 'redact' ? 'Pixel size' : mode === 'blur' ? 'Blur strength' : mode === 'rectangles' ? 'Edge strength' : 'Threshold'} type="range" min="1" max="255" value={amount} onChange={event => setAmount(Number(event.target.value))} /></label>}
        {mode==='redact'&&<><div className="lab-switcher" role="group" aria-label="Redaction selection"><button type="button" aria-pressed={!automatic} onClick={()=>setAutomatic(false)}>Manual regions</button><button type="button" aria-pressed={automatic} onClick={()=>setAutomatic(true)}>Detect faces</button></div>{automatic&&<><div className="lab-switcher" role="group" aria-label="Face sensitivity">{[[.85,'Strict'],[.65,'Balanced'],[.45,'Thorough']].map(([value,label])=><button type="button" key={value} aria-pressed={faceThreshold===value} onClick={()=>setFaceThreshold(value)}>{label}</button>)}</div><p>Strict reduces false detections; Thorough may find more faces but can also redact other objects.</p><p>Pretrained YuNet finds faces, then pixelates padded regions. Check the result; a face can be missed.</p><button type="button" onClick={()=>chooseSource({id:'face',title:'Face detection example',src:assetBase+'model-examples/face.png'})}>Try face example</button></>}</>}
        {mode==='rectangles'&&<button type="button" onClick={()=>chooseSource({id:'rectangles',title:'Perspective rectangles',src:assetBase+'rectangles.png'})}>Try rectangle example</button>}
        {mode === 'redact' && <div className="vision-region-actions"><button type="button" onClick={() => setRegions(previous => [...previous, { x: after.current.width * .3, y: after.current.height * .25, width: after.current.width * .4, height: after.current.height * .5 }])}>Add center region</button><button type="button" disabled={!regions.length} onClick={() => setRegions(previous => previous.slice(0, -1))}>Undo region</button><button type="button" disabled={!regions.length} onClick={() => setRegions([])}>Clear regions</button></div>}
        <div className="vision-readout"><span>{camera ? 'Live camera · up to 10 fps' : source.title}</span><strong>{metrics ? `${metrics.width} × ${metrics.height}` : 'Opening image…'}</strong>{metrics && <span>{tracking || ['redact','rectangles'].includes(mode) ? `${metrics.regions} region${metrics.regions === 1 ? '' : 's'} · ` : ''}{metrics.milliseconds.toFixed(1)} ms processing</span>}</div>
        <button className="vision-export" type="button" onClick={exportImage} disabled={!metrics || loading}>Export result</button>
      </aside>
      <div className="vision-preview"><div className="lab-switcher" role="group" aria-label="Vision preview">{[['input','Source'],['result','Result'],['compare','Compare']].map(([value,label])=><button type="button" key={value} aria-pressed={preview===value} onClick={()=>setPreview(value)}>{label}</button>)}</div><p className="lab-live-state" role="status">{loading?(progress||'Processing image…'):picking?'Click or tap the color you want in the source image.':'Live result · adjustments update automatically'}</p><div className="vision-images" data-preview={preview} aria-busy={loading}>
        <figure hidden={preview==='result'}><figcaption><span>01</span> Source</figcaption><canvas ref={before} aria-label={picking?"Source image: click to pick a target color":"Source image"} className={picking?'vision-pick-canvas':''} width="768" height="512" onPointerDown={event=>{if(!picking)return;const {x,y}=point(event,before.current);const rgb=before.current.getContext('2d').getImageData(x,y,1,1).data;setColor('#'+[...rgb].slice(0,3).map(c=>c.toString(16).padStart(2,'0')).join(''));setPicking(false);setPreview('compare');}} /></figure>
        <figure hidden={preview==="input"}><figcaption><span>02</span> {selected.label}{mode === 'redact' && <small>Drag to select</small>}</figcaption><canvas ref={after} aria-label="Processed image; in redaction mode drag to select a region" className={mode === 'redact' ? 'vision-redaction-canvas' : ''} width="768" height="512" onPointerDown={event => { if (mode === 'redact') { pointer.current = point(event); event.currentTarget.setPointerCapture(event.pointerId); } }} onPointerUp={finishRegion} onPointerCancel={() => { pointer.current = null; }} /></figure>
      </div></div>
    </div>
    {error && <p className="vision-error" role="alert">{error}</p>}
    <p className="vision-privacy">Images and camera frames stay in this browser. Processing is capped at 768 × 512 pixels; exports use that working resolution.</p>
    <video ref={video} muted playsInline className="vision-hidden-video" aria-hidden="true" />
  </section>;
}
