import {lazy,Suspense,useState} from 'react';
import VisionLab from './VisionLab.jsx';
import './models.css';
const ModelLab=lazy(()=>import('./ModelLab.jsx'));
export default function VisionWorkspace({assetBase='/assets/computer-vision/'}) {
 const [view,setView]=useState('tools');
 return <div className="vision-complete-workspace">
  <div className="vision-workspace-switcher" role="group" aria-label="Computer Vision workspace">
   <button type="button" aria-pressed={view==='tools'} onClick={()=>setView('tools')}>Image tools & tracking</button>
   <button type="button" aria-pressed={view==='models'} onClick={()=>setView('models')}>Trained models</button>
  </div>
  {view==='tools'?<VisionLab assetBase={assetBase}/>:<Suspense fallback={<p role="status">Opening the model workspace…</p>}><ModelLab assetBase={assetBase}/></Suspense>}
 </div>;
}
