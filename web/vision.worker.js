import {processImage} from './visionAlgorithms.js';
import {inferModel} from './modelEngine.js';
self.onmessage=async({data:{id,image,options}})=>{
 try{
  const start=performance.now();
  if(options.mode==='redact'&&options.automatic){
   const faces=await inferModel(image,'face',{assetBase:options.assetBase,threshold:options.faceThreshold??.65,progress:message=>self.postMessage({id,progress:message})});
   options={...options,regions:[...options.regions,...faces.boxes.map(box=>({x:box.x-box.width*.12,y:box.y-box.height*.15,width:box.width*1.24,height:box.height*1.3}))]};
  }
  const result=processImage(image,options);
  self.postMessage({id,result,milliseconds:performance.now()-start},[result.data.buffer]);
 }catch(error){self.postMessage({id,error:error.message});}
};
