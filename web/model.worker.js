import {inferModel} from './modelEngine.js';
self.onmessage=async({data:{id,image,model,options}})=>{
 try{
  const result=await inferModel(image,model,{...options,progress:message=>self.postMessage({id,progress:message})});
  self.postMessage({id,result});
 }catch(error){self.postMessage({id,error:error.message||'The model could not run on this image.'});}
};
