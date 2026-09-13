/** Bounded, deterministic preprocessing and decoding. No browser APIs required. */
export function validateImage({data,width,height}) {
 if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width*height>1024*1024||data?.length!==width*height*4)throw new Error('Use an image of at most one megapixel.');
}
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const evenRound=x=>Math.abs(x%1-.5)<1e-10 ? 2*Math.round(x/2) : Math.round(x);

// PIL-compatible separable bilinear resize with antialiasing on downsampling.
function weights(from,to) {
 const scale=from/to,filter=Math.max(1,scale);
 return Array.from({length:to},(_,i)=>{
  const center=(i+.5)*scale;
  const start=Math.max(0,Math.floor(center-filter+.5)),end=Math.min(from,Math.floor(center+filter+.5));
  const pairs=[];let total=0;
  for(let p=start;p<end;p++){const w=Math.max(0,1-Math.abs((p+.5-center)/filter));pairs.push([p,w]);total+=w;}
  return pairs.map(([p,w])=>[p,w/total]);
 });
}
export function resizeRGB(image,width,height,{area=false}={}) {
 validateImage(image);
 const {data,width:sw,height:sh}=image,out=new Uint8Array(width*height*3);
 if(area){
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
   const x0=x*sw/width,x1=(x+1)*sw/width,y0=y*sh/height,y1=(y+1)*sh/height,sums=[0,0,0];
   for(let sy=Math.floor(y0);sy<Math.ceil(y1);sy++)for(let sx=Math.floor(x0);sx<Math.ceil(x1);sx++){
    const w=(Math.min(x1,sx+1)-Math.max(x0,sx))*(Math.min(y1,sy+1)-Math.max(y0,sy));
    for(let c=0;c<3;c++)sums[c]+=data[(Math.min(sh-1,sy)*sw+Math.min(sw-1,sx))*4+c]*w;
   }
   for(let c=0;c<3;c++)out[(y*width+x)*3+c]=evenRound(sums[c]/((x1-x0)*(y1-y0)));
  }
  return out;
 }
 const wx=weights(sw,width),wy=weights(sh,height),horizontal=new Uint8Array(width*sh*3);
 for(let y=0;y<sh;y++)for(let x=0;x<width;x++)for(let c=0;c<3;c++){
  let value=0;for(const [sx,w] of wx[x])value+=data[(y*sw+sx)*4+c]*w;
  horizontal[(y*width+x)*3+c]=Math.round(value);
 }
 for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let c=0;c<3;c++){
  let value=0;for(const [sy,w] of wy[y])value+=horizontal[(sy*width+x)*3+c]*w;
  out[(y*width+x)*3+c]=Math.round(value);
 }
 return out;
}
export function classificationTensor(image,size=64) {
 const short=Math.min(image.width,image.height),width=Math.floor(image.width*size/short),height=Math.floor(image.height*size/short);
 const pixels=resizeRGB(image,width,height),left=evenRound((width-size)/2),top=evenRound((height-size)/2),tensor=new Float32Array(3*size*size);
 for(let y=0;y<size;y++)for(let x=0;x<size;x++)for(let c=0;c<3;c++)tensor[c*size*size+y*size+x]=pixels[((top+y)*width+left+x)*3+c]/255;
 return tensor;
}
export function letterboxTensor(image,size,{bgr=false}={}) {
 validateImage(image);
 const scale=Math.min(size/image.width,size/image.height),width=Math.round(image.width*scale),height=Math.round(image.height*scale);
 const left=Math.round((size-width)/2-.1),top=Math.round((size-height)/2-.1),pixels=resizeRGB(image,width,height);
 const tensor=new Float32Array(3*size*size);tensor.fill(bgr?0:114/255);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let c=0;c<3;c++)tensor[c*size*size+(y+top)*size+x+left]=pixels[(y*width+x)*3+(bgr?2-c:c)]/(bgr?1:255);
 return {tensor,scale,left,top,width:image.width,height:image.height};
}
export function parkingPrediction(image,model) {
 const rgb=resizeRGB(image,15,15,{area:true}),vector=Float32Array.from(rgb,v=>v/255);
 const standardized=Array.from(vector,(v,i)=>(v-model.mean[i])/model.scale[i]);
 let margin=model.intercept;
 for(let j=0;j<model.support.length;j++){
  let squared=0;for(let i=0;i<standardized.length;i++)squared+=(standardized[i]-model.support[j][i])**2;
  margin+=model.coefficients[j]*Math.exp(-model.gamma*squared);
 }
 return {label:model.classes[margin>0?1:0],margin};
}
export function intersectionOverUnion(a,b) {
 const intersection=Math.max(0,Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y));
 return intersection/(a.width*a.height+b.width*b.height-intersection||1);
}
export function suppress(boxes,threshold=.45,limit=100) {
 const kept=[];
 for(const box of boxes.sort((a,b)=>b.score-a.score).slice(0,2000)){
  if(kept.every(other=>intersectionOverUnion(box,other)<threshold))kept.push(box);
  if(kept.length>=limit)break;
 }
 return kept;
}
function sourceBox(cx,cy,w,h,score,geometry) {
 const {scale,left,top,width,height}=geometry;
 const x=clamp((cx-w/2-left)/scale,0,width),y=clamp((cy-h/2-top)/scale,0,height);
 const right=clamp((cx+w/2-left)/scale,0,width),bottom=clamp((cy+h/2-top)/scale,0,height);
 return {x,y,width:right-x,height:bottom-y,score};
}
export function decodeAlpacas(values,geometry,threshold=.25) {
 const count=values.length/5,boxes=[];
 for(let i=0;i<count;i++)if(values[count*4+i]>=threshold){
  const box=sourceBox(values[i],values[count+i],values[count*2+i],values[count*3+i],values[count*4+i],geometry);
  if(box.width>1&&box.height>1)boxes.push({...box,label:'alpaca'});
 }
 return suppress(boxes);
}
export function decodeFaces(outputs,geometry,size=640,threshold=.7) {
 const boxes=[];
 for(const stride of [8,16,32]){
  const columns=size/stride,cls=outputs['cls_'+stride].data,obj=outputs['obj_'+stride].data,bbox=outputs['bbox_'+stride].data;
  for(let i=0;i<cls.length;i++){
   const score=Math.sqrt(clamp(cls[i],0,1)*clamp(obj[i],0,1));if(score<threshold)continue;
   const cx=(i%columns+bbox[i*4])*stride,cy=(Math.floor(i/columns)+bbox[i*4+1])*stride;
   const box=sourceBox(cx,cy,Math.exp(bbox[i*4+2])*stride,Math.exp(bbox[i*4+3])*stride,score,geometry);
   if(box.width>2&&box.height>2)boxes.push({...box,label:'face'});
  }
 }
 return suppress(boxes,.3,30);
}

/** Decode the explicit 12×2 landmark layout of the separately trained tiger prototype. */
export function decodePose(values,geometry,keypoints,threshold=.25) {
 const channels=5+keypoints.length*2,count=values.length/channels,boxes=[];
 if(!Number.isInteger(count))throw new Error('Unexpected pose output dimensions');
 for(let i=0;i<count;i++)if(values[count*4+i]>=threshold){
  const box=sourceBox(values[i],values[count+i],values[count*2+i],values[count*3+i],values[count*4+i],geometry);
  if(box.width<=1||box.height<=1)continue;
  const points=keypoints.map((name,j)=>{
   const x=(values[count*(5+j*2)+i]-geometry.left)/geometry.scale,y=(values[count*(6+j*2)+i]-geometry.top)/geometry.scale;
   return {name,x,y,inFrame:Number.isFinite(x)&&Number.isFinite(y)&&x>=0&&y>=0&&x<=geometry.width&&y<=geometry.height};
  });
  boxes.push({...box,label:'tiger',keypoints:points});
 }
 return suppress(boxes);
}
