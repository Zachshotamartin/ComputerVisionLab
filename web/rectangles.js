/** Detect high-contrast convex quadrilateral outlines; this is 2D geometry, not AR pose. */
export function findRectangles(data,width,height,threshold=100) {
 const gray=new Float32Array(width*height),edge=new Uint8Array(width*height);
 for(let i=0;i<gray.length;i++)gray[i]=.2126*data[i*4]+.7152*data[i*4+1]+.0722*data[i*4+2];
 for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
  const i=y*width+x,gx=-gray[i-width-1]+gray[i-width+1]-2*gray[i-1]+2*gray[i+1]-gray[i+width-1]+gray[i+width+1];
  const gy=-gray[i-width-1]-2*gray[i-width]-gray[i-width+1]+gray[i+width-1]+2*gray[i+width]+gray[i+width+1];
  if(Math.hypot(gx,gy)>threshold)edge[i]=1;
 }
 const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
 const area=points=>Math.abs(points.reduce((sum,p,i)=>{const q=points[(i+1)%points.length];return sum+p[0]*q[1]-p[1]*q[0];},0))/2;
 function hull(points){
  points.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);const lower=[],upper=[];
  for(const p of points){while(lower.length>1&&cross(lower.at(-2),lower.at(-1),p)<=0)lower.pop();lower.push(p);}
  for(let i=points.length-1;i>=0;i--){const p=points[i];while(upper.length>1&&cross(upper.at(-2),upper.at(-1),p)<=0)upper.pop();upper.push(p);}
  return lower.slice(0,-1).concat(upper.slice(0,-1));
 }
 const queue=new Int32Array(width*height),boxes=[];
 for(let origin=0;origin<edge.length;origin++){
  if(!edge[origin])continue;
  let first=0,last=1;queue[0]=origin;edge[origin]=0;
  while(first<last){
   const index=queue[first++],x=index%width,y=Math.floor(index/width);
   for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    const nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=width||ny>=height)continue;
    const n=ny*width+nx;if(edge[n]){edge[n]=0;queue[last++]=n;}
   }
  }
  if(last<80)continue;
  const points=[];for(let i=0;i<last;i+=Math.max(1,Math.floor(last/4000)))points.push([queue[i]%width,Math.floor(queue[i]/width)]);
  const polygon=hull(points),originalArea=area(polygon);if(originalArea<width*height*.008)continue;
  while(polygon.length>4){let index=0,smallest=Infinity;for(let i=0;i<polygon.length;i++){const value=Math.abs(cross(polygon[(i+polygon.length-1)%polygon.length],polygon[i],polygon[(i+1)%polygon.length]));if(value<smallest){smallest=value;index=i;}}polygon.splice(index,1);}
  if(polygon.length!==4||area(polygon)/originalArea<.9)continue;
  const sides=polygon.map((p,i)=>Math.hypot(p[0]-polygon[(i+1)%4][0],p[1]-polygon[(i+1)%4][1]));
  if(Math.min(...sides)<20||Math.max(...sides)/Math.min(...sides)>12)continue;
  // A curved blob's interior edge points should not count as straight sides.
  let close=0;const tolerance=Math.max(3,Math.sqrt(originalArea)*.025);
  for(const p of points){let distance=Infinity;for(let i=0;i<4;i++){const a=polygon[i],b=polygon[(i+1)%4];distance=Math.min(distance,Math.abs(cross(a,b,p))/sides[i]);}if(distance<=tolerance)close++;}
  if(close/points.length<.75)continue;
  const xs=polygon.map(p=>p[0]),ys=polygon.map(p=>p[1]);
  boxes.push({x:Math.min(...xs),y:Math.min(...ys),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys),corners:polygon,area:originalArea});
 }
 return boxes.sort((a,b)=>b.area-a.area).slice(0,12);
}
