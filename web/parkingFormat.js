/** Compact RBF SVM transport: float64 scaling/coefficients, float32 support vectors. */
export function decodeParkingModel(buffer) {
 const view=new DataView(buffer);
 if(buffer.byteLength<8||view.getUint32(0,false)!==0x564c5331)throw new Error('Invalid parking model format');
 const length=view.getUint32(4,true);
 if(length>16384||8+length>buffer.byteLength)throw new Error('Invalid parking model header');
 const model=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,8,length)));
 const {features,supportCount,size,classes}=model;
 if(size!==15||features!==675||!Number.isInteger(supportCount)||supportCount<1||supportCount>20000||classes?.length!==2)throw new Error('Unsupported parking model dimensions');
 if(buffer.byteLength!==8+length+features*16+features*supportCount*4+supportCount*8)throw new Error('Incomplete parking model');
 let offset=8+length;
 function values(count,bytes){const result=bytes===4?new Float32Array(count):new Float64Array(count);for(let i=0;i<count;i++){result[i]=bytes===4?view.getFloat32(offset,true):view.getFloat64(offset,true);offset+=bytes;if(!Number.isFinite(result[i]))throw new Error('Non-finite parking parameters');}return result;}
 model.mean=values(features,8);model.scale=values(features,8);model.support=Array.from({length:supportCount},()=>values(features,4));model.coefficients=values(supportCount,8);
 if(model.scale.some(value=>value<=0)||!Number.isFinite(model.intercept)||!Number.isFinite(model.gamma)||model.gamma<=0)throw new Error('Invalid parking scaling');
 return model;
}
