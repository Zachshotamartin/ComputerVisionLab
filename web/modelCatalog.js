import manifest from './assets/models/manifest.json' with { type: 'json' };
export const modelChoices=[
 {id:'weather',title:'Weather',description:'Classify a scene as cloudy, rain, shine, or sunrise.',hint:'The model uses a center crop. It only knows these four weather labels.',...manifest.models.weather},
 {id:'parking',title:'Parking',description:'Check whether a cropped parking space is empty or occupied.',hint:'Use a crop of one parking space. A full parking lot is outside this model’s training.',...manifest.models.parking},
 {id:'oct',title:'Retinal OCT',description:'Explore a four-class retinal-scan research model.',hint:'Research demo, not a diagnostic tool. Use retinal OCT scans; ordinary photos are outside its training.',...manifest.models.oct},
 {id:'alpaca',title:'Alpacas',description:'Find alpacas and draw a box around each detection.',hint:'This detector was trained for alpacas only. Lower the score cutoff to inspect weaker detections.',...manifest.models.alpaca},
 ...(manifest.models.tiger?[{id:'tiger',title:'Tiger pose',description:'Explore a 12-point tiger pose prototype.',hint:'New training on frames from one tiger video. Landmarks are approximate; generalization to other animals or scenes is unproven. Upload a tiger photo to try it.',...manifest.models.tiger}]:[]),
];
export const modelExamples=manifest.examples;
export const classLabel=value=>({shine:'Shine',cloudy:'Cloudy',rain:'Rain',sunrise:'Sunrise',empty:'Empty',not_empty:'Occupied',NORMAL:'Normal',alpaca:'Alpaca',tiger:'Tiger'}[value]||value);
