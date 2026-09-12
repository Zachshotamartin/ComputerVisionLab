import { processImage } from './visionAlgorithms.js';
self.onmessage = ({ data: { id, image, options } }) => {
  try {
    const start = performance.now(), result = processImage(image, options);
    self.postMessage({ id, result, milliseconds: performance.now() - start }, [result.data.buffer]);
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
