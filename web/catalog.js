export const visionModes = [
  { id: 'color', label: 'Color tracking', description: 'Find each region matching a color. Pick it from the source image, then widen or narrow the tolerance.' },
  { id: 'edges', label: 'Edge detection', description: 'Reveal outlines in the image. Raise the threshold to keep stronger edges.' },
  { id: 'threshold', label: 'Threshold', description: 'Split luminance into light and dark regions. Move the threshold to inspect which details remain.' },
  { id: 'blur', label: 'Blur', description: 'Soften the image. Increase the strength to blend a larger neighborhood of pixels.' },
  { id: 'redact', label: 'Region redaction', description: 'Select regions manually, or detect and pixelate faces automatically. Inspect the result before sharing.' },
  { id: 'rectangles', label: 'Rectangle tracking', description: 'Find high-contrast rectangular outlines, including tilted and perspective shapes. Use a photo or explicitly start the camera. This is 2D tracking.' },
  { id: 'overlay', label: 'Marker overlay', description: 'Attach a wireframe to the largest matching colored region. This follows a 2D marker; it does not estimate depth.' },
];
