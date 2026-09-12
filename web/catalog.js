export const visionModes = [
  { id: 'color', label: 'Color tracking', description: 'Choose a target hue. Separate connected regions are detected independently; the hue range wraps correctly through red.' },
  { id: 'edges', label: 'Edge detection', description: 'A Sobel gradient reveals changes in brightness. Raise the threshold to keep only stronger edges.' },
  { id: 'threshold', label: 'Threshold', description: 'Split luminance into light and dark regions. Move the threshold to inspect which details remain.' },
  { id: 'blur', label: 'Blur', description: 'An integral-image box filter averages a neighborhood around each pixel, with correct normalization at the image border.' },
  { id: 'redact', label: 'Region redaction', description: 'Drag across the result to pixelate a region. This browser tool uses your selections; automatic frontal-face detection is available in the Python version.' },
  { id: 'overlay', label: 'Marker overlay', description: 'A wireframe follows the largest region matching your chosen color. This is a 2D color marker, without depth or camera-pose estimation.' },
];
