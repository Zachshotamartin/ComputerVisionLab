/** Pure, bounded image operations shared by the browser worker and tests. */
export function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b), delta = max - min;
  let hue = 0;
  if (delta) hue = max === r ? ((g - b) / delta) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
  return [((hue * 60) + 360) % 360, max ? delta / max : 0, max];
}

export function hueDistance(a, b) {
  const difference = Math.abs(((a - b) % 360 + 360) % 360);
  return Math.min(difference, 360 - difference);
}

export function hueToHex(hue) {
  if (!Number.isFinite(hue)) throw new Error('Hue must be a finite number.');
  const segment = ((hue % 360 + 360) % 360) / 60;
  const x = Math.round(255 * (1 - Math.abs(segment % 2 - 1)));
  const rgb = [[255, x, 0], [x, 255, 0], [0, 255, x], [0, x, 255], [x, 0, 255], [255, 0, x]][Math.floor(segment)];
  return '#' + rgb.map(channel => channel.toString(16).padStart(2, '0')).join('');
}

export function clipRegion(region, width, height) {
  if (![region.x, region.y, region.width, region.height].every(Number.isFinite)) return null;
  const left = Math.max(0, Math.floor(Math.min(region.x, region.x + region.width)));
  const top = Math.max(0, Math.floor(Math.min(region.y, region.y + region.height)));
  const right = Math.min(width, Math.ceil(Math.max(region.x, region.x + region.width)));
  const bottom = Math.min(height, Math.ceil(Math.max(region.y, region.y + region.height)));
  return right > left && bottom > top ? { x: left, y: top, width: right - left, height: bottom - top } : null;
}

export function connectedRegions(mask, width, height, minArea = 30) {
  const visited = new Uint8Array(mask.length), queue = new Int32Array(mask.length), regions = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || visited[start]) continue;
    let head = 0, tail = 1, left = width, top = height, right = 0, bottom = 0;
    queue[0] = start; visited[start] = 1;
    while (head < tail) {
      const index = queue[head++], x = index % width, y = Math.floor(index / width);
      left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y);
      for (const next of [x > 0 ? index - 1 : -1, x + 1 < width ? index + 1 : -1, y > 0 ? index - width : -1, y + 1 < height ? index + width : -1]) {
        if (next < 0 || !mask[next] || visited[next]) continue;
        visited[next] = 1; queue[tail++] = next;
      }
    }
    if (tail >= minArea) regions.push({ x: left, y: top, width: right - left + 1, height: bottom - top + 1, area: tail });
  }
  return regions.sort((a, b) => b.area - a.area);
}

function boxBlur(data, width, height, radius) {
  const result = new Uint8ClampedArray(data), stride = width + 1;
  for (let channel = 0; channel < 3; channel++) {
    const integral = new Float64Array((width + 1) * (height + 1));
    for (let y = 0; y < height; y++) {
      let row = 0;
      for (let x = 0; x < width; x++) {
        row += data[(y * width + x) * 4 + channel];
        integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row;
      }
    }
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const left = Math.max(0, x - radius), top = Math.max(0, y - radius);
      const right = Math.min(width, x + radius + 1), bottom = Math.min(height, y + radius + 1);
      result[(y * width + x) * 4 + channel] = (integral[bottom * stride + right] - integral[top * stride + right] - integral[bottom * stride + left] + integral[top * stride + left]) / ((right - left) * (bottom - top));
    }
  }
  return result;
}

export function processImage({ data, width, height }, options = {}) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || width * height > 1024 * 1024 || data.length !== width * height * 4) throw new Error('Invalid image dimensions; use at most one megapixel.');
  const { mode = 'edges', amount = 100, color = '#e5bc36', tolerance = 24, minArea = 50, regions = [] } = options;
  const output = new Uint8ClampedArray(data), gray = new Uint8ClampedArray(width * height);
  for (let i = 0; i < gray.length; i++) gray[i] = .2126 * data[i * 4] + .7152 * data[i * 4 + 1] + .0722 * data[i * 4 + 2];
  let boxes = [], selectedPixels = 0;
  if (mode === 'blur') return { data: boxBlur(data, width, height, Math.max(1, Math.min(20, Math.round(amount / 12)))), width, height, boxes, selectedPixels };
  if (mode === 'color' || mode === 'overlay') {
    if (!/^#[a-f\d]{6}$/i.test(color)) throw new Error('Choose a six-digit hex color.');
    const [target, targetSaturation] = rgbToHsv(...[1, 3, 5].map(offset => parseInt(color.slice(offset, offset + 2), 16)));
    const mask = new Uint8Array(width * height);
    for (let i = 0; i < mask.length; i++) {
      const [h, s, v] = rgbToHsv(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
      // Achromatic targets have no hue; muted colors still need to be trackable.
      mask[i] = targetSaturation >= .15 && hueDistance(h, target) <= tolerance && s >= .15 && v >= .18 ? 1 : 0;
      selectedPixels += mask[i];
      if (!mask[i] && mode === 'color') for (let channel = 0; channel < 3; channel++) output[i * 4 + channel] = gray[i] * .42;
    }
    boxes = connectedRegions(mask, width, height, minArea);
  } else if (mode === 'redact') {
    const block = Math.max(4, Math.round(amount / 5));
    for (const region of regions) {
      const box = clipRegion(region, width, height);
      if (!box) continue;
      boxes.push(box);
      for (let y = box.y; y < box.y + box.height; y += block) for (let x = box.x; x < box.x + box.width; x += block) {
        const right = Math.min(x + block, box.x + box.width), bottom = Math.min(y + block, box.y + box.height), sums = [0, 0, 0];
        for (let j = y; j < bottom; j++) for (let i = x; i < right; i++) for (let c = 0; c < 3; c++) sums[c] += data[(j * width + i) * 4 + c];
        const count = (right - x) * (bottom - y);
        for (let j = y; j < bottom; j++) for (let i = x; i < right; i++) for (let c = 0; c < 3; c++) output[(j * width + i) * 4 + c] = sums[c] / count;
      }
    }
  } else if (mode !== 'original') {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const index = y * width + x;
      let value = gray[index];
      if (mode === 'threshold') value = value >= amount ? 255 : 0;
      if (mode === 'edges') {
        const at = (dx, dy) => gray[Math.max(0, Math.min(height - 1, y + dy)) * width + Math.max(0, Math.min(width - 1, x + dx))];
        const gx = -at(-1, -1) + at(1, -1) - 2 * at(-1, 0) + 2 * at(1, 0) - at(-1, 1) + at(1, 1);
        const gy = -at(-1, -1) - 2 * at(0, -1) - at(1, -1) + at(-1, 1) + 2 * at(0, 1) + at(1, 1);
        value = Math.hypot(gx, gy) >= amount ? 240 : 18;
      }
      output[index * 4] = output[index * 4 + 1] = output[index * 4 + 2] = value;
    }
  }
  return { data: output, width, height, boxes, selectedPixels };
}
