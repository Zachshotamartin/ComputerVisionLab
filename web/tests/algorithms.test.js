import test from 'node:test';
import assert from 'node:assert/strict';
import { clipRegion, connectedRegions, hueDistance, hueToHex, processImage, rgbToHsv } from '../visionAlgorithms.js';

function image(width, height, fill = [80, 120, 160, 255]) {
  return { width, height, data: Uint8ClampedArray.from(Array.from({ length: width * height }, () => fill).flat()) };
}
test('hue distance wraps red across zero in either direction', () => {
  assert.equal(hueDistance(359, 1), 2);
  assert.equal(hueDistance(1, 359), 2);
  assert.deepEqual(rgbToHsv(255, 0, 0), [0, 1, 1]);
  assert.deepEqual(rgbToHsv(0, 0, 0), [0, 0, 0]);
  const input = image(2, 1); input.data.set([255, 0, 8, 255, 255, 8, 0, 255]);
  assert.equal(processImage(input, { mode: 'color', color: '#ff0000', tolerance: 3, minArea: 1 }).selectedPixels, 2);
});
test('regions remain separate across row boundaries and ignore isolated noise', () => {
  const mask = Uint8Array.from([0, 0, 1, 1, 0, 0, 1, 0, 0]);
  assert.deepEqual(connectedRegions(mask, 3, 3, 2), [{ x: 0, y: 1, width: 1, height: 2, area: 2 }]);
});
test('hue picker spans the spectrum and wraps without losing saturation', () => {
  assert.deepEqual([0, 60, 120, 180, 240, 300, 360].map(hueToHex), ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#ff0000']);
  assert.equal(hueToHex(-60), '#ff00ff');
  assert.throws(() => hueToHex(NaN), /finite/);
});
test('tracking includes muted green and never interprets gray targets as red', () => {
  const input = image(3, 1);
  input.data.set([217, 92, 67, 255, 100, 137, 119, 255, 128, 128, 128, 255]);
  for (const mode of ['color', 'overlay']) {
    const green = processImage(input, { mode, color: '#648977', minArea: 1 });
    assert.equal(green.selectedPixels, 1);
    assert.equal(green.boxes[0].x, 1);
    for (const color of ['#000000', '#ffffff', '#575757']) {
      const result = processImage(input, { mode, color, minArea: 1 });
      assert.equal(result.selectedPixels, 0);
      assert.deepEqual(result.boxes, []);
    }
  }
});
test('clipping handles backward drags and completely outside regions', () => {
  assert.deepEqual(clipRegion({ x: 4, y: 4, width: -6, height: -6 }, 5, 5), { x: 0, y: 0, width: 4, height: 4 });
  assert.equal(clipRegion({ x: -20, y: -20, width: 2, height: 2 }, 5, 5), null);
  assert.equal(clipRegion({ x: NaN, y: 0, width: 2, height: 2 }, 5, 5), null);
});
test('redaction affects only selected pixels and does not mutate source', () => {
  const input = image(8, 8);
  for (let i = 0; i < input.data.length; i += 4) input.data[i] = i % 255;
  const original = input.data.slice();
  const result = processImage(input, { mode: 'redact', amount: 100, regions: [{ x: -2, y: -2, width: 6, height: 6 }] });
  assert.notDeepEqual(result.data.slice(0, 16), original.slice(0, 16));
  assert.deepEqual(result.data.slice(4 * 8 * 4), original.slice(4 * 8 * 4));
  assert.deepEqual(input.data, original);
});
test('blur preserves constant images including corners and alpha', () => {
  const input = image(7, 5, [50, 100, 180, 90]);
  assert.deepEqual(processImage(input, { mode: 'blur', amount: 255 }).data, input.data);
});
test('flat image has no edges and threshold extremes change output', () => {
  const input = image(5, 5);
  assert.equal(processImage(input, { mode: 'edges' }).data[0], 18);
  assert.equal(processImage(input, { mode: 'threshold', amount: 1 }).data[0], 255);
  assert.equal(processImage(input, { mode: 'threshold', amount: 255 }).data[0], 0);
});
test('invalid dimensions and excessive image sizes fail before processing', () => {
  assert.throws(() => processImage({ width: 0, height: 2, data: [] }), /dimensions/);
  assert.throws(() => processImage({ width: 2048, height: 2048, data: [] }), /dimensions/);
  assert.throws(() => processImage(image(2, 2), { mode: 'color', color: 'oops' }), /hex color/);
});
