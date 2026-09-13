import { useId, useState } from 'react';
import { hueToHex, rgbToHsv } from './visionAlgorithms.js';

const swatches = [
  ['Red', '#d95c43'], ['Yellow', '#d9b344'], ['Green', '#648977'],
  ['Turquoise', '#35b8a3'], ['Blue', '#435cda'], ['Purple', '#9e53bf'],
];

export default function ColorPicker({ value, onChange, onPick, picking }) {
  const id = useId(), [draft, setDraft] = useState(null);
  const [hue, saturation] = rgbToHsv(...[1, 3, 5].map(offset => parseInt(value.slice(offset, offset + 2), 16)));
  const invalid = draft !== null && !/^#[a-f\d]{6}$/i.test(draft);
  function choose(color) { setDraft(null); onChange(color); }
  function edit(event) {
    const text = event.currentTarget.value;
    const normalized = text.startsWith('#') ? text : `#${text}`;
    setDraft(text);
    if (/^#[a-f\d]{6}$/i.test(normalized)) { setDraft(null); onChange(normalized.toLowerCase()); }
  }
  return <fieldset className="vision-color-picker">
    <legend>Target color</legend>
    <button type="button" aria-pressed={picking} onClick={onPick}>{picking?"Cancel color pick":"Pick color from image"}</button>
    <details className="lab-disclosure"><summary>Precise color & presets <span className="vision-color-preview" style={{backgroundColor:value}} aria-hidden="true" /></summary><div className="lab-disclosure__body"><div className="vision-color-value">
      <span className="vision-color-preview" style={{ backgroundColor: value }} aria-hidden="true" />
      <input type="text" aria-label="Target color" aria-describedby={`${id}-help`} aria-invalid={invalid} value={draft ?? value} onInput={edit} onChange={edit} onBlur={() => setDraft(null)} onKeyDown={event => { if (event.key === 'Escape') setDraft(null); }} spellCheck={false} autoComplete="off" maxLength={7} />
    </div>
    <label>Target hue <output>{Math.round(hue)}°</output><input className="vision-hue-slider" type="range" aria-label="Target hue" min="0" max="359" value={Math.round(hue)} onInput={event => choose(hueToHex(Number(event.currentTarget.value)))} onChange={event => choose(hueToHex(Number(event.currentTarget.value)))} /></label>
    <div className="vision-color-swatches" aria-label="Preset target colors">{swatches.map(([name, color]) => <button type="button" key={name} aria-label={`Track ${name.toLowerCase()}`} title={name} aria-pressed={value === color} onClick={() => choose(color)}><span style={{ backgroundColor: color }} /></button>)}</div>
    </div></details><p id={`${id}-help`} className="vision-color-help">{invalid ? 'Enter a six-digit hex color, such as #35b8a3.' : saturation < .15 ? 'Black, white, and gray have no hue. Choose a colored target.' : 'Pick a color in the source image, or open precise settings.'}</p>
  </fieldset>;
}
