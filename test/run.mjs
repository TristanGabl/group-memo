// Runs the engine on an audio file in Node: node test/run.mjs <file> [speakers] [fast|accurate]
import { execFileSync } from 'node:child_process';
import * as T from '@huggingface/transformers';
import { createEngine, SAMPLE_RATE } from '../engine.js';

const [file, speakers = '0', quality = 'fast'] = process.argv.slice(2);
const raw = execFileSync('ffmpeg', ['-loglevel', 'error', '-i', file, '-f', 'f32le', '-ac', '1', '-ar', String(SAMPLE_RATE), '-'], { maxBuffer: 1 << 30 });
const audio = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);

let lastStage = '';
const engine = createEngine(T, {
  asrDevice: 'cpu', smallDevice: 'cpu',
  onProgress: p => { if (p.stage !== lastStage) { lastStage = p.stage; console.error(`[${p.stage}]`); } },
});
const result = await engine.run(audio, { quality, language: 'en', speakers: Number(speakers) });
for (const u of result.utterances)
  console.log(`[${u.start.toFixed(1).padStart(5)}] Speaker ${u.speaker + 1}: ${u.text}`);
console.log(`\n${result.speakers} speakers, ${result.duration.toFixed(1)} s audio`, result.timings);
