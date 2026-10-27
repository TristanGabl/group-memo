// Runs the speech engine off the main thread. Models download once from
// Hugging Face and are cached by the browser; audio never leaves this worker.
import * as T from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import { createEngine } from './engine.js';

T.env.allowLocalModels = false;

const report = p => postMessage({ type: 'progress', ...p });
let engine = null;
let device = null;

async function hasWebGPU() {
  try { return !!(await navigator.gpu?.requestAdapter()); } catch { return false; }
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'run') return;
  try {
    if (!engine) {
      // Phones get the compact CPU models: WebGPU's larger model files and GPU buffers
      // can push a mobile tab past its memory limit.
      device = !data.lowMemory && (await hasWebGPU()) ? 'webgpu' : 'wasm';
      engine = createEngine(T, { asrDevice: device, smallDevice: 'wasm', lowMemory: data.lowMemory, onProgress: report });
    }
    postMessage({ type: 'device', device });
    let result;
    try {
      result = await engine.run(data.audio, data.options);
    } catch (err) {
      if (device !== 'webgpu') throw err;
      // Some GPUs advertise WebGPU but fail on these models; fall back to the CPU path once.
      device = 'wasm';
      engine = createEngine(T, { asrDevice: 'wasm', smallDevice: 'wasm', lowMemory: data.lowMemory, onProgress: report });
      postMessage({ type: 'device', device, fallback: String(err?.message ?? err) });
      result = await engine.run(data.audio, data.options);
    }
    postMessage({ type: 'result', result });
  } catch (err) {
    postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
};
