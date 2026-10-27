// Local speech engine: Whisper for words, pyannote for "who spoke when" inside
// short windows, WeSpeaker voiceprints to link those windows into people.
// Takes the transformers.js module as an argument so it runs in a browser
// worker and in Node alike.

export const SAMPLE_RATE = 16000;

export const ASR_MODELS = {
  fast: 'onnx-community/whisper-base_timestamped',
  accurate: 'onnx-community/whisper-small_timestamped',
};
const SEGMENTATION_MODEL = 'onnx-community/pyannote-segmentation-3.0';
const VOICEPRINT_MODEL = 'onnx-community/wespeaker-voxceleb-resnet34-LM';

const WINDOW_S = 10;           // pyannote segmentation was trained on 10 s chunks
const PIECE_S = 2;             // speech is cut into pieces of about this length for voiceprints
const MIN_VOICEPRINT_S = 0.8;  // shorter speech gives unreliable voiceprints
const SAME_CHAIN = 0.65;       // cosine similarity for joining neighbouring pieces
const SAME_PERSON = 0.5;       // average similarity above which two groups are one person

// lowMemory (phones): load one model group at a time and free the speaker models before Whisper loads.
export function createEngine(T, { asrDevice = 'wasm', smallDevice = 'wasm', lowMemory = false, onProgress = () => {} } = {}) {
  const cache = {};
  const files = new Map();

  function progress_callback(e) {
    if (e.status === 'progress' && e.total) {
      files.set(e.file, { loaded: e.loaded, total: e.total });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      onProgress({ stage: 'download', loaded, total });
    }
  }

  async function loadAsr(quality) {
    const id = ASR_MODELS[quality] ?? ASR_MODELS.fast;
    if (!cache[id]) {
      const dtype = asrDevice === 'webgpu'
        ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
        : 'q8';
      cache[id] = T.pipeline('automatic-speech-recognition', id, { device: asrDevice, dtype, progress_callback });
    }
    return cache[id];
  }

  async function loadDiarization() {
    cache.diarization ??= Promise.all([
      T.AutoProcessor.from_pretrained(SEGMENTATION_MODEL, { progress_callback }),
      T.AutoModelForAudioFrameClassification.from_pretrained(SEGMENTATION_MODEL, { device: smallDevice, dtype: 'fp32', progress_callback }),
      T.AutoProcessor.from_pretrained(VOICEPRINT_MODEL, { progress_callback }),
      T.AutoModel.from_pretrained(VOICEPRINT_MODEL, { device: smallDevice, dtype: 'fp32', progress_callback }),
    ]);
    return cache.diarization;
  }

  async function unloadDiarization() {
    if (!cache.diarization) return;
    const [, segModel, , vpModel] = await cache.diarization;
    delete cache.diarization;
    await Promise.all([segModel.dispose(), vpModel.dispose()]);
  }

  // Transcribes only the given speech blocks, each shorter than Whisper's 30 s window,
  // so no words get lost at chunk boundaries and silence is skipped.
  async function transcribe(audio, blocks, { quality, language }) {
    const asr = await loadAsr(quality);
    const words = [];
    for (const [i, [start, end]] of blocks.entries()) {
      onProgress({ stage: 'transcribe', done: i, total: blocks.length });
      const clip = audio.subarray(Math.floor(start * SAMPLE_RATE), Math.ceil(end * SAMPLE_RATE));
      const out = await asr(clip, { return_timestamps: 'word', language: language || null, task: 'transcribe' });
      for (const c of out.chunks ?? []) {
        // Skip empty words and Whisper's non-speech markers such as [BLANK_AUDIO] or (music).
        if (!c.text.trim() || /^\s*[[(].*[\])]\s*$/.test(c.text)) continue;
        words.push({ text: c.text, start: start + c.timestamp[0], end: start + (c.timestamp[1] ?? c.timestamp[0]) });
      }
    }
    return words;
  }

  // Returns { turns: [{ speaker, start, end }], speakers } with speakers numbered by first appearance.
  async function diarize(audio, { speakers: fixedCount } = {}) {
    const [segProc, segModel, vpProc, vpModel] = await loadDiarization();
    const win = WINDOW_S * SAMPLE_RATE;
    const windows = Math.ceil(audio.length / win);

    // 1. Speech regions and change points. pyannote's own speaker labels are ignored:
    //    they are only consistent inside one window and sometimes merge two voices.
    const segments = [];
    for (let w = 0; w < windows; w++) {
      onProgress({ stage: 'diarize', done: w, total: windows });
      const offset = w * win;
      const chunk = audio.subarray(offset, Math.min(offset + win, audio.length));
      if (chunk.length < 0.25 * SAMPLE_RATE) break;
      const { logits } = await segModel(await segProc(chunk));
      const [found] = segProc.post_process_speaker_diarization(logits, chunk.length);
      for (const seg of found) {
        const start = seg.start + offset / SAMPLE_RATE;
        const end = seg.end + offset / SAMPLE_RATE;
        if (segModel.config.id2label[seg.id] !== 'NO_SPEAKER' && end - start >= 0.1) segments.push({ start, end });
      }
    }

    // 2. Cut speech into short pieces so a voice change inside a region is still caught.
    const pieces = segments.flatMap(({ start, end }) => {
      const n = Math.max(1, Math.round((end - start) / PIECE_S));
      const len = (end - start) / n;
      return Array.from({ length: n }, (_, k) => ({ start: start + k * len, end: start + (k + 1) * len }));
    });

    // 3. A voiceprint for every piece long enough to have one.
    for (const [i, p] of pieces.entries()) {
      onProgress({ stage: 'voiceprints', done: i, total: pieces.length });
      if (p.end - p.start < MIN_VOICEPRINT_S) continue;
      const clip = audio.subarray(Math.floor(p.start * SAMPLE_RATE), Math.ceil(p.end * SAMPLE_RATE));
      const out = await vpModel(await vpProc(clip));
      p.voiceprint = normalize((out.embeddings ?? out.last_hidden_state ?? Object.values(out)[0]).data);
    }

    // 4. Chain neighbouring pieces that clearly share a voice, then cluster those chains into people.
    const voiced = pieces.filter(p => p.voiceprint);
    const chains = [];
    for (const p of voiced) {
      const c = chains.at(-1);
      if (c && p.start - c.end < 0.5 && dot(p.voiceprint, normalize(c.sum)) > SAME_CHAIN) {
        c.pieces.push(p); c.end = p.end; c.sum = c.sum.map((x, d) => x + p.voiceprint[d]);
      } else {
        chains.push({ pieces: [p], end: p.end, sum: Float32Array.from(p.voiceprint) });
      }
    }
    const { labels } = cluster(chains.map(c => normalize(c.sum)), fixedCount, chains.map(c => c.pieces.length));
    chains.forEach((c, k) => c.pieces.forEach(p => { p.person = labels[k]; }));
    for (const p of pieces) if (p.person === undefined) p.person = nearestInTime(p, voiced) ?? 0;

    // 5. Merge into turns, renumbering people by first appearance.
    const order = new Map();
    const turns = [];
    for (const p of pieces) {
      if (!order.has(p.person)) order.set(p.person, order.size);
      const speaker = order.get(p.person);
      const last = turns.at(-1);
      if (last && last.speaker === speaker && p.start - last.end < 0.05) last.end = p.end;
      else turns.push({ speaker, start: p.start, end: p.end });
    }
    return { turns, speakers: order.size };
  }

  async function run(audio, { quality = 'fast', language = '', speakers = 0 } = {}) {
    const duration = audio.length / SAMPLE_RATE;
    const t0 = Date.now();
    const dia = await diarize(audio, { speakers });
    const t1 = Date.now();
    if (lowMemory) await unloadDiarization();
    const blocks = planBlocks(dia.turns, duration);
    const words = await transcribe(audio, blocks, { quality, language });
    const t2 = Date.now();
    return {
      duration,
      speakers: Math.max(dia.speakers, 1),
      utterances: assignWords(words, dia.turns),
      timings: { diarize: (t1 - t0) / 1000, transcribe: (t2 - t1) / 1000 },
    };
  }

  return { run, transcribe, diarize };
}

// Groups turns into transcription blocks: one speaker each, cut at pauses, at most MAX_BLOCK_S.
// Whisper tends to skip whole sentences in long blocks that switch between voices.
const MAX_BLOCK_S = 25;
const MAX_PAUSE_S = 1;
const PAD_S = 0.3;
export function planBlocks(turns, duration) {
  const blocks = [];
  let cur = null;
  for (const { speaker, start, end } of turns) {
    if (cur && cur[2] === speaker && start - cur[1] < MAX_PAUSE_S && end - cur[0] <= MAX_BLOCK_S) {
      cur[1] = end;
      continue;
    }
    if (cur) blocks.push(cur);
    cur = [start, end, speaker];
    // A single turn longer than a block gets split hard.
    while (cur[1] - cur[0] > MAX_BLOCK_S) {
      blocks.push([cur[0], cur[0] + MAX_BLOCK_S]);
      cur = [cur[0] + MAX_BLOCK_S, cur[1], speaker];
    }
  }
  if (cur) blocks.push(cur);
  return blocks.map(([s, e]) => [Math.max(0, s - PAD_S), Math.min(duration, e + PAD_S)]);
}

function normalize(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, x => x / n);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Weighted average-linkage agglomerative clustering on cosine similarity.
// With fixedCount it merges until that many people remain; otherwise it stops at SAME_PERSON.
export function cluster(vectors, fixedCount = 0, weights = vectors.map(() => 1)) {
  const n = vectors.length;
  const sim = vectors.map(a => vectors.map(b => dot(a, b)));
  const size = [...weights];
  const alive = new Array(n).fill(true);
  const parent = Array.from({ length: n }, (_, i) => i);
  let groups = n;
  while (groups > 1) {
    let best = -Infinity, bi = -1, bj = -1;
    for (let i = 0; i < n; i++) if (alive[i])
      for (let j = i + 1; j < n; j++) if (alive[j] && sim[i][j] > best) { best = sim[i][j]; bi = i; bj = j; }
    if (fixedCount ? groups <= fixedCount : best < SAME_PERSON) break;
    // Lance-Williams update for average linkage: similarity to the merged group is the size-weighted mean.
    for (let k = 0; k < n; k++) if (alive[k] && k !== bi && k !== bj)
      sim[bi][k] = sim[k][bi] = (size[bi] * sim[bi][k] + size[bj] * sim[bj][k]) / (size[bi] + size[bj]);
    size[bi] += size[bj];
    alive[bj] = false;
    parent[bj] = bi;
    groups--;
  }
  const root = i => (parent[i] === i ? i : (parent[i] = root(parent[i])));
  const ids = new Map();
  const labels = vectors.map((_, i) => {
    const r = root(i);
    if (!ids.has(r)) ids.set(r, ids.size);
    return ids.get(r);
  });
  return { labels, count: ids.size };
}

function nearestInTime(piece, others) {
  const mid = (piece.start + piece.end) / 2;
  let best = null, bestGap = Infinity;
  for (const o of others) {
    const gap = mid < o.start ? o.start - mid : mid > o.end ? mid - o.end : 0;
    if (gap < bestGap) { bestGap = gap; best = o.person; }
  }
  return best;
}

// Give each word the speaker whose turn overlaps it most (or the nearest turn),
// then join consecutive words of the same speaker into utterances.
export function assignWords(words, turns) {
  const utterances = [];
  for (const w of words) {
    let best = null, bestScore = -Infinity;
    for (const t of turns) {
      const overlap = Math.min(w.end, t.end) - Math.max(w.start, t.start);
      const score = overlap > 0 ? overlap : overlap - 1000; // prefer any overlap over proximity
      if (score > bestScore) { bestScore = score; best = t.speaker; }
    }
    const speaker = best ?? 0;
    const last = utterances.at(-1);
    if (last && last.speaker === speaker && w.start - last.end < 2) {
      last.text += w.text;
      last.end = w.end;
    } else {
      utterances.push({ speaker, start: w.start, end: w.end, text: w.text.trimStart() });
    }
  }
  return utterances;
}
