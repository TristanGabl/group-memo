// Page logic: capture audio, decode it to 16 kHz mono, hand it to the worker,
// and render the speaker-labelled transcript.
const SAMPLE_RATE = 16000;
const STORE_KEY = 'group-memo:last';
const $ = id => document.getElementById(id);

const ui = {
  record: $('record'), recordLabel: $('record-label'), timer: $('timer'), level: $('level'),
  file: $('file'), language: $('language'), quality: $('quality'), speakers: $('speakers'),
  progress: $('progress'), stage: $('stage'), device: $('device'), barFill: $('bar-fill'), stageDetail: $('stage-detail'),
  error: $('error'), empty: $('empty'), result: $('result'), summaryLine: $('summary-line'), player: $('player'),
  people: $('people'), lines: $('lines'), copy: $('copy'), download: $('download'),
  summarize: $('summarize'), reprocess: $('reprocess'), summary: $('summary'),
};

const state = { result: null, names: [], audio: null, audioUrl: null, busy: false };
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

// Phones and tablets have far less memory per tab than laptops.
const LOW_MEMORY = navigator.userAgentData?.mobile
  || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
  || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
  || (navigator.deviceMemory ?? 8) <= 4;
if (LOW_MEMORY) {
  const accurate = ui.quality.querySelector('option[value="accurate"]');
  accurate.disabled = true;
  accurate.textContent = 'Accurate · laptops and desktops only';
  ui.quality.querySelector('option[value="fast"]').textContent = 'Fast · 115 MB download';
  ui.quality.value = 'fast';
  document.querySelector('.fast-size').textContent = 'about 115 MB';
}

// ---------- Recording ----------

let rec = null;

ui.record.addEventListener('click', () => (rec ? stopRecording() : startRecording()));

async function startRecording() {
  hideError();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, autoGainControl: true, noiseSuppression: true } });
  } catch (err) {
    return showError(err.name === 'NotAllowedError'
      ? 'Microphone access was blocked. Allow the microphone for this site in the browser settings, then try again.'
      : `The microphone could not be started (${err.message}).`);
  }
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
  recorder.onstop = () => handleAudio(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }));
  recorder.start(1000);

  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  ctx.createMediaStreamSource(stream).connect(analyser);

  rec = { recorder, stream, ctx, analyser, startedAt: Date.now(), wakeLock: null, raf: 0, tick: 0 };
  try { rec.wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* optional */ }

  ui.record.setAttribute('aria-pressed', 'true');
  ui.recordLabel.textContent = 'Stop';
  ui.file.disabled = true;
  rec.tick = setInterval(() => { ui.timer.textContent = clock((Date.now() - rec.startedAt) / 1000); }, 250);
  drawLevel();
}

function stopRecording() {
  const r = rec;
  rec = null;
  clearInterval(r.tick);
  cancelAnimationFrame(r.raf);
  r.recorder.stop();
  r.stream.getTracks().forEach(t => t.stop());
  r.ctx.close();
  r.wakeLock?.release().catch(() => {});
  ui.record.setAttribute('aria-pressed', 'false');
  ui.recordLabel.textContent = 'Start recording';
  ui.file.disabled = false;
  clearLevel();
}

// Keep the screen on again if the browser dropped the lock while the tab was hidden.
document.addEventListener('visibilitychange', async () => {
  if (rec && document.visibilityState === 'visible' && (!rec.wakeLock || rec.wakeLock.released)) {
    try { rec.wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* optional */ }
  }
});

function drawLevel() {
  const c = ui.level, g = c.getContext('2d');
  const data = new Uint8Array(rec.analyser.fftSize);
  const history = [];
  const color = getComputedStyle(document.documentElement).getPropertyValue('--rec');
  const frame = () => {
    if (!rec) return;
    rec.analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (const v of data) peak = Math.max(peak, Math.abs(v - 128) / 128);
    history.push(peak);
    if (history.length > 60) history.shift();
    g.clearRect(0, 0, c.width, c.height);
    g.fillStyle = color;
    history.forEach((v, i) => {
      const h = Math.max(2, v * c.height);
      g.fillRect(i * 4, (c.height - h) / 2, 2, h);
    });
    rec.raf = requestAnimationFrame(frame);
  };
  frame();
}

function clearLevel() {
  ui.level.getContext('2d').clearRect(0, 0, ui.level.width, ui.level.height);
}

// ---------- Upload ----------

ui.file.addEventListener('change', () => {
  const f = ui.file.files[0];
  if (f) handleAudio(f);
  ui.file.value = '';
});

// ---------- Processing ----------

async function handleAudio(blob) {
  hideError();
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = URL.createObjectURL(blob);
  showProgress('Preparing audio');
  try {
    state.audio = await decode(blob);
  } catch {
    hideProgress();
    return showError('This file could not be read as audio. Try an m4a, mp3, wav or webm file.');
  }
  if (state.audio.length < SAMPLE_RATE) {
    hideProgress();
    return showError('The recording is shorter than one second. Record a little longer and try again.');
  }
  process();
}

async function decode(blob) {
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  ctx.close();
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * SAMPLE_RATE), SAMPLE_RATE);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  return (await offline.startRendering()).getChannelData(0);
}

function process() {
  if (state.busy || !state.audio) return;
  state.busy = true;
  setControlsDisabled(true);
  ui.result.hidden = true;
  showProgress('Loading speech models');
  worker.postMessage({
    type: 'run',
    lowMemory: LOW_MEMORY,
    // Copy so the original stays available for "Process again".
    audio: state.audio.slice(),
    options: { quality: ui.quality.value, language: ui.language.value, speakers: Number(ui.speakers.value) },
  });
}

function failed(message) {
  state.busy = false;
  setControlsDisabled(false);
  hideProgress();
  render();
  showError(message);
}

// Fires when the worker itself dies, for example when the browser stops it for using too much memory.
worker.onerror = e => {
  e.preventDefault?.();
  failed('The speech engine stopped unexpectedly, probably because the device ran out of memory. Reload the page, close other tabs and try a shorter recording.');
};

ui.reprocess.addEventListener('click', process);

const STAGES = {
  download: 'Downloading speech models (first time only)',
  diarize: 'Finding where people speak',
  voiceprints: 'Telling voices apart',
  transcribe: 'Transcribing',
};

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case 'device':
      ui.device.textContent = data.device === 'webgpu' ? 'Running on the GPU (WebGPU)'
        : LOW_MEMORY ? 'Phone mode: smaller models on the CPU'
        : 'Running on the CPU (slower; WebGPU not available)';
      break;
    case 'progress': {
      ui.stage.textContent = STAGES[data.stage] ?? data.stage;
      if (data.stage === 'download') {
        setBar(data.loaded / data.total);
        ui.stageDetail.textContent = `${mb(data.loaded)} of ${mb(data.total)}`;
      } else if (data.total) {
        setBar(data.done / data.total);
        ui.stageDetail.textContent = data.stage === 'transcribe' ? `Part ${data.done + 1} of ${data.total}` : '';
      }
      break;
    }
    case 'result':
      if (!data.result.utterances.length) {
        failed('No speech was found in this recording. Check that the right microphone is selected and that people speak loudly enough, then record again.');
        break;
      }
      state.busy = false;
      setControlsDisabled(false);
      hideProgress();
      state.result = data.result;
      state.names = Array.from({ length: data.result.speakers }, (_, i) => `Speaker ${i + 1}`);
      ui.summary.hidden = true;
      save();
      render();
      break;
    case 'error':
      failed(`Processing failed: ${data.message}. Reload the page and try again; switching Accuracy to Fast helps on devices with little memory.`);
      break;
  }
};

// ---------- Rendering ----------

function render() {
  const r = state.result;
  if (!r) return;
  ui.empty.hidden = true;
  ui.result.hidden = false;
  ui.reprocess.hidden = !state.audio;
  ui.player.hidden = !state.audioUrl;
  if (state.audioUrl) ui.player.src = state.audioUrl;

  const talk = new Array(r.speakers).fill(0);
  for (const u of r.utterances) talk[u.speaker] += u.end - u.start;
  ui.summaryLine.textContent = `${clock(r.duration)} · ${r.speakers} ${r.speakers === 1 ? 'person' : 'people'}`;

  ui.people.replaceChildren(...state.names.map((name, i) => {
    const el = document.createElement('label');
    el.className = 'person';
    el.style.setProperty('--c', color(i));
    el.innerHTML = '<i aria-hidden="true"></i><input type="text" autocomplete="off"><small></small>';
    const input = el.querySelector('input');
    input.id = `name-${i}`;
    input.value = name;
    input.setAttribute('aria-label', `Name for speaker ${i + 1}`);
    input.addEventListener('input', () => {
      state.names[i] = input.value;
      document.querySelectorAll(`.who[data-s="${i}"]`).forEach(w => { w.textContent = displayName(i); });
      save();
    });
    el.querySelector('small').textContent = clock(talk[i]);
    return el;
  }));

  ui.lines.replaceChildren(...r.utterances.map(u => {
    const li = document.createElement('li');
    li.className = 'line';
    li.style.setProperty('--c', color(u.speaker));
    li.innerHTML = '<button class="at" type="button"></button><div><span class="who"></span><p></p></div>';
    const at = li.querySelector('.at');
    at.textContent = clock(u.start);
    at.title = 'Play from here';
    at.disabled = !state.audioUrl;
    at.addEventListener('click', () => { ui.player.currentTime = u.start; ui.player.play(); });
    const who = li.querySelector('.who');
    who.dataset.s = u.speaker;
    who.textContent = displayName(u.speaker);
    li.querySelector('p').textContent = u.text;
    return li;
  }));
}

const displayName = i => state.names[i]?.trim() || `Speaker ${i + 1}`;
const color = i => `var(--s${(i % 8) + 1})`;

function transcriptText() {
  return state.result.utterances.map(u => `[${clock(u.start)}] ${displayName(u.speaker)}: ${u.text}`).join('\n');
}

// ---------- Actions ----------

ui.copy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(transcriptText());
    flash(ui.copy, 'Copied');
  } catch {
    showError('Copying is blocked in this browser. Use Download .txt instead.');
  }
});

ui.download.addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([transcriptText() + '\n'], { type: 'text/plain' }));
  a.download = `group-memo-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// Chrome's built-in Summarizer runs Gemini Nano on the device, so it keeps the local-only promise.
if ('Summarizer' in self) {
  Summarizer.availability().then(a => { ui.summarize.hidden = a === 'unavailable'; }).catch(() => {});
}
ui.summarize.addEventListener('click', async () => {
  ui.summarize.disabled = true;
  ui.summary.hidden = false;
  ui.summary.textContent = 'Summarizing …';
  try {
    const s = await Summarizer.create({
      type: 'key-points', format: 'plain-text', length: 'medium',
      sharedContext: 'A transcript of a group conversation with speaker names.',
      monitor: m => m.addEventListener('downloadprogress', e => { ui.summary.textContent = `Downloading the summary model … ${Math.round(e.loaded * 100)}%`; }),
    });
    ui.summary.textContent = await s.summarize(transcriptText());
    s.destroy();
  } catch (err) {
    ui.summary.textContent = `The summary could not be created (${err.message}).`;
  } finally {
    ui.summarize.disabled = false;
  }
});

// ---------- Helpers ----------

function showProgress(text) {
  ui.progress.hidden = false;
  ui.stage.textContent = text;
  ui.stageDetail.textContent = '';
  setBar(null);
}
function hideProgress() { ui.progress.hidden = true; }
function setBar(fraction) {
  ui.barFill.classList.toggle('indeterminate', fraction == null);
  ui.barFill.style.width = fraction == null ? '' : `${Math.round(Math.min(1, fraction) * 100)}%`;
}
function setControlsDisabled(on) {
  ui.record.disabled = on;
  ui.file.disabled = on;
  ui.reprocess.disabled = on;
}
function showError(msg) { ui.error.textContent = msg; ui.error.hidden = false; }
function hideError() { ui.error.hidden = true; }
function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = old; }, 1500);
}
function clock(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = String(m).padStart(2, '0'), ss = String(sec).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
const mb = bytes => `${Math.round(bytes / 1e6)} MB`;

// The last transcript (not the audio) is kept in this browser so a reload doesn't lose it.
function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ result: state.result, names: state.names })); } catch { /* optional */ }
}
try {
  const saved = JSON.parse(localStorage.getItem(STORE_KEY));
  if (saved?.result) { state.result = saved.result; state.names = saved.names; render(); }
} catch { /* optional */ }
