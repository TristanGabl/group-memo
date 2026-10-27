// Drives headless Chrome through the DevTools protocol: loads the app, uploads a file,
// waits for the transcript and prints it. Usage: [QUALITY=accurate] [MOBILE=1] [MODE=auto|light|full] [URL=…] node test/browser.mjs <audio file> [screenshot.png]
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const file = resolve(process.argv[2] ?? 'test/meeting.wav');
// Each run gets its own profile and a free port, so no state leaks between runs.
const profile = mkdtempSync(join(tmpdir(), 'gm-'));
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--no-first-run', 'about:blank',
], { stdio: 'ignore' });

process.on('exit', () => chrome.kill('SIGKILL'));
let targets;
for (let i = 0; i < 50 && !targets; i++) {
  await new Promise(r => setTimeout(r, 200));
  try {
    const port = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0];
    targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  } catch {}
}
const ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0;
const pending = new Map();
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') console.error('EXCEPTION', m.params.exceptionDetails.exception?.description);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') console.error('console.error', m.params.args.map(a => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async expr => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result?.result?.value;

await send('Runtime.enable');
await send('Page.enable');
// MOBILE=1 pretends to be an iPhone, to exercise the low-memory path.
if (process.env.MOBILE) await send('Emulation.setUserAgentOverride', { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1' });
await send('Page.navigate', { url: process.env.URL ?? 'http://127.0.0.1:8000/' });
await new Promise(r => setTimeout(r, 2000));
const { root } = (await send('DOM.getDocument')).result;
const { nodeId } = (await send('DOM.querySelector', { nodeId: root.nodeId, selector: '#file' })).result;
if (process.env.MODE) await evaluate(`(() => { const m = document.getElementById('mode'); m.value = ${JSON.stringify(process.env.MODE)}; m.dispatchEvent(new Event('change')); })()`);
console.log('mode hint:', await evaluate(`document.getElementById('mode-hint').textContent || '(manual)'`), '|', await evaluate(`document.getElementById('version').textContent`));
if (process.env.QUALITY) await evaluate(`document.getElementById('quality').value = ${JSON.stringify(process.env.QUALITY)}`);
await send('DOM.setFileInputFiles', { nodeId, files: [file] });

const t0 = Date.now();
let last = '';
while (Date.now() - t0 < 600000) {
  await new Promise(r => setTimeout(r, 1000));
  const s = await evaluate(`JSON.stringify({ stage: document.getElementById('stage').textContent, detail: document.getElementById('stage-detail').textContent, device: document.getElementById('device').textContent, error: document.getElementById('error').hidden ? '' : document.getElementById('error').textContent, done: !document.getElementById('result').hidden })`);
  const st = JSON.parse(s);
  const line = `${st.stage} ${st.detail} | ${st.device}`;
  if (line !== last) { console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s  ${line}`); last = line; }
  if (st.error) { console.log('ERROR:', st.error); break; }
  if (st.done) {
    console.log(await evaluate(`document.getElementById('summary-line').textContent + '\\n' + [...document.querySelectorAll('.line')].map(l => l.innerText.replace(/\\n/g, ' ')).join('\\n')`));
    await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 1100, deviceScaleFactor: 2, mobile: true });
    const shot = (await send('Page.captureScreenshot', { format: 'png' })).result.data;
    (await import('node:fs')).writeFileSync(process.argv[3] ?? 'test/screenshot.png', Buffer.from(shot, 'base64'));
    break;
  }
}
ws.close();
process.exit(0);
