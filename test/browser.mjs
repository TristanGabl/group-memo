// Drives headless Chrome through the DevTools protocol: loads the app, uploads a file,
// waits for the transcript and prints it. Usage: [QUALITY=accurate] node test/browser.mjs <audio file> [screenshot.png]
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const file = resolve(process.argv[2] ?? 'test/meeting.wav');
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--remote-debugging-port=9333', `--user-data-dir=${mkdtempSync(join(tmpdir(), 'gm-'))}`,
  '--enable-unsafe-webgpu', '--no-first-run', 'about:blank',
], { stdio: 'ignore' });

let targets;
for (let i = 0; i < 50 && !targets; i++) {
  await new Promise(r => setTimeout(r, 200));
  try { targets = await (await fetch('http://127.0.0.1:9333/json')).json(); } catch {}
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
await send('Page.navigate', { url: 'http://127.0.0.1:8000/' });
await new Promise(r => setTimeout(r, 2000));
const { root } = (await send('DOM.getDocument')).result;
const { nodeId } = (await send('DOM.querySelector', { nodeId: root.nodeId, selector: '#file' })).result;
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
chrome.kill();
