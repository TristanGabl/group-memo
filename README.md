# Group Memo

Records a conversation in the browser and writes a transcript of who said what.
Everything runs locally: the speech models download once from Hugging Face and are
cached by the browser; the audio never leaves the device.

## Run

    npm run serve        # then open http://localhost:8000

Microphone access needs `localhost` or HTTPS, so phones need an HTTPS deployment.

## How it works

`engine.js` (shared by the browser worker and the Node tests):

1. **Speech regions**: pyannote segmentation 3.0 over 10 s windows finds where someone speaks.
2. **Voiceprints**: speech is cut into ~2 s pieces; WeSpeaker ResNet34 makes a voiceprint for each.
3. **People**: neighbouring pieces with the same voice are chained, then clustered
   (average linkage, cosine similarity ≥ 0.5, or a fixed number of people).
4. **Words**: Whisper (`base` = Fast, `small` = Accurate) transcribes one block per speaker turn,
   cut at pauses, with word timestamps; each word goes to the speaker turn it overlaps.

`worker.js` runs the engine on WebGPU when available and falls back to WebAssembly.
`app.js` handles recording, decoding to 16 kHz mono, and the transcript UI.

## Test

    npm install
    npm test                                   # engine in Node on test/meeting.wav
    node test/run.mjs file.m4a 4 accurate      # any file, fixed 4 people, Accurate model
    node test/browser.mjs test/meeting.wav     # full app in headless Chrome (server must be running)

`test/meeting.wav` is a 42 s, 3-voice conversation made with macOS `say`.
