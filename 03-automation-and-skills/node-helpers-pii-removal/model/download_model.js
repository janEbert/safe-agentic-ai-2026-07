#!/usr/bin/env node
'use strict';
/*
 * Download all files of a HuggingFace *model* repository using only the
 * Node.js standard library (built-in `fetch`; no npm packages).
 *
 * Example:
 *   node download_model.js --repo nvidia/gliner-PII
 *   node download_model.js --repo nvidia/gliner-PII --out-dir ./model --include '*.json,*.bin'
 *
 * Features: resumable (HTTP Range), progress bar, skips complete files,
 * verifies Content-Length, HF token support for gated/private models.
 * Requires Node >= 18.
 */

const fs = require('node:fs');
const path = require('node:path');

// The HF API wants the repo id verbatim (do NOT encode the '/').
const API = (repo) => `https://huggingface.co/api/models/${repo}`;
// Only individual path segments of the filename need encoding.
const RESOLVE = (repo, file) => `https://huggingface.co/${repo}/resolve/main/${file.split('/').map(encodeURIComponent).join('/')}`;

function getToken(token) {
  return token || process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || null;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getModelFiles(repo, token) {
  const headers = { Accept: 'application/json' };
  if (token) headers.Authorization = 'Bearer ' + token;
  for (let i = 0; i < 5; i++) {
    try {
      const r = await fetch(API(repo), { headers });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
      const data = await r.json();
      return (data.siblings || []).map((s) => s.rfilename).filter((f) => !f.startsWith('.'));
    } catch (e) {
      if (i === 4) throw e;
      log(`listing files failed: ${e.message}; retrying...`);
      await sleep(1500 * (i + 1));
    }
  }
}

function log(m) { process.stderr.write(m + '\n'); }

function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 ** 2) return (n / 1024).toFixed(1) + ' KiB';
  if (n < 1024 ** 3) return (n / 1024 ** 2).toFixed(1) + ' MiB';
  return (n / 1024 ** 3).toFixed(2) + ' GiB';
}

// HEAD the resolved URL to get the final size (follows redirects to the CDN).
async function getSize(url, token) {
  const headers = {};
  if (token) headers.Authorization = 'Bearer ' + token;
  for (let i = 0; i < 3; i++) {
    try {
      const r = await fetch(url, { method: 'HEAD', headers });
      if (r.ok) {
        const len = Number(r.headers.get('content-length'));
        if (len > 0) return len;
      }
      // some CDNs don't support HEAD well; fall through to GET range below
      return null;
    } catch (e) { if (i === 2) return null; await sleep(1000 * (i + 1)); }
  }
  return null;
}

async function downloadFile(repo, file, outDir, token, totalIdx, total) {
  const url = RESOLVE(repo, file);
  const local = path.join(outDir, file);
  fs.mkdirSync(path.dirname(local), { recursive: true });

  const remoteSize = await getSize(url, token);
  let have = 0;
  if (fs.existsSync(local)) have = fs.statSync(local).size;

  if (remoteSize && have === remoteSize) {
    log(`  [${totalIdx}/${total}] ${file}  (${fmtBytes(have)}) -- already complete, skip`);
    return;
  }

  const resume = have > 0;
  const headers = {};
  if (token) headers.Authorization = 'Bearer ' + token;
  if (resume) headers.Range = `bytes=${have}-`;

  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      const r = await fetch(url, { headers });
      if (!r.ok && r.status !== 206) {
        if (r.status === 416) { // range not satisfiable -> already complete
          log(`  [${totalIdx}/${total}] ${file}  (already complete)`);
          return;
        }
        throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
      }
      const contentLength = Number(r.headers.get('content-length') || 0);
      const mode = resume && r.status === 206 ? 'a' : 'w';
      if (mode === 'w' && fs.existsSync(local)) fs.unlinkSync(local);
      const out = fs.createWriteStream(local, { flags: mode });
      const startHave = mode === 'a' ? have : 0;
      const totalSize = remoteSize || (resume ? have + contentLength : contentLength);
      let got = startHave;
      let lastTick = 0;

      const reader = r.body.getReader();
      const tick = () => {
        const now = Date.now();
        if (now - lastTick < 250 && got < totalSize) return;
        lastTick = now;
        const pct = totalSize ? ((got / totalSize) * 100).toFixed(1) : '?';
        process.stderr.write(`\r  [${totalIdx}/${total}] ${file}  ${fmtBytes(got)}/${fmtBytes(totalSize)} (${pct}%)   `);
      };
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        await new Promise((res, rej) => {
          out.write(Buffer.from(value), (err) => err ? rej(err) : res());
        });
        got += value.length;
        tick();
      }
      await new Promise((res, rej) => out.end((e) => e ? rej(e) : res()));
      process.stderr.write('\n');
      const finalSize = fs.statSync(local).size;
      if (remoteSize && finalSize !== remoteSize) {
        throw new Error(`size mismatch: got ${finalSize}, expected ${remoteSize}`);
      }
      log(`  [${totalIdx}/${total}] ${file}  -> ${fmtBytes(finalSize)} done`);
      return;
    } catch (e) {
      if (fs.existsSync(local) && mode !== 'a') { /* keep partial for resume */ }
      if (attempt === 5) throw e;
      log(`\n  ${file}: ${e.message}; retrying...`);
      have = fs.existsSync(local) ? fs.statSync(local).size : 0;
      await sleep(1500 * (2 ** attempt));
    }
  }
}

async function main() {
  const args = process.argv.slice(2);
  const opts = { repo: 'nvidia/gliner-PII', outDir: null, token: null, include: null, exclude: null, list: false };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '-h' || a === '--help') { printHelp(); return; }
    if (a === '--list') { opts.list = true; }
    else if (a === '--repo') opts.repo = args[++i];
    else if (a === '--out-dir') opts.outDir = args[++i];
    else if (a === '--token') opts.token = args[++i];
    else if (a === '--include') opts.include = args[++i];
    else if (a === '--exclude') opts.exclude = args[++i];
    else if (a.startsWith('--')) { const [k, v] = a.includes('=') ? a.split('=') : [a, args[++i]]; /* ignore unknown */ }
    else if (a.includes('=')) { const [k, v] = a.split('='); if (k === '--repo') opts.repo = v; else if (k === '--out-dir') opts.outDir = v; else if (k === '--token') opts.token = v; else if (k === '--include') opts.include = v; else if (k === '--exclude') opts.exclude = v; }
  }
  if (!opts.outDir) opts.outDir = './' + opts.repo.split('/').pop();

  const token = getToken(opts.token);
  log(`Model: ${opts.repo}`);
  let files;
  try { files = await getModelFiles(opts.repo, token); }
  catch (e) { log(`Error listing files: ${e.message}`); process.exit(1); }

  if (opts.include) {
    const pats = opts.include.split(',').map((s) => s.trim());
    files = files.filter((f) => pats.some((p) => matchGlob(f, p)));
  }
  if (opts.exclude) {
    const pats = opts.exclude.split(',').map((s) => s.trim());
    files = files.filter((f) => !pats.some((p) => matchGlob(f, p)));
  }

  if (opts.list) { log('Files:'); for (const f of files) log('  ' + f); return; }

  log(`Output dir: ${opts.outDir}`);
  log(`Downloading ${files.length} file(s)...\n`);
  for (let i = 0; i < files.length; i++) {
    await downloadFile(opts.repo, files[i], opts.outDir, token, i + 1, files.length);
  }
  log('\nAll done. Files in ' + opts.outDir);
}

function matchGlob(name, pat) {
  // minimal glob: * and ?; matches the whole string
  const re = '^' + pat.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.') + '$';
  return new RegExp(re).test(name);
}

function printHelp() {
  console.log(`Usage: node download_model.js [options]

Download all files of a HuggingFace model repository (stdlib-only).

Options:
  --repo ID        HF model repo id (default: nvidia/gliner-PII)
  --out-dir DIR    output directory (default: ./<repo-name>)
  --token TOKEN    HF token (or set HF_TOKEN env)
  --include PATS   comma-separated glob patterns to include (e.g. '*.json,*.bin')
  --exclude PATS   comma-separated glob patterns to exclude
  --list           list files that would be downloaded and exit
  -h, --help       show this help

Examples:
  node download_model.js --repo nvidia/gliner-PII
  node download_model.js --include '*.json,spm.model,pytorch_model.bin'
`);
}

main().catch((e) => { log('Error: ' + (e.message || e)); process.exit(1); });
