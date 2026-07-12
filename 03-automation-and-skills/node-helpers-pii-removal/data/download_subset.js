#!/usr/bin/env node
'use strict';
/*
 * Download a configurable subset (first N rows) of a HuggingFace dataset.
 *
 * Cross-platform. Uses ONLY the Node.js standard library (built-in `fetch`,
 * `node:fs`, `node:path`) plus the sibling `parquet.js` writer — no npm
 * packages, no node_modules.
 *
 * Data comes from the HuggingFace datasets-server REST API (/rows endpoint,
 * 100 rows per request) and the /size endpoint for split discovery.
 *
 * Examples:
 *   node download_subset.js --rows 100
 *   node download_subset.js -n 1000 --split train --format parquet -o sub.parquet
 *   node download_subset.js -n 50 --format csv
 *   node download_subset.js --list-splits
 *
 * Requires Node >= 18 (for the built-in global `fetch`).
 */

const fs = require('node:fs');
const path = require('node:path');
const { writeParquet } = require('./parquet');

const ROWS_API = 'https://datasets-server.huggingface.co/rows';
const SIZE_API = 'https://datasets-server.huggingface.co/size';
const FORMATS = ['jsonl', 'csv', 'parquet'];

function log(msg) { process.stderr.write(msg + '\n'); }
function getToken(token) {
  return token || process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || null;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- HTTP (built-in fetch) with retry/backoff ------------------------------
async function httpGetJson(url, token, timeoutMs = 60000) {
  const headers = { Accept: 'application/json', 'User-Agent': 'hf-data-dl-node/1.0' };
  if (token) headers.Authorization = 'Bearer ' + token;
  for (let attempt = 0; attempt < 5; attempt++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const resp = await fetch(url, { headers, signal: ctrl.signal });
      clearTimeout(timer);
      if (resp.ok) return await resp.json();
      const code = resp.status;
      const body = await resp.text();
      if ([429, 500, 502, 503, 504].includes(code) && attempt < 4) {
        const wait = 1.5 * Math.pow(2, attempt);
        log(`  HTTP ${code}; retrying in ${wait.toFixed(1)}s ...`);
        await sleep(wait * 1000);
        continue;
      }
      throw new Error(`datasets-server returned HTTP ${code}: ${body.slice(0, 500)}`);
    } catch (e) {
      clearTimeout(timer);
      const msg = e && e.message ? e.message : String(e);
      if (msg.startsWith('datasets-server returned')) throw e; // hard HTTP error
      if (attempt < 4) { log(`  network error: ${msg}; retrying...`); await sleep(1500 * (attempt + 1)); continue; }
      throw new Error('network error after retries: ' + msg);
    }
  }
  throw new Error('network error: exhausted retries');
}

// --- row fetching (paginate the /rows endpoint) ---------------------------
async function fetchRows({ dataset, config, split, offset, n, token, maxChunk = 100 }) {
  const enc = encodeURIComponent;
  const rows = [];
  let fetched = 0, cur = offset;
  while (fetched < n) {
    const length = Math.min(maxChunk, n - fetched);
    const url = `${ROWS_API}?dataset=${enc(dataset)}&config=${enc(config)}&split=${enc(split)}&offset=${cur}&length=${length}`;
    const data = await httpGetJson(url, token);
    const chunk = data.rows || [];
    if (chunk.length === 0) break; // end of split
    for (const item of chunk) rows.push(item.row);
    fetched += chunk.length;
    cur += chunk.length;
    log(`  fetched ${fetched}/${n} rows`);
    if (chunk.length < length) break; // fewer than requested => end of split
  }
  return rows;
}

// --- split discovery -------------------------------------------------------
async function listSplits(dataset, token) {
  const url = `${SIZE_API}?dataset=${encodeURIComponent(dataset)}`;
  let info;
  try { info = (await httpGetJson(url, token)).size || {}; }
  catch (e) { log(`Error: ${e.message}`); process.exit(1); }
  const total = info.dataset && info.dataset.num_rows != null ? info.dataset.num_rows : '?';
  console.log(`Dataset: ${dataset}`);
  console.log(`Total rows: ${total}\n`);
  console.log(`${'config'.padEnd(22)} ${'split'.padEnd(14)} ${'rows'.padStart(10)}  ${'parquet MB'.padStart(10)}`);
  console.log('-'.repeat(62));
  for (const s of info.splits || []) {
    const mb = ((s.num_bytes_parquet_files || 0) / 1e6).toFixed(1);
    const rows = s.num_rows != null ? s.num_rows : '?';
    console.log(`${String(s.config || '').padEnd(22)} ${String(s.split || '').padEnd(14)} ${String(rows).padStart(10)}  ${mb.padStart(10)}`);
  }
}

// --- output writers --------------------------------------------------------
function saveJsonl(rows, file) {
  const fd = fs.openSync(file, 'w');
  for (const row of rows) fs.writeSync(fd, JSON.stringify(row) + '\n');
  fs.closeSync(fd);
}

function saveCsv(rows, file) {
  const fd = fs.openSync(file, 'w');
  const fieldnames = [];
  const seen = new Set();
  for (const r of rows) for (const k of Object.keys(r)) if (!seen.has(k)) { seen.add(k); fieldnames.push(k); }
  const esc = (v) => {
    let s;
    if (v === null || v === undefined) s = '';
    else if (typeof v === 'string') s = v;
    else s = JSON.stringify(v);
    if (/[",\n\r]/.test(s)) s = '"' + s.replace(/"/g, '""') + '"';
    return s;
  };
  fs.writeSync(fd, fieldnames.join(',') + '\n');
  for (const r of rows) fs.writeSync(fd, fieldnames.map((k) => esc(r[k])).join(',') + '\n');
  fs.closeSync(fd);
}

// --- CLI -------------------------------------------------------------------
function printHelp() {
  const h = `Usage: node download_subset.js [options]

Download a subset (first N rows) of a HuggingFace dataset using only the
Node.js standard library (built-in fetch; no npm packages).

Options:
  --dataset ID        HF dataset id (default: nvidia/Nemotron-PII)
  --config NAME       dataset config name (default: default)
  --split NAME        split name: train/test/validation/... (default: train)
  -n, --rows N        number of rows to download (default: 100)
  --offset N          skip this many rows first (default: 0)
  --format F          output format: jsonl|csv|parquet (default: jsonl)
  -o, --output FILE   output file path (auto-named if omitted)
  --token TOKEN       HF token (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN env)
  --list-splits       print available configs/splits/sizes and exit
  -h, --help          show this help

Examples:
  node download_subset.js --rows 100
  node download_subset.js -n 1000 --split train --format parquet -o sub.parquet
  node download_subset.js --list-splits

Requires Node >= 18 (built-in global fetch). Parquet output is written by a
pure-standard-library parquet writer (sibling parquet.js).`;
  console.log(h);
}

function parseArgs(argv) {
  const args = {
    dataset: 'nvidia/Nemotron-PII', config: 'default', split: 'train',
    rows: 100, offset: 0, format: 'jsonl', output: null, token: null, listSplits: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-h' || a === '--help') { printHelp(); process.exit(0); }
    let name, inlineVal;
    const eq = a.indexOf('=');
    if (a.startsWith('--') && eq !== -1) { name = a.slice(0, eq); inlineVal = a.slice(eq + 1); }
    else { name = a; }
    if (name[0] !== '-') throw new Error(`unexpected argument: ${a}`);
    const take = () => {
      if (inlineVal !== undefined) { const v = inlineVal; inlineVal = undefined; return v; }
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${name}`);
      return v;
    };
    switch (name) {
      case '--dataset': args.dataset = take(); break;
      case '--config': args.config = take(); break;
      case '--split': args.split = take(); break;
      case '-n': case '--rows': args.rows = parseInt(take(), 10); break;
      case '--offset': args.offset = parseInt(take(), 10); break;
      case '--format': args.format = take(); break;
      case '-o': case '--output': args.output = take(); break;
      case '--token': args.token = take(); break;
      case '--list-splits': args.listSplits = true; break;
      default: throw new Error(`unknown argument: ${name}`);
    }
  }
  return args;
}

async function main() {
  let args;
  try { args = parseArgs(process.argv.slice(2)); }
  catch (e) { log(`Error: ${e.message}\nRun with --help for usage.`); process.exit(2); }

  if (!FORMATS.includes(args.format)) { log(`Error: invalid --format '${args.format}' (choose: ${FORMATS.join(', ')})`); process.exit(2); }
  if (Number.isNaN(args.rows) || !Number.isInteger(args.rows) || args.rows <= 0) {
    log('Error: --rows must be a positive integer'); process.exit(2);
  }

  const token = getToken(args.token);

  if (args.listSplits) { await listSplits(args.dataset, token); return; }

  log(`Downloading ${args.rows} rows from ${args.dataset} [${args.config}/${args.split}] starting at offset ${args.offset} ...`);
  let rows;
  try {
    rows = await fetchRows({ dataset: args.dataset, config: args.config, split: args.split, offset: args.offset, n: args.rows, token });
  } catch (e) { log(`Error: ${e.message}`); process.exit(1); }

  if (rows.length === 0) { log('No rows returned (offset may be past the end of the split).'); return; }
  const got = rows.length;

  const out = args.output
    ? path.resolve(args.output)
    : path.resolve(`${args.dataset.replace(/\//g, '__')}_${args.split}_${got}rows_offset${args.offset}.${args.format}`);
  fs.mkdirSync(path.dirname(out), { recursive: true });

  try {
    if (args.format === 'jsonl') saveJsonl(rows, out);
    else if (args.format === 'csv') saveCsv(rows, out);
    else writeParquet(rows, out);
  } catch (e) { log(`Error: ${e.message}`); process.exit(1); }

  const size = fs.statSync(out).size;
  const cols = Object.keys(rows[0]);
  log(`\nDone: ${got} rows -> ${out} (${(size / 1024).toFixed(1)} KiB)`);
  log(`Columns (${cols.length}): ${cols.join(', ')}`);
}

main().catch((e) => { log(`Error: ${e && e.message ? e.message : e}`); process.exit(1); });
