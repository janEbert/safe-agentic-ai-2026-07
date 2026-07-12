#!/usr/bin/env node
'use strict';
/*
 * GLiNER inference pipeline (pure Node.js stdlib): text + labels -> entities.
 *
 * Pipeline:
 *   whitespace word-split (with char offsets)
 *   -> prompt = [<<ENT>> label ...] + [<<SEP>>]
 *   -> tokenize (is_split_into_words) -> input_ids, word_ids
 *   -> words_mask (skip prompt), span_idx, span_mask
 *   -> forward -> logits (W, K, C)
 *   -> sigmoid, threshold, valid-span, greedy non-overlap -> char spans
 */
const fs = require('node:fs');
const path = require('node:path');
const { loadStateDict, loadStateDictShared } = require('./torch_load');
const { GLiNERModel } = require('./model');
const { Tokenizer } = require('./tokenizer');
const wasmKernel = require('./wasm-kernel');
const T = require('./tensor');

const ENT = '<<ENT>>';
const SEP = '<<SEP>>';

// Whitespace word splitter (matches GLiNER's regex `\w+(?:[-_]\w+)*|\S`).
function splitWords(text) {
  const re = /\w+(?:[-_]\w+)*|\S/gu;
  const words = [], starts = [], ends = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    words.push(m[0]); starts.push(m.index); ends.push(m.index + m[0].length);
  }
  return { words, starts, ends };
}

// prepare_span_idx(numWords, max_width): (numWords*max_width, 2) [start, start+offset]
function prepareSpanIdx(n, maxW) {
  const idx = new Int32Array(n * maxW * 2);
  let p = 0;
  for (let s = 0; s < n; s++) for (let w = 0; w < maxW; w++) { idx[p++] = s; idx[p++] = s + w; }
  return idx;
}

class GLiNER {
  constructor(modelDir) {
    this.dir = modelDir;
    const g = JSON.parse(fs.readFileSync(path.join(modelDir, 'gliner_config.json'), 'utf8'));
    const e = g.encoder_config;
    this.cfg = Object.assign({}, e, { max_width: g.max_width, class_token_index: g.class_token_index, hidden_size_gliner: g.hidden_size });
    this.maxWidth = g.max_width;
    this.tokenizer = new Tokenizer(path.join(modelDir, 'tokenizer.json'));
    this._sd = null;
    this.fast = false;
  }
  loadWeights({ fast = false } = {}) {
    if (!this._sd) {
      if (fast) {
        if (!wasmKernel.simdAvailable()) {
          console.error('warning: WebAssembly f32x4 SIMD unavailable; --fast ignored (using JS kernel)');
          this._sd = loadStateDict(path.join(this.dir, 'pytorch_model.bin'));
          this.model = new GLiNERModel(this._sd, this.cfg);
          return this;
        }
        const bin = path.join(this.dir, 'pytorch_model.bin');
        // First pass: total weight floats + largest linear output to size the shared memory.
        const { loadArchive } = require('./torch_load');
        const { descriptors } = loadArchive(bin);
        let total = 0;
        for (const d of Object.values(descriptors)) total += d.size.reduce((a, b) => a * b, 1);
        // Largest matmul output across the model: encoder intermediate L*4096 (L up to
        // ~max_len+prompt) and span out_project (W*max_width)*2048. Size scratch from the
        // absolute cap (max_len=384, so L<=~459; spans W<=384, *12) with headroom, so no
        // forward ever triggers Memory.grow (which would detach cached weight views).
        const maxL = 512, maxSpans = 384 * 12;
        const maxOut = Math.max(maxL * 4096, maxSpans * 2048, maxL * 1024);
        const scratchEach = (maxOut + (1 << 20)) | 0; // one region for input, one for output
        const PAGE = 65536; // bytes
        const needBytes = total * 4 + 2 * scratchEach * 4;
        const initPages = Math.ceil(needBytes / PAGE);
        const mem = new WebAssembly.Memory({ initial: initPages, maximum: 65536 });
        const heap = new Float32Array(mem.buffer, 0, Math.floor(mem.buffer.byteLength / 4));
        const { sd } = loadStateDictShared(bin, heap);
        const scratchX = Math.ceil(total / 4) * 4;
        const scratchO = scratchX + scratchEach;
        wasmKernel.init(mem, scratchX, scratchO);
        T.setKernel(wasmKernel);
        this._sd = sd;
        this.fast = true;
      } else {
        this._sd = loadStateDict(path.join(this.dir, 'pytorch_model.bin'));
      }
      this.model = new GLiNERModel(this._sd, this.cfg);
      if (fast) this.model._fastHeap = () => wasmKernel.heapView();
    }
    return this;
  }

  predict(text, labels, threshold = 0.5) {
    const { words, starts, ends } = splitWords(text);
    if (words.length === 0) return [];
    const entityTypes = [...new Set(labels)];
    // prompt words
    const prompt = [];
    for (const lbl of entityTypes) { prompt.push(ENT, lbl); }
    prompt.push(SEP);
    const skipWords = prompt.length;
    const seq = prompt.concat(words);
    const { ids, wordIds } = this.tokenizer.encodeWords(seq);
    // words_mask: 1-indexed text-word index after skipping prompt words
    const wordsMask = new Int32Array(ids.length);
    let prev = -1, seen = 0;
    for (let t = 0; t < ids.length; t++) {
      const wid = wordIds[t];
      if (wid === null) { wordsMask[t] = 0; prev = -1; continue; }
      if (wid !== prev) { seen++; prev = wid; }
      else { wordsMask[t] = 0; continue; }
      if (seen <= skipWords) wordsMask[t] = 0;
      else wordsMask[t] = seen - skipWords;
    }
    const W = words.length;
    const spanIdx = prepareSpanIdx(W, this.maxWidth);
    const spanMask = new Int32Array(W * this.maxWidth);
    for (let s = 0; s < W * this.maxWidth; s++) spanMask[s] = spanIdx[2 * s + 1] <= (W - 1) ? 1 : 0;
    // forward
    const out = this.model.forward({ input_ids: ids, words_mask: wordsMask, span_idx: spanIdx, span_mask: spanMask, text_length: W });
    const { logits, K, C } = out;
    // decode
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) probs[i] = 1 / (1 + Math.exp(-logits[i]));
    const spans = [];
    for (let w = 0; w < W; w++) {
      for (let k = 0; k < K; k++) {
        if (w + k + 1 > W) continue;
        for (let c = 0; c < C; c++) {
          const p = probs[w * K * C + k * C + c];
          if (p > threshold) spans.push({ start: w, end: w + k, label: entityTypes[c], score: p });
        }
      }
    }
    // greedy non-overlap (flat NER): sort by score desc, keep non-overlapping
    spans.sort((a, b) => b.score - a.score);
    const sel = [];
    for (const sp of spans) {
      let overlap = false;
      for (const ex of sel) {
        if (!(sp.start > ex.end || ex.start > sp.end)) { overlap = true; break; }
      }
      if (!overlap) sel.push(sp);
    }
    sel.sort((a, b) => a.start - b.start);
    // map to char positions
    return sel.map((sp) => {
      const st = starts[sp.start], en = ends[sp.end];
      return { start: st, end: en, text: text.slice(st, en), label: sp.label, score: sp.score };
    });
  }
}

module.exports = { GLiNER, splitWords, prepareSpanIdx };

// --- CLI ---
function printHelp() {
  console.log(`Usage: node run.js --model DIR [options]

Run the GLiNER model (nvidia/gliner-PII) in pure Node.js stdlib. No npm deps.

Options:
  --model DIR       model directory (from download_model.js)
  --text STR        text to extract entities from
  --labels LBS      comma-separated entity labels
  --threshold F     confidence threshold (default 0.5)
  --fast            use the WASM SIMD tiled-GEMM kernel (faster, ~7x; pure stdlib WebAssembly)
  --input FILE      JSONL with {"text":...,"labels":[...]} per line (batch)
  --output FILE     write JSONL results (one line per input)
  -h, --help        show this help

Examples:
  node run.js --model ../model --text "My name is Jason, born 1987-05-22." --labels person,date_of_birth
  node run.js --model ../model --input rows.jsonl --output out.jsonl --labels first_name,date_of_birth
`);
}
async function main() {
  const a = process.argv.slice(2);
  let modelDir = null, text = null, labels = null, threshold = 0.5, input = null, output = null, fast = false;
  for (let i = 0; i < a.length; i++) {
    const k = a[i];
    if (k === '-h' || k === '--help') { printHelp(); return; }
    else if (k === '--model') modelDir = a[++i];
    else if (k === '--text') text = a[++i];
    else if (k === '--labels') labels = a[++i];
    else if (k === '--threshold') threshold = parseFloat(a[++i]);
    else if (k === '--input') input = a[++i];
    else if (k === '--output') output = a[++i];
    else if (k === '--fast') fast = true;
    else if (k.includes('=')) { const [kk, vv] = k.split('='); ({ '--model': modelDir, '--text': text, '--labels': labels, '--threshold': threshold, '--input': input, '--output': output }[kk] = vv); }
  }
  if (!modelDir) { printHelp(); process.exit(2); }
  console.error(`Loading model from ${modelDir} ${fast ? '(fast/WASM)' : ''} ...`);
  const g = new GLiNER(modelDir).loadWeights({ fast });
  if (input) {
    const lines = fs.readFileSync(input, 'utf8').trim().split('\n').map(JSON.parse);
    const outs = lines.map((r) => {
      const lbs = r.labels ? r.labels : (labels ? labels.split(',') : []);
      return { text: r.text, entities: g.predict(r.text, lbs, threshold) };
    });
    const data = outs.map((o) => JSON.stringify(o)).join('\n') + '\n';
    if (output) fs.writeFileSync(output, data); else console.log(data);
    console.error(`Done: ${outs.length} texts processed.`);
  } else {
    if (!text || !labels) { console.error('Need --text and --labels'); process.exit(2); }
    const ents = g.predict(text, labels.split(','), threshold);
    console.log(JSON.stringify(ents, null, 2));
  }
}
if (require.main === module) main();
