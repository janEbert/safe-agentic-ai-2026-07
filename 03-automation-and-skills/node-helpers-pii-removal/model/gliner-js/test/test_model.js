'use strict';
/* Validate the JS forward pass against the captured reference tensors. */
const fs = require('node:fs');
const { loadStateDict, loadStateDictShared } = require('../src/torch_load');
const { GLiNERModel } = require('../src/model');
const T = require('../src/tensor');
const wasmKernel = require('../src/wasm-kernel');

function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); }
function readI64(p) { const b = fs.readFileSync(p); const a = new BigInt64Array(b.buffer, b.byteOffset, b.length / 8); return Array.from(a, (x) => Number(x)); }

function cmp(name, got, ref, tol) {
  tol = tol == null ? 1e-3 : tol;
  if (got.length !== ref.length) { console.log(`  ${name}: LEN MISMATCH ${got.length} vs ${ref.length}`); return false; }
  let maxAbs = 0, maxRel = 0, idx = -1;
  for (let i = 0; i < got.length; i++) {
    const d = Math.abs(got[i] - ref[i]);
    if (!Number.isFinite(d)) { if (idx < 0) { idx = i; } maxAbs = d; continue; }
    if (d > maxAbs) { maxAbs = d; idx = i; }
    const ar = Math.abs(ref[i]) > 1e-6 ? d / Math.abs(ref[i]) : 0;
    if (ar > maxRel) maxRel = ar;
  }
  const ok = Number.isFinite(maxAbs) && maxAbs < tol;
  console.log(`  ${name}: ${ok ? 'OK' : 'FAIL'} maxAbs=${maxAbs.toExponential(3)} maxRel=${maxRel.toExponential(3)} (n=${got.length})`);
  if (!ok && idx >= 0) console.log(`    first bad idx ${idx}: got=${got[idx]} ref=${ref[idx]}`);
  return ok;
}

function loadConfig(p) {
  const g = JSON.parse(fs.readFileSync(p, 'utf8'));
  const e = g.encoder_config;
  return Object.assign({}, e, {
    max_width: g.max_width, class_token_index: g.class_token_index, hidden_size_gliner: g.hidden_size,
  });
}

async function main() {
  const modelDir = process.argv[2];
  const refDir = process.argv[3];
  const fast = process.argv[4] === '--fast';
  const bin = modelDir + '/pytorch_model.bin';
  let sd;
  console.log('loading weights' + (fast ? ' (fast/WASM)' : '') + '...');
  if (fast) {
    const { loadArchive } = require('../src/torch_load');
    const { descriptors } = loadArchive(bin);
    let total = 0; for (const d of Object.values(descriptors)) total += d.size.reduce((a,b)=>a*b,1);
    const PAGE=65536;
    const maxL=512, maxSpans=384*12;
    const maxOut=Math.max(maxL*4096, maxSpans*2048, maxL*1024);
    const scratchEach=(maxOut+(1<<20))|0;
    const initPages=Math.ceil((total*4 + 2*scratchEach*4)/PAGE);
    const mem=new WebAssembly.Memory({initial:initPages, maximum:65536});
    const heap=new Float32Array(mem.buffer,0,Math.floor(mem.buffer.byteLength/4));
    ({sd} = loadStateDictShared(bin, heap));
    const sX=Math.ceil(total/4)*4, sO=sX+scratchEach;
    wasmKernel.init(mem, sX, sO); T.setKernel(wasmKernel);
  } else {
    sd = loadStateDict(bin);
  }
  const cfg = loadConfig(modelDir + '/gliner_config.json');
  const m = new GLiNERModel(sd, cfg);
  if (fast) m._fastHeap = () => wasmKernel.heapView();

  const input_ids = readI64(refDir + '/batch_input_ids.i64');      // length 59
  const words_mask = readI64(refDir + '/batch_words_mask.i64');
  const span_idx = readI64(refDir + '/batch_span_idx.i64');        // 336*2
  const span_mask = readI64(refDir + '/batch_span_mask.i64');
  const text_length = readI64(refDir + '/batch_text_lengths.i64')[0];

  console.log(`forward: L=${input_ids.length} W=${text_length} spans=${span_idx.length / 2}`);
  const t0 = Date.now();
  const r = m.forward({ input_ids, words_mask, span_idx, span_mask, text_length });
  console.log(`forward took ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const refTokenRep = readF32(refDir + '/mid_token_rep.f32');
  const refWordsEmb = readF32(refDir + '/mid_words_embedding.f32');
  const refPromptsPre = readF32(refDir + '/mid_prompts_embedding_pre.f32');
  const refPromptRep = readF32(refDir + '/mid_prompt_rep.f32');
  const refSpanRep = readF32(refDir + '/mid_span_rep.f32');
  const refLogits = readF32(refDir + '/out_logits.f32');

  console.log('comparing intermediates (tol=1e-3):');
  let allOk = true;
  allOk &= cmp('token_rep (encoder)', r.tokenRep, refTokenRep);
  allOk &= cmp('words_embedding (post-lstm)', r.wordsEmb, refWordsEmb, 5e-3);
  allOk &= cmp('prompts_pre', r.promptsPreFlat, refPromptsPre);
  allOk &= cmp('prompt_rep', r.promptRep, refPromptRep);
  allOk &= cmp('span_rep', r.spanRep, refSpanRep);
  allOk &= cmp('logits', r.logits, refLogits);
  console.log('OVERALL:', allOk ? 'PASS' : 'FAIL (see above)');
}
main();
