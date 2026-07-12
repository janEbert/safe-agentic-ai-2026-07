# gliner-js

Run NVIDIA **gliner-PII** — a DeBERTa-v3-large encoder + GLiNER span head for
PII/PHI named-entity recognition — using **only the Node.js standard library**.

No `npm install`, no `node_modules`, no Python, no native addons, no ONNX.
Everything (PyTorch checkpoint parsing, the SentencePiece/Unigram tokenizer, the
24-layer disentangled-attention transformer, the bidirectional LSTM, the span
scorer, and decoding) is reimplemented in pure JS and validated against the
Python `gliner` reference to **exact float agreement**.

## Requirements

- Node.js >= 18 (built-in `fetch`, `Float32Array`, etc.).
- Run with `--max-old-space-size=4096` (the model is 1.78 GB of float32 weights).

## Get the model

Use the sibling downloader (one level up) — pure stdlib, resumable:

```bash
cd .. && node download_model.js --repo nvidia/gliner-PII --out-dir ./model
```

This fetches `pytorch_model.bin` (1.78 GB), `tokenizer.json`, `spm.model`,
`gliner_config.json`, etc. into `./model`.

## Run

Single text:

```bash
node --max-old-space-size=4096 src/run.js --model ../model \
  --text "My name is Jason and I was born on 1987-05-22. Call me at 555-1234." \
  --labels person,date_of_birth,phone_number --threshold 0.5
```

Add `--fast` to use the embedded WebAssembly f32x4 SIMD **tiled GEMM** kernel
(pure stdlib `WebAssembly`; faster, same results). It falls back to the JS
kernel automatically if the runtime lacks SIMD:

```bash
node --max-old-space-size=4096 src/run.js --model ../model --fast \
  --text "My name is Jason, born 1987-05-22." --labels person,date_of_birth
```

Batch (JSONL in/out — each line `{"text":..., "labels":[...]}`):

```bash
node --max-old-space-size=4096 src/run.js --model ../model \
  --input rows.jsonl --output out.jsonl --labels first_name,date_of_birth,street_address
```

## What it implements (and how it was validated)

Every component was checked against a captured Python `gliner` reference run on
the exact same input:

| Component | Validation | Result |
|-----------|------------|--------|
| `torch_load.js` — PyTorch zip+pickle parser | all 416 params: names, shapes, first values | **416/416 exact** |
| `tokenizer.js` — HF Unigram+Metaspace+charsmap | input_ids + word_ids on the prompt+text | **exact** (59/59) |
| `model.js` — DeBERTa-v3 encoder output | vs reference `token_rep` | maxAbs ~1.7e-6 |
| word pooling + LSTM | vs reference `words_embedding` | maxAbs ~1.3e-6 |
| prompt embeddings | vs reference `prompts_embedding` | maxAbs ~1.6e-6 |
| SpanMarkerV0 + prompt MLP + einsum | vs reference `logits` (1,28,12,6) | maxAbs ~2.3e-5 |
| full decode (sigmoid/threshold/greedy) | entities vs reference | **exact match** |
| Nemotron-PII (20 rows) | JS vs Python `gliner.predict_entities` | **20/20 exact** |

Component tests (point them at the model dir + `ref/`):

```bash
node test/test_torch_load.js ../model/pytorch_model.bin ../ref/ref_state.json
node test/test_tokenizer.js   ../model/tokenizer.json
node --max-old-space-size=4096 test/test_model.js ../model ../ref
node --max-old-space-size=4096 test/test_e2e.js   ../model ../ref/meta.json
```

## Architecture (from the GLiNER source + `gliner_config.json`)

```
text + labels
  -> whitespace word split (\w+(?:[-_]\w+)*|\S)  with char offsets
  -> prompt = [<<ENT>> label]*  +  [<<SEP>>]
  -> tokenizer (is_split_into_words): normalizer(Strip+Precompiled) -> Metaspace -> Unigram Viterbi
  -> words_mask (skip prompt), span_idx (W*max_width), span_mask
  -> DeBERTa-v3-large: LayerNorm(word_embeddings); 24x disentangled-attention
       layers (content-content + content-position + position-content, log-bucketed
       relative positions, share_att_key) + FFN(gelu); projection 1024->512
  -> extract prompts (at <<ENT>>/128002) + words (first-subtoken pooling)
  -> bidirectional LSTM (512->256x2) over words
  -> SpanMarkerV0(start||end MLP) . prompt_rep MLP -> dot-product span scores
  -> sigmoid, threshold, valid-span, greedy non-overlap -> char spans
```

## Performance

~21 s per text on CPU with the pure-JS kernel; **~2.8 s per text with `--fast`**
(≈7.5× vs the JS kernel). `--fast` replaces the matmul with a hand-written,
**cache-tiled f32x4 SIMD GEMM**: it blocks over (4 rows × 4 output cols) tiles,
hoisting each weight block out of the row loop so weights are read ~nRows/4×
instead of ~nRows× (the matmul was memory-bound — measured ~51× weight re-reads
per row). Numerics are bit-identical to the reference within f32 reassociation
tolerance (validated on all intermediates + 20/20 Nemotron-PII rows).

## Files

- `src/torch_load.js` — PyTorch checkpoint (zip + pickle) parser, pure stdlib.
- `src/tokenizer.js` — HF `tokenizers` pipeline (Unigram/Metaspace/Precompiled charsmap).
- `src/tensor.js` — matmul, layernorm, gelu(erf), softmax, LSTM ops.
- `src/wasm-kernel.js` — `--fast` path: loads `wasm/gemm_4x4.wasm`, drops the
  tiled SIMD GEMM in for `linear()` (with n-padding to the 4-row tile, SIMD
  detection, and a JS fallback).
- `wasm/gemm_4x4.wat` / `gemm_4x4.wasm` — the hand-written **tiled f32x4 GEMM**
  (4×4 tile; the W block is reused across rows). `wasm/gen_gemm.js` generates the
  WAT for any (R,O) tile; `wasm/mm.wat`/`mm.wasm` is the simpler non-tiled
  predecessor (kept for reference). WebAssembly is built into Node, so the
  `.wasm` needs no toolchain at runtime; `wat2wasm` (wabt) is only needed to
  regenerate `.wasm` from `.wat`.
- `src/model.js` — DeBERTa-v3-large + GLiNER span head forward pass.
- `src/run.js` — full inference pipeline + CLI.
- `test/` — validation harnesses vs the Python reference.
