'use strict';
/* DeBERTa-v3-large + GLiNER span head, pure stdlib forward pass (batch=1). */
const T = require('./tensor');

const SQRT2 = Math.SQRT2;

// relative-position log bucketing (DeBERTa). returns Int32Array (L*L).
function buildRelPos(L, bucket = 256, maxPos = 512) {
  const mid = bucket >> 1; // 128
  const logDenom = Math.log((maxPos - 1) / mid);
  const out = new Int32Array(L * L);
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      let r = i - j;
      const ar = Math.abs(r);
      let bp;
      if (ar <= mid) {
        bp = r;
      } else {
        const sign = r < 0 ? -1 : 1;
        const logPos = Math.ceil(Math.log(ar / mid) / logDenom * (mid - 1)) + mid;
        bp = logPos * sign;
      }
      out[i * L + j] = bp;
    }
  }
  return out;
}

class GLiNERModel {
  constructor(sd, cfg) {
    this.sd = sd;      // {name: {data: Float32Array, shape: []}}
    this.cfg = cfg;
    this.H = cfg.num_attention_heads;       // 16
    this.hd = cfg.hidden_size / this.H;      // 64
    this.layers = cfg.num_hidden_layers;     // 24
    this.D = cfg.hidden_size;                // 1024
    this.ID = cfg.intermediate_size;         // 4096
    this.eps = cfg.layer_norm_eps;           // 1e-7
    this.maxWidth = cfg.max_width;           // 12
    this.hidden = cfg.hidden_size_gliner;   // 512
    this.classTokenIndex = cfg.class_token_index; // 128002
    this._relPosCache = {};
    this.W = (name) => this._fastHeap
      ? this._fastHeap().subarray(sd[name].off, sd[name].off + sd[name].numel)
      : sd[name].data;
    this.S = (name) => sd[name].shape;
  }

  relPos(L) {
    if (!this._relPosCache[L]) this._relPosCache[L] = buildRelPos(L, 256, 512);
    return this._relPosCache[L];
  }

  // Gather only the rel-embedding rows actually indexed by the bucketed relative positions.
  _posNeeded(L, brel, relEmb) {
    if (this._posNeededCache && this._posNeededCache.L === L) return this._posNeededCache;
    const used = new Uint8Array(512);
    for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
      const idx = brel[i * L + j] + 256; used[idx < 0 ? 0 : (idx > 511 ? 511 : idx)] = 1;
    }
    const neededIdx = [];
    const idxMap = new Int32Array(512).fill(-1);
    for (let r = 0; r < 512; r++) if (used[r]) { idxMap[r] = neededIdx.length; neededIdx.push(r); }
    const n = neededIdx.length;
    const D = this.D;
    const relEmbNeeded = new Float32Array(n * D);
    for (let k = 0; k < n; k++) { const src = neededIdx[k] * D; for (let d = 0; d < D; d++) relEmbNeeded[k * D + d] = relEmb[src + d]; }
    this._posNeededCache = { L, idxMap, relEmbNeeded, n };
    return this._posNeededCache;
  }

  // DeBERTa encoder -> token embeddings (L, 1024)
  encode(inputIds) {
    const sd = this.sd, L = inputIds.length, D = this.D, H = this.H, hd = this.hd;
    // embeddings
    const wordEmb = this.W('token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight');
    const embLNw = this.W('token_rep_layer.bert_layer.model.embeddings.LayerNorm.weight');
    const embLNb = this.W('token_rep_layer.bert_layer.model.embeddings.LayerNorm.bias');
    const emb = new Float32Array(L * D);
    for (let t = 0; t < L; t++) {
      const id = inputIds[t];
      const rowOff = id * D;
      T.layernorm(wordEmb, rowOff, D, embLNw, embLNb, this.eps, emb, t * D);
    }
    // rel embeddings (512, 1024) layer-normed per row
    const relEmb0 = this.W('token_rep_layer.bert_layer.model.encoder.rel_embeddings.weight');
    const relLNw = this.W('token_rep_layer.bert_layer.model.encoder.LayerNorm.weight');
    const relLNb = this.W('token_rep_layer.bert_layer.model.encoder.LayerNorm.bias');
    const nRel = relEmb0.length / D; // 512
    const relEmb = new Float32Array(nRel * D);
    for (let b = 0; b < nRel; b++) T.layernorm(relEmb0, b * D, D, relLNw, relLNb, this.eps, relEmb, b * D);
    // pos_key, pos_query reuse each layer's own query/key_proj (share_att_key). Computed per layer below.
    const brel = this.relPos(L);
    const scale = Math.sqrt(hd * 3);
    // Only the relative-position rows actually indexed are needed; gather them.
    const needed = this._posNeeded(L, brel, relEmb); // {idxMap:Int32[512], relEmbNeeded:(n,D)}

    let h = emb; // (L, D)
    for (let li = 0; li < this.layers; li++) {
      const pre = `token_rep_layer.bert_layer.model.encoder.layer.${li}.`;
      const qW = this.W(pre + 'attention.self.query_proj.weight'), qB = this.W(pre + 'attention.self.query_proj.bias');
      const kW = this.W(pre + 'attention.self.key_proj.weight'), kB = this.W(pre + 'attention.self.key_proj.bias');
      const vW = this.W(pre + 'attention.self.value_proj.weight'), vB = this.W(pre + 'attention.self.value_proj.bias');
      const qProj = T.linear(h, L, D, qW, D, qB); // (L, D)
      const kProj = T.linear(h, L, D, kW, D, kB);
      const vProj = T.linear(h, L, D, vW, D, vB);
      // pos projections: only needed relEmb rows through this layer's key/query_proj
      const posKeyC = T.linear(needed.relEmbNeeded, needed.n, D, kW, D, kB); // (n, D)
      const posQryC = T.linear(needed.relEmbNeeded, needed.n, D, qW, D, qB);
      // reshape to (H, L, hd): hh[t*hd+d] -> q[h*L*hd + t*hd + d] where the head dim is contiguous in (L, H*hd)
      // q[h,t,d] = qProj[t*H*hd + h*hd + d]
      const ctx = new Float32Array(L * D); // (L, H*hd): ctx[t*H*hd + h*hd + d]
      for (let hh = 0; hh < H; hh++) {
        const qi = hh * hd; // head's offset within a row of (L, H*hd)
        // scores (L, L)
        const scores = new Float32Array(L * L);
        for (let i = 0; i < L; i++) {
          for (let j = 0; j < L; j++) {
            let s = 0;
            for (let d = 0; d < hd; d++) s += qProj[i * D + qi + d] * kProj[j * D + qi + d];
            // rel_att (only needed pos rows are populated)
            const idx = brel[i * L + j] + 256;
            const cl = idx < 0 ? 0 : (idx > 511 ? 511 : idx);
            const pk = needed.idxMap[cl] * D + qi;
            let ra = 0;
            for (let d = 0; d < hd; d++) {
              ra += qProj[i * D + qi + d] * posKeyC[pk + d];
              ra += kProj[j * D + qi + d] * posQryC[pk + d];
            }
            s = (s + ra) / scale;
            scores[i * L + j] = s;
          }
        }
        // softmax over j (all real for B=1 no padding)
        const probs = T.softmaxRows(scores, L, L, null);
        // ctx[h,i,d] = sum_j probs[i,j]*v[h,j,d]; write into ctx[i*D + hh*hd + d]
        for (let i = 0; i < L; i++) {
          for (let d = 0; d < hd; d++) {
            let acc = 0;
            for (let j = 0; j < L; j++) acc += probs[i * L + j] * vProj[j * D + qi + d];
            ctx[i * D + qi + d] = acc;
          }
        }
      }
      // attention output: dense(ctx) + h, layernorm
      const aoDense = T.linear(ctx, L, D, this.W(pre + 'attention.output.dense.weight'), D, this.W(pre + 'attention.output.dense.bias'));
      const attnOut = new Float32Array(L * D);
      const aoLNw = this.W(pre + 'attention.output.LayerNorm.weight'), aoLNb = this.W(pre + 'attention.output.LayerNorm.bias');
      for (let t = 0; t < L; t++) {
        for (let i = 0; i < D; i++) aoDense[t * D + i] += h[t * D + i];
        T.layernorm(aoDense, t * D, D, aoLNw, aoLNb, this.eps, attnOut, t * D);
      }
      // intermediate
      const inter = T.linear(attnOut, L, D, this.W(pre + 'intermediate.dense.weight'), this.ID, this.W(pre + 'intermediate.dense.bias'));
      for (let i = 0; i < inter.length; i++) inter[i] = T.gelu(inter[i]);
      // output
      const outDense = T.linear(inter, L, this.ID, this.W(pre + 'output.dense.weight'), D, this.W(pre + 'output.dense.bias'));
      const layerOut = new Float32Array(L * D);
      const oLNw = this.W(pre + 'output.LayerNorm.weight'), oLNb = this.W(pre + 'output.LayerNorm.bias');
      for (let t = 0; t < L; t++) {
        for (let i = 0; i < D; i++) outDense[t * D + i] += attnOut[t * D + i];
        T.layernorm(outDense, t * D, D, oLNw, oLNb, this.eps, layerOut, t * D);
      }
      h = layerOut;
    }
    // projection 1024 -> 512
    const tokenRep = T.linear(h, L, D, this.W('token_rep_layer.projection.weight'), this.hidden, this.W('token_rep_layer.projection.bias'));
    return tokenRep; // (L, 512)
  }

  // bidirectional LSTM over (T, 512) -> (T, 512)
  lstm(x, Tlen) {
    const D = this.hidden;          // 512
    const hid = D >> 1;             // 256
    const wIh = this.W('rnn.lstm.weight_ih_l0'), wHh = this.W('rnn.lstm.weight_hh_l0');
    const bIh = this.W('rnn.lstm.bias_ih_l0'), bHh = this.W('rnn.lstm.bias_hh_l0');
    const wIhR = this.W('rnn.lstm.weight_ih_l0_reverse'), wHhR = this.W('rnn.lstm.weight_hh_l0_reverse');
    const bIhR = this.W('rnn.lstm.bias_ih_l0_reverse'), bHhR = this.W('rnn.lstm.bias_hh_l0_reverse');
    const fwd = this._lstmDir(x, Tlen, D, hid, wIh, wHh, bIh, bHh, 1);
    const bwd = this._lstmDir(x, Tlen, D, hid, wIhR, wHhR, bIhR, bHhR, -1);
    const out = new Float32Array(Tlen * D);
    for (let t = 0; t < Tlen; t++) for (let i = 0; i < hid; i++) {
      out[t * D + i] = fwd[t * hid + i];
      out[t * D + hid + i] = bwd[t * hid + i];
    }
    return out;
  }
  _lstmDir(x, Tlen, D, hid, wIh, wHh, bIh, bHh, dir) {
    const out = new Float32Array(Tlen * hid);
    const c = new Float32Array(hid), hprev = new Float32Array(hid);
    const gates = new Float32Array(4 * hid);
    const start = dir === 1 ? 0 : Tlen - 1;
    for (let n = 0; n < Tlen; n++) {
      const t = dir === 1 ? n : (Tlen - 1 - n);
      const xo = t * D;
      // gates = x_t @ W_ih^T + b_ih + hprev @ W_hh^T + b_hh
      for (let g = 0; g < 4 * hid; g++) {
        let acc = bIh[g] + bHh[g];
        const wi = g * D;
        for (let i = 0; i < D; i++) acc += x[xo + i] * wIh[wi + i];
        const wh = g * hid;
        for (let i = 0; i < hid; i++) acc += hprev[i] * wHh[wh + i];
        gates[g] = acc;
      }
      for (let i = 0; i < hid; i++) {
        const ig = T.sigmoid(gates[i]);
        const fg = T.sigmoid(gates[hid + i]);
        const gg = Math.tanh(gates[2 * hid + i]);
        const og = T.sigmoid(gates[3 * hid + i]);
        c[i] = fg * c[i] + ig * gg;
        const hn = og * Math.tanh(c[i]);
        hprev[i] = hn;
        out[t * hid + i] = hn;
      }
    }
    return out;
  }

  // MLP: Sequential(Linear(in, 4*in), ReLU, Linear(4*in, out)) using keys prefix + '.0'/'+ .3'
  _mlp(x, nRows, inD, outD, prefix) {
    const mid = outD * 4; // create_projection_layer expands to out_dim*4
    const h = T.linear(x, nRows, inD, this.W(prefix + '.0.weight'), mid, this.W(prefix + '.0.bias'));
    for (let i = 0; i < h.length; i++) h[i] = Math.max(0, h[i]); // ReLU (dropout eval = identity)
    return T.linear(h, nRows, mid, this.W(prefix + '.3.weight'), outD, this.W(prefix + '.3.bias'));
  }

  // Full forward given the collated inputs (B=1). Returns { logits, intermediates }.
  forward({ input_ids, words_mask, span_idx, span_mask, text_length }) {
    const L = input_ids.length;
    const tokenRep = this.encode(input_ids); // (L, 512)
    const hid = this.hidden; // 512
    // extract words_embedding (first-subtoken pooling) and prompts_embedding
    const W = text_length;
    const wordsEmbPre = new Float32Array(W * hid);
    const promptsPre = [];
    for (let t = 0; t < L; t++) {
      const m = words_mask[t];
      if (m > 0) {
        const wi = m - 1;
        for (let d = 0; d < hid; d++) wordsEmbPre[wi * hid + d] = tokenRep[t * hid + d];
      }
      if (input_ids[t] === this.classTokenIndex) {
        const row = new Float32Array(hid);
        for (let d = 0; d < hid; d++) row[d] = tokenRep[t * hid + d];
        promptsPre.push(row);
      }
    }
    const C = promptsPre.length;
    const wordsEmb = this.lstm(wordsEmbPre, W); // (W, 512)
    // prompt_rep
    const promptsPreFlat = new Float32Array(C * hid);
    for (let c = 0; c < C; c++) for (let d = 0; d < hid; d++) promptsPreFlat[c * hid + d] = promptsPre[c][d];
    const promptRep = this._mlp(promptsPreFlat, C, hid, hid, 'prompt_rep_layer'); // (C, 512)
    // span_rep (SpanMarkerV0)
    const startRep = this._mlp(wordsEmb, W, hid, hid, 'span_rep_layer.span_rep_layer.project_start'); // (W,512)
    const endRep = this._mlp(wordsEmb, W, hid, hid, 'span_rep_layer.span_rep_layer.project_end');
    const K = this.maxWidth;
    const numSpans = span_idx.length / 2; // (W*K, 2)
    const cat = new Float32Array(numSpans * 2 * hid); // (numSpans, 1024) after relu
    for (let s = 0; s < numSpans; s++) {
      const sm = span_mask[s];
      const si = span_idx[2 * s] * sm;     // mask invalid spans to (0,0)
      const ei = span_idx[2 * s + 1] * sm;
      for (let d = 0; d < hid; d++) {
        let a = startRep[si * hid + d], b = endRep[ei * hid + d];
        cat[s * 2 * hid + d] = a > 0 ? a : 0;
        cat[s * 2 * hid + hid + d] = b > 0 ? b : 0;
      }
    }
    const spanRep = this._mlp(cat, numSpans, 2 * hid, hid, 'span_rep_layer.span_rep_layer.out_project'); // (numSpans, 512)
    const logits = new Float32Array(W * K * C);
    for (let s = 0; s < numSpans; s++) {
      const wi = Math.floor(s / K), k = s % K;
      for (let c = 0; c < C; c++) {
        let dot = 0;
        for (let d = 0; d < hid; d++) dot += spanRep[s * hid + d] * promptRep[c * hid + d];
        logits[wi * K * C + k * C + c] = dot;
      }
    }
    return { logits, tokenRep, wordsEmb, wordsEmbPre, promptsPreFlat, promptRep, spanRep, startRep, endRep, cat, W, K, C };
  }
}

module.exports = { GLiNERModel, buildRelPos };
