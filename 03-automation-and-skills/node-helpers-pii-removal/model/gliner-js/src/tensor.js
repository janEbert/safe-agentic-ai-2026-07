'use strict';
/* Numerical ops for the DeBERTa-v3 + GLiNER forward pass (pure stdlib, Float32Array). */

// Abramowitz & Stegun 7.1.26 erf (max abs error ~1.5e-7) — sufficient vs float32.
function _erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const y = 1 - poly * Math.exp(-ax * ax);
  return sign * y;
}
function gelu(x) { return 0.5 * x * (1 + _erf(x / Math.SQRT2)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// LayerNorm over the last dim (length D). gamma,beta length D. eps default 1e-7.
function layernorm(x, off, D, gamma, beta, eps = 1e-7, out, outOff) {
  let mean = 0;
  for (let i = 0; i < D; i++) mean += x[off + i];
  mean /= D;
  let varr = 0;
  for (let i = 0; i < D; i++) { const d = x[off + i] - mean; varr += d * d; }
  varr /= D;
  const inv = 1 / Math.sqrt(varr + eps);
  for (let i = 0; i < D; i++) {
    out[outOff + i] = (x[off + i] - mean) * inv * gamma[i] + beta[i];
  }
}

// Linear: y[row, o] = b[o] + sum_i x[row, i] * W[o, i].   W is (Dout, Din).
// x: Float32Array (nRows, Din). Returns Float32Array (nRows, Dout).
let _kernel = null; // optional WASM kernel (set via setKernel for --fast)
function setKernel(k) { _kernel = k; }
function linear(x, nRows, Din, W, Dout, b, out) {
  if (_kernel) return _kernel.linear(x, nRows, Din, W, Dout, b, out);
  out = out || new Float32Array(nRows * Dout);
  for (let r = 0; r < nRows; r++) {
    const xo = r * Din, yo = r * Dout;
    const xr = x.subarray(xo, xo + Din);
    for (let o = 0; o < Dout; o++) {
      let acc = b ? b[o] : 0;
      const wr = W.subarray(o * Din, o * Din + Din);
      for (let i = 0; i < Din; i++) acc += xr[i] * wr[i];
      out[yo + o] = acc;
    }
  }
  return out;
}

// Row-wise matmul C (n, m) = A (n, k) @ B (m, k)^T  -> C[a,j] = sum_i A[a,i]*B[j,i]
function bmm_AT(A, n, k, B, m, out) {
  out = out || new Float32Array(n * m);
  for (let a = 0; a < n; a++) {
    const ao = a * k, co = a * m;
    for (let j = 0; j < m; j++) {
      const bo = j * k;
      let s = 0;
      for (let i = 0; i < k; i++) s += A[ao + i] * B[bo + i];
      out[co + j] = s;
    }
  }
  return out;
}

// Masked softmax over the last dim (m). mask (n, m) optional Float32Array (1=valid,0=pad).
function softmaxRows(scores, n, m, mask, out) {
  out = out || new Float32Array(n * m);
  for (let a = 0; a < n; a++) {
    const ao = a * m, mo = a * m;
    let maxv = -Infinity;
    for (let j = 0; j < m; j++) { if (mask && !mask[mo + j]) continue; const v = scores[ao + j]; if (v > maxv) maxv = v; }
    if (maxv === -Infinity) { for (let j = 0; j < m; j++) out[ao + j] = 0; continue; }
    let sum = 0;
    for (let j = 0; j < m; j++) {
      if (mask && !mask[mo + j]) { out[ao + j] = 0; continue; }
      const e = Math.exp(scores[ao + j] - maxv);
      out[ao + j] = e; sum += e;
    }
    for (let j = 0; j < m; j++) out[ao + j] = sum > 0 ? out[ao + j] / sum : 0;
  }
  return out;
}

// Sigmoid a whole array.
function sigmoidArr(x, n, out) {
  out = out || new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = sigmoid(x[i]);
  return out;
}

module.exports = { _erf, gelu, sigmoid, layernorm, linear, bmm_AT, softmaxRows, sigmoidArr, setKernel };
