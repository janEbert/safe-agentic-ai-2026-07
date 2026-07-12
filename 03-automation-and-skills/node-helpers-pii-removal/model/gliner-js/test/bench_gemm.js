'use strict';
/* Build, validate, and benchmark tiled GEMM variants vs the JS reference. */
const fs = require('node:fs');
const cp = require('node:child_process');
const { gen } = require('../wasm/gen_gemm.js');

const WAT2WASM = (() => {
  for (const p of ['./.wat2wasm', '/tmp/wabt-1.0.36/bin/wat2wasm']) {
    try { if (fs.existsSync(p)) return p; } catch (e) {}
  }
  try { const w = cp.execSync('command -v wat2wasm').toString().trim(); if (w) return w; } catch (e) {}
  return null;
})();
const OUT = __dirname + '/wasm';
fs.mkdirSync(OUT, { recursive: true });

function build(R, O) {
  if (!WAT2WASM) throw new Error('wat2wasm not found; install wabt or place it at ./.wat2wasm');
  const wat = `${OUT}/gemm_${R}x${O}.wat`;
  const wasm = `${OUT}/gemm_${R}x${O}.wasm`;
  fs.writeFileSync(wat, gen(R, O));
  cp.execFileSync(WAT2WASM, [wat, '-o', wasm]);
  return wasm;
}

function loadInstance(wasmPath, mem) {
  const bytes = fs.readFileSync(wasmPath);
  return new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { mem } });
}

// JS reference
function mmJS(x, n, Din, W, Dout, b) {
  const out = new Float32Array(n * Dout);
  for (let r = 0; r < n; r++) { const xo = r * Din, yo = r * Dout; for (let o = 0; o < Dout; o++) { let acc = b ? b[o] : 0; const wo = o * Din; for (let i = 0; i < Din; i++) acc += x[xo + i] * W[wo + i]; out[yo + o] = acc; } }
  return out;
}

function test( R, O, n, Din, Dout, withBias) {
  const mem = new WebAssembly.Memory({ initial: 2048, maximum: 65536 });
  const F = new Float32Array(mem.buffer);
  const inst = loadInstance(build(R, O), mem);
  const woff = 0, boff = Dout * Din, xoff = boff + Dout, ooff = xoff + Math.ceil(n / R) * R * Din;
  const W = new Float32Array(Dout * Din), b = new Float32Array(Dout), x = new Float32Array(Math.ceil(n / R) * R * Din);
  for (let i = 0; i < W.length; i++) W[i] = Math.sin(i * 1e-4);
  for (let i = 0; i < b.length; i++) b[i] = Math.cos(i * 1e-2);
  for (let i = 0; i < n * Din; i++) x[i] = Math.sin(i * 3e-4);
  F.set(W, woff); F.set(b, boff); F.set(x, xoff);
  const npad = Math.ceil(n / R) * R;
  inst.exports.mm(npad, Din, Dout, xoff, woff, withBias ? boff : -1, ooff);
  const got = F.subarray(ooff, ooff + n * Dout);
  const ref = mmJS(x, n, Din, W, Dout, withBias ? b : null);
  let maxd = 0;
  for (let i = 0; i < ref.length; i++) { const d = Math.abs(got[i] - ref[i]); if (d > maxd) maxd = d; }
  return maxd;
}

function bench(R, O, n, Din, Dout, N) {
  const mem = new WebAssembly.Memory({ initial: 4096, maximum: 65536 });
  const F = new Float32Array(mem.buffer);
  const inst = loadInstance(build(R, O), mem);
  const woff = 0, boff = Dout * Din, xoff = boff + Dout, ooff = xoff + Math.ceil(n / R) * R * Din;
  for (let i = 0; i < Dout * Din; i++) F[woff + i] = Math.sin(i * 1e-4);
  for (let i = 0; i < Dout; i++) F[boff + i] = Math.cos(i * 1e-2);
  for (let i = 0; i < Math.ceil(n / R) * R * Din; i++) F[xoff + i] = Math.sin(i * 3e-4);
  const npad = Math.ceil(n / R) * R;
  const run = () => inst.exports.mm(npad, Din, Dout, xoff, woff, boff, ooff);
  run();
  let t = Date.now(); for (let k = 0; k < N; k++) run(); return (Date.now() - t) / N;
}

if (!WAT2WASM) {
  console.log('(wat2wasm not found — install wabt or place at ./.wat2wasm to run this benchmark)');
} else {
  console.log('=== correctness (vs JS ref), tol ~1e-3 ===');
  for (const [R, O] of [[4, 4], [4, 8], [8, 4], [8, 8]]) {
    const d1 = test(R, O, 59, 1024, 1024, true);
    const d2 = test(R, O, 16, 4096, 1024, true);
    const d3 = test(R, O, 12, 1024, 4096, false);
    console.log(`  ${R}x${O}: 59x1024x1024 maxd=${d1.toExponential(2)} | 16x4096x1024 maxd=${d2.toExponential(2)} | nobias maxd=${d3.toExponential(2)} ${d1 < 1e-2 && d2 < 1e-2 ? 'OK' : 'BAD'}`);
  }

  console.log('=== benchmark: Q-proj-like (n=59,Din=Dout=1024), 20 runs ===');
  for (const [R, O] of [[4, 4], [4, 8], [8, 4], [8, 8]]) {
    const ms = bench(R, O, 59, 1024, 1024, 20);
    console.log(`  ${R}x${O}: ${ms.toFixed(2)} ms`);
  }
  console.log('=== benchmark: intermediate-like (n=16,Din=4096,Dout=1024), 20 runs ===');
  for (const [R, O] of [[4, 4], [4, 8], [8, 4], [8, 8]]) {
    const ms = bench(R, O, 16, 4096, 1024, 20);
    console.log(`  ${R}x${O}: ${ms.toFixed(2)} ms`);
  }

}
