'use strict';
/*
 * WASM SIMD matmul kernel, loaded from the sibling mm.wasm. Pure stdlib
 * (WebAssembly is built into Node; no npm). Used by the --fast path.
 *
 * The kernel operates on a shared LinearMemory (`mem`) holding the weights
 * (loaded by loadStateDictShared) plus a scratch region for inputs/outputs.
 * `linear()` is a drop-in for tensor.js's JS `linear`: same numeric result
 * (within f32 reassociation tolerance), ~7x faster.
 */

const fs = require('node:fs');
const path = require('node:path');

let _inst = null, _mem = null, _F = null, _scratchX = 0, _scratchO = 0, _scratchEnd = 0, _simdOK = null;
const TILE_R = 4; // tiled GEMM processes R rows per tile; n is padded up to a multiple of R.
const TILE_O = 4; // output columns per tile (Dout must be a multiple of O; all are).

function simdAvailable() {
  if (_simdOK !== null) return _simdOK;
  // Minimal SIMD module: func returning f32x4.mul of two zero v128s.
  const bytes = new Uint8Array([
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,7,8,1,4,116,101,115,116,0,0,
    10,43,1,41,0,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,230,1,11
  ]);
  try { _simdOK = WebAssembly.validate(bytes); } catch (e) { _simdOK = false; }
  return _simdOK;
}

function init(mem, scratchXOff, scratchOOff) {
  const wasmPath = path.join(__dirname, '..', 'wasm', 'gemm_4x4.wasm');
  const bytes = fs.readFileSync(wasmPath);
  _inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { mem } });
  _mem = mem; _F = new Float32Array(mem.buffer);
  _scratchX = scratchXOff; _scratchO = scratchOOff; _scratchEnd = scratchOOff;
}

// Current heap view (refreshed after any Memory.grow). Weight lookups must go
// through this — never a cached subarray — since grow detaches the old buffer
// and resets cached views' byteOffset to 0.
function heapView() { return _F; }

// ensure scratch can hold `need` floats for input and output
function ensureScratch(needX, needO) {
  const need = Math.max(needX, needO);
  if (_scratchEnd + need <= _F.length) return;
  const pagesNeeded = Math.ceil(((_scratchEnd + need) * 4 - _F.length * 4) / 65536);
  _mem.grow(pagesNeeded);
  _F = new Float32Array(_mem.buffer); // grow detaches the old buffer; refresh view
}

// Drop-in for tensor.js linear(). x may be off-heap (copied in) or a heap view
// (passed by offset). W and b must be heap views (loaded by loadStateDictShared).
// The tiled kernel requires nRows % TILE_R == 0 and Dout % TILE_O == 0; nRows is
// padded up to a multiple of TILE_R (zero-fill pad rows; only valid rows copied out).
function linear(x, nRows, Din, W, Dout, b, out) {
  out = out || new Float32Array(nRows * Dout);
  const mm = _inst.exports.mm;
  const woff = W.byteOffset / 4;
  const boff = b ? (b.byteOffset / 4) : -1;
  const npad = Math.ceil(nRows / TILE_R) * TILE_R;
  const sameHeap = x.buffer === _F.buffer;
  ensureScratch(npad * Din, npad * Dout);
  if (sameHeap && npad === nRows) {
    // input already on the heap and no padding needed -> use its offset directly
    const xoff = x.byteOffset / 4;
    mm(npad, Din, Dout, xoff, woff, boff, _scratchO);
  } else {
    // copy input into scratch (padded), zeroing the pad rows
    const xoff = _scratchX;
    _F.set(x, _scratchX);
    for (let i = nRows * Din; i < npad * Din; i++) _F[_scratchX + i] = 0;
    mm(npad, Din, Dout, xoff, woff, boff, _scratchO);
  }
  out.set(_F.subarray(_scratchO, _scratchO + nRows * Dout));
  return out;
}

module.exports = { init, linear, heapView, simdAvailable };
