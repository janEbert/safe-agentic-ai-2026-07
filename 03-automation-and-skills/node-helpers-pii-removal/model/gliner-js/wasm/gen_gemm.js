'use strict';
/*
 * Generate a tiled f32x4 SIMD GEMM in WAT:  out[n,dout] = b[dout] + x[n,din] @ W[dout,din].
 *
 * Tile = R rows × O output columns. The W block (O cols × 4 k's) is loaded ONCE
 * per (o0, k) and reused across all R rows in the tile, so each weight is read
 * ~nRows/R times instead of nRows times (the current kernel is memory-bound:
 * ~51× W re-read). Din must be a multiple of 4; n and dout must be multiples of
 * R and O respectively (the JS caller pads n up to a multiple of R).
 *
 * Accumulators are v128 (4 partial sums, one per k in the 4-wide block),
 * initialized to 0; the bias is added at store time via a select.
 */
function gen(R, O) {
  const lines = [];
  const L = (s) => lines.push(s);
  L(';; Auto-generated tiled f32x4 GEMM (R=' + R + ', O=' + O + '). Do not edit; see gen_gemm.js.');
  L('(module');
  L('  (import "env" "mem" (memory 1))');
  L('  (func (export "mm")');
  L('    (param $n i32) (param $din i32) (param $dout i32)');
  L('    (param $xoff i32) (param $woff i32) (param $boff i32) (param $ooff i32)');
  L('    (local $r0 i32) (local $o0 i32) (local $k i32) (local $addr i32) (local $bv f32)');
  for (let r = 0; r < R; r++) L('    (local $a' + r + ' v128)');
  for (let c = 0; c < O; c++) L('    (local $w' + c + ' v128)');
  for (let r = 0; r < R; r++) for (let c = 0; c < O; c++) L('    (local $acc' + r + '_' + c + ' v128)');

  L('    (local.set $o0 (i32.const 0))');
  L('    (block $bo (loop $lo');
  L('      (br_if $bo (i32.ge_s (local.get $o0) (local.get $dout)))');
  L('      (local.set $r0 (i32.const 0))');
  L('      (block $br (loop $lr');
  L('        (br_if $br (i32.ge_s (local.get $r0) (local.get $n)))');

  // init acc[r][c] = 0
  for (let r = 0; r < R; r++) for (let c = 0; c < O; c++)
    L('        (local.set $acc' + r + '_' + c + ' (v128.const i32x4 0 0 0 0))');

  // k = 0; while k < din:
  L('        (local.set $k (i32.const 0))');
  L('        (block $bk (loop $lk');
  L('          (br_if $bk (i32.ge_s (local.get $k) (local.get $din)))');
  // load w[c] = W[o0+c, k:k+4]
  for (let c = 0; c < O; c++) {
    L('          (local.set $addr (i32.shl (i32.add (local.get $woff) (i32.add (i32.mul (i32.add (local.get $o0) (i32.const ' + c + ')) (local.get $din)) (local.get $k))) (i32.const 2)))');
    L('          (local.set $w' + c + ' (v128.load (local.get $addr)))');
  }
  // for each row r: load a[r]; for each c: acc[r][c] += a[r] * w[c]
  for (let r = 0; r < R; r++) {
    L('          (local.set $addr (i32.shl (i32.add (local.get $xoff) (i32.add (i32.mul (i32.add (local.get $r0) (i32.const ' + r + ')) (local.get $din)) (local.get $k))) (i32.const 2)))');
    L('          (local.set $a' + r + ' (v128.load (local.get $addr)))');
    for (let c = 0; c < O; c++) {
      L('          (local.set $acc' + r + '_' + c + ' (f32x4.add (local.get $acc' + r + '_' + c + ') (f32x4.mul (local.get $a' + r + ') (local.get $w' + c + '))))');
    }
  }
  L('          (local.set $k (i32.add (local.get $k) (i32.const 4)))');
  L('          (br $lk)))');

  // hsum + store: C[r0+r, o0+c] = hsum(acc[r][c]) + (boff>=0 ? b[o0+c] : 0)
  // NB: WASM `select` evaluates BOTH arms, so the f32.load must always be in-bounds.
  // Clamp the load address to 0 when boff<0 (reads harmless junk, discarded by select).
  const baddr = (c) => '(i32.shl (select (i32.add (local.get $boff) (i32.add (local.get $o0) (i32.const ' + c + '))) (i32.const 0) (i32.ge_s (local.get $boff) (i32.const 0))) (i32.const 2))';
  for (let r = 0; r < R; r++) for (let c = 0; c < O; c++) {
    L('        (local.set $bv (select (f32.load ' + baddr(c) + ') (f32.const 0) (i32.ge_s (local.get $boff) (i32.const 0))))');
    L('        (local.set $addr (i32.shl (i32.add (local.get $ooff) (i32.add (i32.mul (i32.add (local.get $r0) (i32.const ' + r + ')) (local.get $dout)) (i32.add (local.get $o0) (i32.const ' + c + ')))) (i32.const 2)))');
    L('        (f32.store (local.get $addr) (f32.add (f32.add (f32x4.extract_lane 0 (local.get $acc' + r + '_' + c + ')) (f32x4.extract_lane 1 (local.get $acc' + r + '_' + c + '))) (f32.add (f32.add (f32x4.extract_lane 2 (local.get $acc' + r + '_' + c + ')) (f32x4.extract_lane 3 (local.get $acc' + r + '_' + c + '))) (local.get $bv))))');
  }

  L('        (local.set $r0 (i32.add (local.get $r0) (i32.const ' + R + ')))');
  L('        (br $lr)))');
  L('      (local.set $o0 (i32.add (local.get $o0) (i32.const ' + O + ')))');
  L('      (br $lo)))');
  L(')');
  L(')');
  return lines.join('\n');
}

module.exports = { gen };
if (require.main === module) {
  const R = parseInt(process.argv[2] || '4', 10);
  const O = parseInt(process.argv[3] || '4', 10);
  process.stdout.write(gen(R, O) + '\n');
}
