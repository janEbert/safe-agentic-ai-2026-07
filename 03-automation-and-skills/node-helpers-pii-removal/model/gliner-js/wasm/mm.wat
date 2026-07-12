;; f32x4 SIMD matmul:  out[n,dout] = b[dout] + x[n,din] @ W[dout,din]
;; All offsets are in FLOATS (byte addr = off*4). Din must be a multiple of 4.
;; Memory is imported from JS (the shared heap holding weights + scratch).
(module
  (import "env" "mem" (memory 1))
  (func (export "mm")
    (param $n i32) (param $din i32) (param $dout i32)
    (param $xoff i32) (param $woff i32) (param $boff i32) (param $ooff i32)
    (local $r i32) (local $o i32) (local $i i32)
    (local $acc v128)
    (local $xrow i32) (local $orow i32) (local $wrow i32)
    (local $addr i32) (local $b0 f32)

    (local.set $r (i32.const 0))
    (block $brk_r
      (loop $lr
        (br_if $brk_r (i32.ge_s (local.get $r) (local.get $n)))
        (local.set $xrow (i32.add (i32.mul (local.get $r) (local.get $din)) (local.get $xoff)))
        (local.set $orow (i32.add (i32.mul (local.get $r) (local.get $dout)) (local.get $ooff)))
        (local.set $o (i32.const 0))
        (block $brk_o
          (loop $lo
            (br_if $brk_o (i32.ge_s (local.get $o) (local.get $dout)))
            ;; b0 = (boff>=0) ? f32.load(boff+o) : 0
            (if (result f32) (i32.ge_s (local.get $boff) (i32.const 0))
              (then (f32.load (i32.shl (i32.add (local.get $boff) (local.get $o)) (i32.const 2))))
              (else (f32.const 0)))
            (local.set $b0)
            (local.set $acc (f32x4.replace_lane 0 (v128.const i32x4 0 0 0 0) (local.get $b0)))
            ;; inner loop over i in steps of 4 (Din is a multiple of 4)
            (local.set $wrow (i32.add (i32.mul (local.get $o) (local.get $din)) (local.get $woff)))
            (local.set $i (i32.const 0))
            (block $brk_i
              (loop $li
                (br_if $brk_i (i32.ge_s (local.get $i) (local.get $din)))
                (local.set $addr (i32.shl (i32.add (local.get $xrow) (local.get $i)) (i32.const 2)))
                (local.set $acc
                  (f32x4.add (local.get $acc)
                    (f32x4.mul
                      (v128.load (local.get $addr))
                      (v128.load (i32.shl (i32.add (local.get $wrow) (local.get $i)) (i32.const 2))))))
                (local.set $i (i32.add (local.get $i) (i32.const 4)))
                (br $li)))
            ;; horizontal sum -> store at out[orow+o]
            (f32.store (i32.shl (i32.add (local.get $orow) (local.get $o)) (i32.const 2))
              (f32.add
                (f32.add (f32x4.extract_lane 0 (local.get $acc)) (f32x4.extract_lane 1 (local.get $acc)))
                (f32.add (f32x4.extract_lane 2 (local.get $acc)) (f32x4.extract_lane 3 (local.get $acc)))))
            (local.set $o (i32.add (local.get $o) (i32.const 1)))
            (br $lo)))
        (local.set $r (i32.add (local.get $r) (i32.const 1)))
        (br $lr))))
)
