'use strict';
/*
 * Pure-stdlib loader for PyTorch `torch.save` archives (pytorch_model.bin).
 *
 * `pytorch_model.bin` is a ZIP archive containing:
 *   - pytorch_model/data.pkl   : a pickle (protocol 2) describing an OrderedDict
 *                                {param_name: _rebuild_tensor_v2(storage, ...)}.
 *   - pytorch_model/data/<key> : raw little-endian float32 bytes for each storage.
 *
 * This module parses the ZIP central directory (STORED entries only — torch does
 * not compress), interprets the pickle opcodes to recover each tensor's storage
 * key / offset / shape / stride, then reads the float32 bytes on demand.
 *
 * No npm dependencies. Requires Node >= 14 (for BigInt if used; we avoid it).
 */

const fs = require('node:fs');

// --------------------------------------------------------------------------
// Minimal ZIP reader for STORED entries (torch archives are never compressed).
// --------------------------------------------------------------------------
function readZip(path) {
  const fd = fs.openSync(path, 'r');
  const stat = fs.fstatSync(fd);
  const size = stat.size;

  // Find End Of Central Directory record (PK\x05\x06) scanning from the end.
  const scanLen = Math.min(size, 65557);
  const tail = Buffer.alloc(scanLen);
  fs.readSync(fd, tail, 0, scanLen, size - scanLen);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) {
    if (tail[i] === 0x50 && tail[i + 1] === 0x4b && tail[i + 2] === 0x05 && tail[i + 3] === 0x06) {
      eocd = i; break;
    }
  }
  if (eocd < 0) throw new Error('ZIP EOCD not found');
  const base = size - scanLen;
  const cdEntries = tail.readUInt16LE(eocd + 10);
  const cdOffset = tail.readUInt32LE(eocd + 16);

  // Read the central directory.
  const entries = {};
  let p = cdOffset;
  for (let e = 0; e < cdEntries; e++) {
    const sig = Buffer.alloc(46);
    fs.readSync(fd, sig, 0, 46, p);
    if (sig.readUInt32LE(0) !== 0x02014b50) throw new Error('bad central dir entry at ' + p);
    const compressMethod = sig.readUInt16LE(10);
    const compSize = sig.readUInt32LE(20);
    const uncompSize = sig.readUInt32LE(24);
    const nameLen = sig.readUInt16LE(28);
    const extraLen = sig.readUInt16LE(30);
    const commentLen = sig.readUInt16LE(32);
    const localOff = sig.readUInt32LE(42);
    const nameBuf = Buffer.alloc(nameLen);
    fs.readSync(fd, nameBuf, 0, nameLen, p + 46);
    const name = nameBuf.toString('utf8');
    p += 46 + nameLen + extraLen + commentLen;
    // Read local header to find exact data offset.
    const lh = Buffer.alloc(30);
    fs.readSync(fd, lh, 0, 30, localOff);
    if (lh.readUInt32LE(0) !== 0x04034b50) throw new Error('bad local header for ' + name);
    const lNameLen = lh.readUInt16LE(26);
    const lExtraLen = lh.readUInt16LE(28);
    const dataOffset = localOff + 30 + lNameLen + lExtraLen;
    entries[name] = { dataOffset, size: compressMethod === 0 ? uncompSize : compSize, method: compressMethod };
  }
  return { fd, entries, size };
}

// --------------------------------------------------------------------------
// Pickle (protocol 2) interpreter, specialized for torch tensor archives.
// --------------------------------------------------------------------------
const OP = {
  PROTO: 0x80, FRAME: 0x95, EMPTY_DICT: 0x7d, EMPTY_TUPLE: 0x29, MARK: 0x28,
  STOP: 0x2e, POP: 0x30, POP_MARK: 0x31, DUP: 0x32,
  NONE: 0x4e, NEWFALSE: 0x88, NEWTRUE: 0x89,
  BININT: 0x4a, BININT1: 0x4b, BININT2: 0x4d, LONG: 0x4c, LONG1: 0x8a, LONG4: 0x8b, INT: 0x49, FLOAT: 0x46,
  BINUNICODE: 0x58, SHORT_BINUNICODE: 0x8c, BINUNICODE8: 0x8d,
  SHORT_BINSTRING: 0x55, BINSTRING: 0x54,
  GLOBAL: 0x63, STACK_GLOBAL: 0x93,
  REDUCE: 0x52, BUILD: 0x62, MARK_DICT: 0x7d,
  TUPLE: 0x74, TUPLE1: 0x85, TUPLE2: 0x86, TUPLE3: 0x87,
  SETITEM: 0x73, SETITEMS: 0x75, APPEND: 0x61, APPENDS: 0x65,
  BINPUT: 0x71, BINGET: 0x68, LONG_BINPUT: 0x72, LONG_BINGET: 0x6a, MEMOIZE: 0x94,
  PERSID: 0x50, BINPERSID: 0x51,
  NEWOBJ: 0x81, NEWOBJ_EX: 0x92, OBJ: 0x6f, INST: 0x69,
};

const MARK = Symbol('mark');

function parsePickle(buf) {
  const stack = [];
  const memo = [];
  let i = 0;
  const n = buf.length;

  function persistentLoad(pid) {
    // pid = ('storage', StorageTypeGlobal, key, location, numel)
    const key = pid[2];
    return { key, dtype: 'float32', numel: pid[4] };
  }
  function reduceGlobal(g, args) {
    const fn = g.module + '.' + g.name;
    if (fn === 'torch._utils._rebuild_tensor_v2') {
      // args = [storage, storage_offset, size, stride, requires_grad, backward_hooks]
      const [storage, storage_offset, size, stride] = args;
      return { storageKey: storage.key, dtype: storage.dtype, storage_offset, size: size.slice(), stride: stride.slice() };
    }
    if (fn === 'torch._utils._rebuild_tensor' || fn === 'torch._utils._rebuild_parameter' || fn === 'torch._utils._rebuild_parameter_with_state') {
      // _rebuild_parameter(data, requires_grad): data is the rebuilt tensor (args[0])
      if (fn === 'torch._utils._rebuild_parameter_with_state') return args[0];
      return args[0];
    }
    if (g.name === 'OrderedDict' || fn === 'collections.OrderedDict') {
      // OrderedDict() with no args -> empty dict; with iterable -> dict
      if (!args || args.length === 0) return {};
      const d = {};
      for (const [k, v] of args[0]) d[k] = v;
      return d;
    }
    if (fn === 'builtins.dict' || g.name === 'dict') {
      if (!args || args.length === 0) return {};
      const d = {}; for (const [k, v] of args[0]) d[k] = v; return d;
    }
    // Unknown reduce: best-effort, return args (shouldn't happen for plain state_dicts)
    return { __reduce__: fn, args };
  }

  while (i < n) {
    const op = buf[i++];
    switch (op) {
      case OP.PROTO: i++; break;
      case OP.FRAME: i += 8; break;
      case OP.EMPTY_DICT: stack.push({}); break;
      case OP.EMPTY_TUPLE: stack.push([]); break;
      case OP.MARK: stack.push(MARK); break;
      case OP.STOP: return stack[stack.length - 1];
      case OP.NONE: stack.push(null); break;
      case OP.NEWFALSE: stack.push(false); break;
      case OP.NEWTRUE: stack.push(true); break;
      case OP.BININT: { const v = buf.readInt32LE(i); i += 4; stack.push(v); break; }
      case OP.BININT1: stack.push(buf[i++]); break;
      case OP.BININT2: { const v = buf.readUInt16LE(i); i += 2; stack.push(v); break; }
      case OP.LONG: { // newline-terminated decimal
        const e = buf.indexOf(0x0a, i); const s = buf.slice(i, e).toString('ascii').trimEnd('L'); i = e + 1;
        stack.push(parseInt(s, 10)); break;
      }
      case OP.LONG1: { const l = buf[i++]; const bytes = buf.slice(i, i + l); i += l; stack.push(twosComplement(bytes)); break; }
      case OP.LONG4: { const l = buf.readUInt32LE(i); i += 4; const bytes = buf.slice(i, i + l); i += l; stack.push(twosComplement(bytes)); break; }
      case OP.INT: { const e = buf.indexOf(0x0a, i); let s = buf.slice(i, e).toString('ascii'); i = e + 1;
        if (s === '01') stack.push(true); else if (s === '00') stack.push(false); else stack.push(parseInt(s.trimEnd('L'), 10)); break; }
      case OP.BINUNICODE: { const l = buf.readUInt32LE(i); i += 4; stack.push(buf.slice(i, i + l).toString('utf8')); i += l; break; }
      case OP.SHORT_BINUNICODE: { const l = buf[i++]; stack.push(buf.slice(i, i + l).toString('utf8')); i += l; break; }
      case OP.BINUNICODE8: { const l = Number(buf.readBigUInt64LE(i)); i += 8; stack.push(buf.slice(i, i + l).toString('utf8')); i += l; break; }
      case OP.SHORT_BINSTRING: { const l = buf[i++]; stack.push(buf.slice(i, i + l)); i += l; break; }
      case OP.BINSTRING: { const l = buf.readUInt32LE(i); i += 4; stack.push(buf.slice(i, i + l)); i += l; break; }
      case OP.GLOBAL: {
        const mEnd = buf.indexOf(0x0a, i); const module = buf.slice(i, mEnd).toString('ascii'); i = mEnd + 1;
        const nEnd = buf.indexOf(0x0a, i); const name = buf.slice(i, nEnd).toString('ascii'); i = nEnd + 1;
        stack.push({ __global__: true, module, name }); break;
      }
      case OP.STACK_GLOBAL: {
        const name = stack.pop(), module = stack.pop(); stack.push({ __global__: true, module, name }); break;
      }
      case OP.REDUCE: {
        const args = stack.pop(); const g = stack.pop(); stack.push(reduceGlobal(g, args)); break;
      }
      case OP.BUILD: { const state = stack.pop(); const obj = stack[stack.length - 1];
        if (obj && typeof obj === 'object') Object.assign(obj, state); break; }
      case OP.TUPLE: { const out = popToMark(stack); stack.push(out); break; }
      case OP.TUPLE1: { const a = stack.pop(); stack.push([a]); break; }
      case OP.TUPLE2: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; }
      case OP.TUPLE3: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; }
      case OP.SETITEM: { const v = stack.pop(), k = stack.pop(); const d = stack[stack.length - 1]; d[k] = v; break; }
      case OP.SETITEMS: {
        const items = popToMark(stack); const d = stack[stack.length - 1];
        for (let k = 0; k < items.length; k += 2) d[items[k]] = items[k + 1];
        break;
      }
      case OP.APPEND: { const v = stack.pop(); const lst = stack[stack.length - 1]; lst.push(v); break; }
      case OP.APPENDS: { const items = popToMark(stack); const lst = stack[stack.length - 1]; for (const v of items) lst.push(v); break; }
      case OP.BINPUT: memo[buf[i++]] = stack[stack.length - 1]; break;
      case OP.LONG_BINPUT: { const idx = buf.readUInt32LE(i); i += 4; memo[idx] = stack[stack.length - 1]; break; }
      case OP.BINGET: stack.push(memo[buf[i++]]); break;
      case OP.LONG_BINGET: { const idx = buf.readUInt32LE(i); i += 4; stack.push(memo[idx]); break; }
      case OP.MEMOIZE: memo.push(stack[stack.length - 1]); break;
      case OP.BINPERSID: { const pid = stack.pop(); stack.push(persistentLoad(pid)); break; }
      case OP.PERSID: { const e = buf.indexOf(0x0a, i); const id = buf.slice(i, e).toString('ascii'); i = e + 1;
        // not used by torch modern format; ignore
        stack.push({ __persid__: id }); break; }
      case OP.POP: stack.pop(); break;
      case OP.POP_MARK: popToMark(stack); break;
      case OP.DUP: stack.push(stack[stack.length - 1]); break;
      default:
        throw new Error('unsupported pickle opcode 0x' + op.toString(16) + ' at ' + (i - 1));
    }
  }
  throw new Error('pickle: no STOP');
}

function popToMark(stack) {
  const out = [];
  while (stack.length) {
    const v = stack.pop();
    if (v === MARK) return out;
    out.unshift(v);
  }
  return out;
}
function twosComplement(bytes) {
  if (bytes.length === 0) return 0;
  let v = 0;
  for (let k = bytes.length - 1; k >= 0; k--) v = v * 256 + bytes[k];
  if (bytes[bytes.length - 1] & 0x80) v -= 2 ** (8 * bytes.length);
  return v;
}

// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------
function loadArchive(zipPath) {
  const zip = readZip(zipPath);
  // find the data.pkl entry (prefix may vary: 'archive/data.pkl' or 'data.pkl')
  const pklName = Object.keys(zip.entries).find((n) => n.endsWith('/data.pkl') || n === 'data.pkl');
  if (!pklName) throw new Error('no data.pkl in archive');
  const pklEntry = zip.entries[pklName];
  const pklBuf = Buffer.alloc(pklEntry.size);
  fs.readSync(zip.fd, pklBuf, 0, pklEntry.size, pklEntry.dataOffset);
  const descriptors = parsePickle(pklBuf); // {name: {storageKey, dtype, storage_offset, size, stride}}
  return { zip, descriptors };
}

// Materialize a single tensor's data as a Float32Array.
function readTensor(zip, desc) {
  const entryName = Object.keys(zip.entries).find((n) => n === `pytorch_model/data/${desc.storageKey}`);
  const entry = zip.entries[entryName];
  if (!entry) throw new Error('storage not found: ' + desc.storageKey);
  const numel = desc.size.reduce((a, b) => a * b, 1);
  const bytesPerEl = 4;
  const byteOff = entry.dataOffset + desc.storage_offset * bytesPerEl;
  const byteLen = numel * bytesPerEl;
  const out = new Float32Array(numel);
  const tmp = Buffer.alloc(byteLen);
  fs.readSync(zip.fd, tmp, 0, byteLen, byteOff);
  const view = new Float32Array(tmp.buffer, tmp.byteOffset, numel);
  out.set(view);
  return out;
}

// Read a tensor's float bytes directly into a target Float32Array view at targetOff (floats).
function readTensorInto(zip, desc, target, targetOff) {
  const entryName = Object.keys(zip.entries).find((n) => n === `pytorch_model/data/${desc.storageKey}`);
  const entry = zip.entries[entryName];
  if (!entry) throw new Error('storage not found: ' + desc.storageKey);
  const numel = desc.size.reduce((a, b) => a * b, 1);
  const bytesPerEl = 4;
  const byteOff = entry.dataOffset + desc.storage_offset * bytesPerEl;
  const byteLen = numel * bytesPerEl;
  // overlay a Buffer view on the target region and read straight into it (no extra copy)
  const buf = Buffer.from(target.buffer, targetOff * bytesPerEl, byteLen);
  fs.readSync(zip.fd, buf, 0, byteLen, byteOff);
}

// Load the full state dict: {name: {data: Float32Array, shape: number[]}}.
function loadStateDict(zipPath) {
  const { zip, descriptors } = loadArchive(zipPath);
  const sd = {};
  for (const [name, desc] of Object.entries(descriptors)) {
    sd[name] = { data: readTensor(zip, desc), shape: desc.size };
  }
  try { fs.closeSync(zip.fd); } catch (e) {}
  return sd;
}

// Load weights directly into a shared Float32Array `heap` (e.g. a WASM LinearMemory
// view), assigning each param a 16-byte-aligned region. Returns {sd, totalFloats}
// where sd[name] = {data: Float32Array view of heap, shape, off (float offset)}.
// No per-param allocation; the heap holds all weights contiguously (used by --fast).
function loadStateDictShared(zipPath, heap) {
  const { zip, descriptors } = loadArchive(zipPath);
  const names = Object.keys(descriptors);
  // assign aligned offsets (float offset multiple of 4 => 16-byte aligned for v128)
  let off = 0;
  const sd = {};
  for (const name of names) {
    const desc = descriptors[name];
    const numel = desc.size.reduce((a, b) => a * b, 1);
    if (off % 4 !== 0) off += 4 - (off % 4); // align to 16 bytes
    readTensorInto(zip, desc, heap, off);
    sd[name] = { data: heap.subarray(off, off + numel), shape: desc.size, off, numel };
    off += numel;
  }
  try { fs.closeSync(zip.fd); } catch (e) {}
  return { sd, totalFloats: off };
}

module.exports = { loadArchive, loadStateDict, loadStateDictShared, readTensor, readTensorInto, parsePickle, readZip };
