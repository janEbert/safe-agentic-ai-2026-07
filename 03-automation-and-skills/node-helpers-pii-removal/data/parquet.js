'use strict';
/*
 * Minimal Apache Parquet writer using ONLY the Node.js standard library.
 *
 * No npm dependencies. Produces files readable by pyarrow / DuckDB / pandas.
 *
 * Scope (kept small on purpose; sufficient for tabular dataset subsets):
 *   - A single row group, one data page (V1) per column.
 *   - All columns are OPTIONAL (max definition level = 1, repetition level = 0),
 *     so NULLs are handled losslessly via definition levels.
 *   - PLAIN encoding for values; UNCOMPRESSED (no codec dependency).
 *   - Definition levels use the RLE/Bit-Packing-Hybrid encoding (= parquet
 *     `Encoding.RLE`), declared in each page header.
 *   - Type inference: BOOLEAN, INT64, DOUBLE, BYTE_ARRAY/UTF8 (string).
 *     Arrays/objects are JSON-serialized into UTF8 BYTE_ARRAY so nothing is lost.
 *
 * Parquet metadata is serialized with a hand-written Thrift *compact protocol*
 * encoder (the format parquet uses for its footer).
 *
 * Thrift compact-protocol gotchas (verified against pyarrow's output):
 *   - i16/i32/i64 FIELD VALUES are zigzag-varint encoded.
 *   - BINARY (string) LENGTHS and list/map/set SIZES are PLAIN (unsigned)
 *     varint, NOT zigzag.
 *   - DataPage V1 prefixes the definition levels with a 4-byte LE length of
 *     the encoded levels (repetition levels are omitted when max_rep_level=0).
 *   - RLE/Bit-Packing-Hybrid: an RLE run header is `(count << 1)` (LSB=0);
 *     a bit-packed run header is `((count/8) << 1) | 1` (LSB=1). We emit
 *     RLE runs only.
 *
 * File layout produced:
 *     PAR1
 *     <column chunk 1: page header + page data> ...
 *     <FileMetaData (thrift compact)>
 *     <4-byte LE length of FileMetaData>
 *     PAR1
 */

const fs = require('node:fs');

// --- parquet thrift enum values --------------------------------------------
const Type = { BOOLEAN: 0, INT32: 1, INT64: 2, INT96: 3, FLOAT: 4, DOUBLE: 5, BYTE_ARRAY: 6, FIXED_LEN_BYTE_ARRAY: 7 };
const FieldRepetitionType = { REQUIRED: 0, OPTIONAL: 1, REPEATED: 2 };
const CompressionCodec = { UNCOMPRESSED: 0 };
const Encoding = { PLAIN: 0, RLE: 3 };
const PageType = { DATA_PAGE: 0 };
const ConvertedType = { UTF8: 0 };

// --- Thrift compact protocol writer ----------------------------------------
// Compact type ids used in field/list headers.
const CT = { TRUE: 1, FALSE: 2, BYTE: 3, I16: 4, I32: 5, I64: 6, DOUBLE: 7, BINARY: 8, LIST: 9, SET: 10, MAP: 11, STRUCT: 12 };

class Thrift {
  constructor() {
    this.b = [];          // byte accumulator
    this.lastStack = [0];  // last field id per struct frame
  }
  get last() { return this.lastStack[this.lastStack.length - 1]; }
  set last(v) { this.lastStack[this.lastStack.length - 1] = v; }

  writeByte(x) { this.b.push(x & 0xff); }

  _varint(v) {
    v = BigInt(v);
    if (v < 0n) v += (1n << 64n); // wrap (shouldn't happen for zigzag output)
    do {
      let byte = Number(v & 0x7fn);
      v >>= 7n;
      if (v !== 0n) byte |= 0x80;
      this.b.push(byte);
    } while (v !== 0n);
  }

  writeI16(n) { this._varint((BigInt(n) << 1n) ^ (BigInt(n) >> 15n)); }
  writeI32(n) { this._varint((BigInt(n) << 1n) ^ (BigInt(n) >> 31n)); }
  writeI64(n) { this._varint((BigInt(n) << 1n) ^ (BigInt(n) >> 63n)); }
  // Plain (non-zigzag) varint, for binary lengths and collection sizes.
  writeVarint(n) { this._varint(BigInt(n)); }

  writeBinary(s) {
    const u = Buffer.from(s, 'utf8');
    this.writeVarint(u.length); // parquet compact protocol: length is PLAIN varint
    for (let i = 0; i < u.length; i++) this.b.push(u[i]);
  }

  // Write a field header. For non-bool types, follow with the value.
  writeField(fieldId, compactType) {
    const delta = fieldId - this.last;
    if (delta > 0 && delta <= 15) this.writeByte((delta << 4) | (compactType & 0x0f));
    else { this.writeByte(compactType & 0x0f); this.writeI16(fieldId); }
    this.last = fieldId;
  }
  // Booleans encode their value in the field-header type id (TRUE/FALSE).
  writeBool(fieldId, val) { this.writeField(fieldId, val ? CT.TRUE : CT.FALSE); }

  structBegin() { this.lastStack.push(0); }
  structEnd() { this.writeByte(0); this.lastStack.pop(); } // 0x00 = stop

  listBegin(elemType, size) {
    if (size < 15) this.writeByte((size << 4) | (elemType & 0x0f));
    else { this.writeByte(0xf0 | (elemType & 0x0f)); this.writeVarint(size); } // size is PLAIN varint
  }

  bytes() { return Buffer.from(this.b); }
}

// --- value & level encoders ------------------------------------------------
function encodePlainValues(type, values) {
  if (type === Type.BOOLEAN) {
    const bytes = []; let acc = 0, bits = 0;
    for (const v of values) { if (v) acc |= (1 << bits); if (++bits === 8) { bytes.push(acc); acc = 0; bits = 0; } }
    if (bits > 0) bytes.push(acc);
    return Buffer.from(bytes);
  }
  if (type === Type.INT64) {
    const b = Buffer.alloc(values.length * 8);
    for (let i = 0; i < values.length; i++) b.writeBigInt64LE(BigInt(Math.trunc(Number(values[i]))), i * 8);
    return b;
  }
  if (type === Type.DOUBLE) {
    const b = Buffer.alloc(values.length * 8);
    for (let i = 0; i < values.length; i++) b.writeDoubleLE(Number(values[i]), i * 8);
    return b;
  }
  if (type === Type.BYTE_ARRAY) {
    const parts = [];
    for (const v of values) {
      const s = typeof v === 'string' ? v : JSON.stringify(v);
      const u = Buffer.from(s, 'utf8');
      const len = Buffer.alloc(4); len.writeUInt32LE(u.length, 0);
      parts.push(len, u);
    }
    return Buffer.concat(parts);
  }
  throw new Error('unsupported parquet type: ' + type);
}

// RLE/Bit-Packing-Hybrid encoding for definition levels. We emit pure RLE
// runs (one per maximal run of equal levels), which any compliant reader
// accepts and which is trivial to generate correctly.
function encodeRLELevels(levels, bitWidth) {
  if (levels.length === 0) return Buffer.alloc(0);
  const valueBytes = Math.max(1, Math.ceil(bitWidth / 8));
  const out = [];
  const pushVarint = (n) => { let v = BigInt(n); do { let byte = Number(v & 0x7fn); v >>= 7n; if (v !== 0n) byte |= 0x80; out.push(byte); } while (v !== 0n); };
  const pushValue = (val) => { for (let k = 0; k < valueBytes; k++) { out.push(val & 0xff); val = Math.floor(val / 256); } };
  const MAX_RUN = 1 << 20; // split very long runs to stay well within reader limits
  let i = 0;
  while (i < levels.length) {
    const v = levels[i];
    let j = i; while (j < levels.length && levels[j] === v) j++;
    let run = j - i;
    while (run > 0) {
      const chunk = Math.min(run, MAX_RUN);
      pushVarint(BigInt(chunk) << 1n); // RLE header: LSB=0 (per parquet spec; LSB=1 would mean bit-packed)
      pushValue(v);
      run -= chunk;
    }
    i = j;
  }
  return Buffer.from(out);
}

// --- schema inference ------------------------------------------------------
function inferColumns(rows) {
  const names = []; const seen = new Set();
  for (const r of rows) for (const k of Object.keys(r)) if (!seen.has(k)) { seen.add(k); names.push(k); }
  return names.map((name) => {
    const vals = rows.map((r) => r[name]).filter((v) => v !== null && v !== undefined);
    let type, convertedType;
    if (vals.length === 0) { type = Type.BYTE_ARRAY; convertedType = ConvertedType.UTF8; }
    else if (vals.every((v) => typeof v === 'boolean')) { type = Type.BOOLEAN; }
    else if (vals.every((v) => typeof v === 'number' && Number.isSafeInteger(v))) { type = Type.INT64; }
    else if (vals.every((v) => typeof v === 'number')) { type = Type.DOUBLE; }
    else { type = Type.BYTE_ARRAY; convertedType = ConvertedType.UTF8; }
    return { name, type, convertedType };
  });
}

// --- thrift struct writers -------------------------------------------------
function writeSchemaElement(t, s) {
  t.structBegin();
  if (s.type !== undefined) { t.writeField(1, CT.I32); t.writeI32(s.type); }
  if (s.typeLength !== undefined) { t.writeField(2, CT.I32); t.writeI32(s.typeLength); }
  if (s.repetitionType !== undefined) { t.writeField(3, CT.I32); t.writeI32(s.repetitionType); }
  t.writeField(4, CT.BINARY); t.writeBinary(s.name);
  if (s.numChildren !== undefined) { t.writeField(5, CT.I32); t.writeI32(s.numChildren); }
  if (s.convertedType !== undefined) { t.writeField(6, CT.I32); t.writeI32(s.convertedType); }
  t.structEnd();
}

function writePageHeader(t, uncompressed, compressed, numValues) {
  t.structBegin();
  t.writeField(1, CT.I32); t.writeI32(PageType.DATA_PAGE);
  t.writeField(2, CT.I32); t.writeI32(uncompressed);
  t.writeField(3, CT.I32); t.writeI32(compressed);
  t.writeField(5, CT.STRUCT); // data_page_header
  t.structBegin();
  t.writeField(1, CT.I32); t.writeI32(numValues);
  t.writeField(2, CT.I32); t.writeI32(Encoding.PLAIN);
  t.writeField(3, CT.I32); t.writeI32(Encoding.RLE); // definition_level_encoding
  t.writeField(4, CT.I32); t.writeI32(Encoding.RLE); // repetition_level_encoding
  t.structEnd();
  t.structEnd();
}

// Build a single column's page header + page data.
function buildColumnPage(rows, col) {
  const defLevels = []; const nonNull = [];
  for (const r of rows) {
    const v = r[col.name];
    if (v === null || v === undefined) defLevels.push(0);
    else { defLevels.push(1); nonNull.push(v); }
  }
  const defBytes = encodeRLELevels(defLevels, 1); // max_def_level = 1
  const valBytes = encodePlainValues(col.type, nonNull);
  // DataPage V1: definition levels are prefixed with a 4-byte LE length of the
  // encoded levels. Repetition levels are omitted (max_rep_level = 0).
  const defLen = Buffer.alloc(4);
  defLen.writeUInt32LE(defBytes.length, 0);
  const pageData = Buffer.concat([defLen, defBytes, valBytes]);
  const numValues = defLevels.length;

  const t = new Thrift();
  writePageHeader(t, pageData.length, pageData.length, numValues);
  return { header: t.bytes(), data: pageData, numValues };
}

function buildFileMetaData(schemaEls, numRows, colChunks, totalByteSize, createdBy) {
  const t = new Thrift();
  t.structBegin(); // FileMetaData
  t.writeField(1, CT.I32); t.writeI32(1); // version
  t.writeField(2, CT.LIST); t.listBegin(CT.STRUCT, schemaEls.length);
  for (const s of schemaEls) writeSchemaElement(t, s);
  t.writeField(3, CT.I64); t.writeI64(numRows);
  t.writeField(4, CT.LIST); t.listBegin(CT.STRUCT, 1); // one row group
  t.structBegin(); // RowGroup
  t.writeField(1, CT.LIST); t.listBegin(CT.STRUCT, colChunks.length);
  for (const c of colChunks) {
    t.structBegin(); // ColumnChunk
    t.writeField(2, CT.I64); t.writeI64(c.dataPageOffset); // file_offset
    t.writeField(3, CT.STRUCT); // meta_data
    t.structBegin(); // ColumnMetaData
    t.writeField(1, CT.I32); t.writeI32(c.type);
    t.writeField(2, CT.LIST); t.listBegin(CT.I32, 2); t.writeI32(Encoding.PLAIN); t.writeI32(Encoding.RLE);
    t.writeField(3, CT.LIST); t.listBegin(CT.BINARY, 1); t.writeBinary(c.name); // path_in_schema
    t.writeField(4, CT.I32); t.writeI32(CompressionCodec.UNCOMPRESSED);
    t.writeField(5, CT.I64); t.writeI64(c.numValues);
    t.writeField(6, CT.I64); t.writeI64(c.chunkLen); // total_uncompressed_size
    t.writeField(7, CT.I64); t.writeI64(c.chunkLen); // total_compressed_size
    t.writeField(9, CT.I64); t.writeI64(c.dataPageOffset); // data_page_offset
    t.structEnd(); // ColumnMetaData
    t.structEnd(); // ColumnChunk
  }
  t.writeField(2, CT.I64); t.writeI64(totalByteSize);
  t.writeField(3, CT.I64); t.writeI64(numRows);
  t.structEnd(); // RowGroup
  t.writeField(6, CT.BINARY); t.writeBinary(createdBy);
  t.structEnd(); // FileMetaData
  return t.bytes();
}

// --- public API ------------------------------------------------------------
function writeParquet(rows, filePath) {
  if (!Array.isArray(rows)) throw new TypeError('rows must be an array of objects');
  const columns = inferColumns(rows);
  const numRows = rows.length;

  const schemaEls = [{ name: 'schema', numChildren: columns.length }];
  for (const c of columns) {
    schemaEls.push({
      name: c.name,
      type: c.type,
      repetitionType: FieldRepetitionType.OPTIONAL,
      ...(c.convertedType !== undefined ? { convertedType: c.convertedType } : {}),
    });
  }

  // Build all column chunks; record their absolute file offsets (after PAR1).
  let offset = 4;
  const colChunks = [];
  const chunkBuffers = [];
  for (const col of columns) {
    const page = buildColumnPage(rows, col);
    const chunk = Buffer.concat([page.header, page.data]);
    colChunks.push({ name: col.name, type: col.type, dataPageOffset: offset, chunkLen: chunk.length, numValues: page.numValues });
    chunkBuffers.push(chunk);
    offset += chunk.length;
  }
  const totalByteSize = chunkBuffers.reduce((s, b) => s + b.length, 0);

  const meta = buildFileMetaData(schemaEls, numRows, colChunks, totalByteSize, 'hf-data-dl (node stdlib parquet writer)');
  const metaLen = Buffer.alloc(4); metaLen.writeUInt32LE(meta.length, 0);

  const fd = fs.openSync(filePath, 'w');
  fs.writeSync(fd, Buffer.from('PAR1'));
  for (const chunk of chunkBuffers) fs.writeSync(fd, chunk);
  fs.writeSync(fd, meta);
  fs.writeSync(fd, metaLen);
  fs.writeSync(fd, Buffer.from('PAR1'));
  fs.closeSync(fd);
}

module.exports = { writeParquet, Type, Thrift };
