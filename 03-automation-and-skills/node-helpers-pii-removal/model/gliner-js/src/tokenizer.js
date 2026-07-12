'use strict';
/*
 * DeBERTa-v3 tokenizer for GLiNER, implemented from the HF `tokenizer.json`
 * using only the Node.js standard library.
 *
 * Pipeline (matching HF `tokenizers`):
 *   added-token split  ->  Normalizer (Strip + Precompiled SentencePiece charsmap)
 *                      ->  Metaspace pre-tokenizer (prepend ▁ per word)
 *                      ->  Unigram model (Viterbi)
 *                      ->  post-processor ([CLS] … [SEP])
 *
 * Used in `is_split_into_words` mode: each input "word" is tokenized
 * independently and tagged with a word id.
 */

const fs = require('node:fs');

// --------------------------------------------------------------------------
// Precompiled SentencePiece charsmap normalizer (DoubleArray trie).
// Format: [u32 trie_size_bytes][trie: u32...][normalized: utf8 string].
// --------------------------------------------------------------------------
class PrecompiledNormalizer {
  constructor(base64) {
    const raw = Buffer.from(base64, 'base64');
    let off = 0;
    const trieSize = raw.readUInt32LE(off); off += 4;
    const nUnits = trieSize >>> 2;
    this.array = new Uint32Array(nUnits);
    for (let k = 0; k < nUnits; k++) { this.array[k] = raw.readUInt32LE(off); off += 4; }
    this.normalized = raw.slice(off).toString('utf8');
    this.normBytes = Buffer.from(this.normalized, 'utf8');
  }
  _hasLeaf(u) { return ((u >>> 8) & 1) === 1; }
  _value(u) { return u & 0x7fffffff; }
  _label(u) { return (u & 0x800000ff) >>> 0; }
  _offset(u) { return ((u >>> 10) << ((u & 512) >>> 6)) >>> 0; }
  _commonPrefixSearch(keyBytes) {
    const a = this.array, results = [];
    let nodePos = 0;
    let unit = a[0];
    nodePos ^= this._offset(unit);
    for (let i = 0; i < keyBytes.length; i++) {
      const c = keyBytes[i];
      if (c === 0) break;
      nodePos ^= c;
      unit = a[nodePos];
      if (this._label(unit) !== c) return results;
      nodePos ^= this._offset(unit);
      if (this._hasLeaf(unit)) results.push(this._value(a[nodePos]));
    }
    return results;
  }
  transform(chunk) {
    const kb = Buffer.from(chunk, 'utf8');
    const r = this._commonPrefixSearch(kb);
    if (r.length === 0) return null;
    const index = r[0];
    let i2 = index;
    while (i2 < this.normBytes.length && this.normBytes[i2] !== 0) i2++;
    return this.normBytes.slice(index, i2).toString('utf8');
  }
  normalize(str) {
    // grapheme ~= base char + combining marks; for ASCII each char is a grapheme.
    const chars = [...str];
    let out = '';
    for (let i = 0; i < chars.length; i++) {
      // gather combining marks (Mn/Me/Mc ranges, simplified) into the grapheme
      let g = chars[i];
      while (i + 1 < chars.length) {
        const c = chars[i + 1];
        const cp = c.codePointAt(0);
        const combining = (cp >= 0x0300 && cp <= 0x036f) || (cp >= 0x1ab0 && cp <= 0x1aff) || (cp >= 0xfe20 && cp <= 0xfe2f);
        if (!combining) break;
        g += c; i++;
      }
      if (Buffer.byteLength(g, 'utf8') < 6) {
        const t = this.transform(g);
        if (t !== null) { out += t; continue; }
      }
      // per-char fallback
      for (const c of g) {
        const t = this.transform(c);
        out += (t !== null) ? t : c;
      }
    }
    return out;
  }
}

// --------------------------------------------------------------------------
// Unigram model with Viterbi segmentation.
// --------------------------------------------------------------------------
class Unigram {
  constructor(modelJson) {
    this.unkId = modelJson.unk_id;
    const vocab = modelJson.vocab;
    this.tokens = new Map(); // piece(string) -> [id, score]
    let maxLen = 0, minScore = Infinity;
    for (const [tok, score] of vocab) {
      if (!this.tokens.has(tok)) this.tokens.set(tok, [this.tokens.size, score]);
      if (tok.length > maxLen) maxLen = tok.length;
      if (score < minScore) minScore = score;
    }
    this.maxLen = maxLen;
    this.unkScore = minScore - 10.0; // K_UNK_PENALTY
    this.unkIdValue = this.tokens.has('[UNK]') ? this.tokens.get('[UNK]')[0] : this.unkId;
  }
  // Viterbi over chars; returns array of token ids.
  encode(s) {
    const chars = [...s];
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const back = new Array(n + 1).fill(null); // [prevPos, length, id]
    best[0] = 0;
    for (let p = 0; p < n; p++) {
      if (best[p] === -Infinity) continue;
      const maxL = Math.min(this.maxLen, n - p);
      let hasSingle = false;
      for (let L = 1; L <= maxL; L++) {
        const sub = chars.slice(p, p + L).join('');
        const t = this.tokens.get(sub);
        if (t) {
          if (L === 1) hasSingle = true;
          const end = p + L;
          const cand = best[p] + t[1];
          if (cand > best[end]) { best[end] = cand; back[end] = [p, L, t[0]]; }
        }
      }
      if (!hasSingle && this.unkId !== undefined) {
        const end = p + 1;
        const cand = best[p] + this.unkScore;
        if (cand > best[end]) { best[end] = cand; back[end] = [p, 1, this.unkId]; }
      }
    }
    // backtrack
    const ids = [];
    let pos = n;
    if (best[n] === -Infinity) {
      // fallback: emit unk per char
      for (let i = 0; i < n; i++) ids.push(this.unkId);
      return ids;
    }
    while (pos > 0) {
      const b = back[pos];
      if (!b) { ids.length = 0; for (let i = 0; i < n; i++) ids.push(this.unkId); return ids; }
      ids.push(b[2]);
      pos = b[0];
    }
    ids.reverse();
    return ids;
  }
}

// --------------------------------------------------------------------------
// Full tokenizer.
// --------------------------------------------------------------------------
class Tokenizer {
  constructor(jsonPath) {
    const t = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
    this.clsId = 1; this.sepId = 2; this.padId = 0;
    // added tokens (matched literally on raw input words)
    this.added = new Map();
    for (const a of t.added_tokens || []) this.added.set(a.content, a.id);
    // normalizer: Sequence of [Strip, Precompiled]
    this.normalizer = null;
    const normSeq = t.normalizer && t.normalizer.normalizers;
    if (normSeq) {
      const pre = normSeq.find((n) => n.type === 'Precompiled');
      if (pre) this.normalizer = new PrecompiledNormalizer(pre.precompiled_charsmap);
    }
    this.unigram = new Unigram(t.model);
  }

  normalizeWord(w) {
    if (this.normalizer) {
      let n = this.normalizer.normalize(w);
      return n.replace(/^\s+|\s+$/g, ''); // Strip
    }
    return w;
  }

  // Tokenize a single word -> list of ids (no special tokens).
  tokenizeWord(w) {
    if (this.added.has(w)) return [this.added.get(w)];
    const nw = this.normalizeWord(w);
    const s = '\u2581' + nw; // Metaspace: prepend ▁ (U+2581), prepend_scheme=always
    return this.unigram.encode(s);
  }

  // is_split_into_words: tokenizes a list of words; returns {ids, wordIds}.
  encodeWords(words) {
    const ids = [this.clsId];
    const wordIds = [null];
    for (let wi = 0; wi < words.length; wi++) {
      const tids = this.tokenizeWord(words[wi]);
      for (const id of tids) { ids.push(id); wordIds.push(wi); }
    }
    ids.push(this.sepId);
    wordIds.push(null);
    return { ids, wordIds };
  }
}

module.exports = { Tokenizer, PrecompiledNormalizer, Unigram };
