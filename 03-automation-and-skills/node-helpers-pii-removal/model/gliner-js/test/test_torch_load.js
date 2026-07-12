'use strict';
/* Validate torch_load.js against the Python reference state_dict. */
const fs = require('node:fs');
const path = require('node:path');
const { loadStateDict } = require('../src/torch_load');

const BIN = process.argv[2];
const REF = process.argv[3]; // JSON file: {name: {shape, sample:[...first N floats]}}

const sd = loadStateDict(BIN);
const names = Object.keys(sd);
console.log('JS loaded params:', names.length);

const ref = JSON.parse(fs.readFileSync(REF, 'utf8'));
const refNames = Object.keys(ref);
console.log('ref params:', refNames.length);

// name set match
const jsSet = new Set(names), refSet = new Set(refNames);
const missing = refNames.filter((n) => !jsSet.has(n));
const extra = names.filter((n) => !refSet.has(n));
console.log('missing in JS:', missing.length, missing.slice(0, 5));
console.log('extra in JS:', extra.length, extra.slice(0, 5));

// shape + sample value match for a few key tensors
let okShape = 0, badShape = [], okVal = 0, badVal = [];
for (const n of refNames) {
  if (!sd[n]) continue;
  const jsShape = sd[n].shape.join(',');
  if (jsShape === ref[n].shape.join(',')) okShape++; else badShape.push([n, jsShape, ref[n].shape.join(',')]);
  // compare first 8 floats
  const jsVals = Array.from(sd[n].data.slice(0, 8));
  const rVals = ref[n].sample;
  let vok = true;
  for (let k = 0; k < rVals.length; k++) {
    if (Math.abs(jsVals[k] - rVals[k]) > 1e-5) { vok = false; break; }
  }
  if (vok) okVal++; else badVal.push([n, jsVals.slice(0, 3), rVals.slice(0, 3)]);
}
console.log('shapes match:', okShape, '/', refNames.length, 'bad:', badShape.length);
if (badShape.length) console.log('  bad shapes sample:', badShape.slice(0, 5));
console.log('sample values match:', okVal, '/', refNames.length, 'bad:', badVal.length);
if (badVal.length) console.log('  bad vals sample:', badVal.slice(0, 5));
console.log('RESULT:', (missing.length === 0 && extra.length === 0 && badShape.length === 0 && badVal.length === 0) ? 'PASS' : 'FAIL');
