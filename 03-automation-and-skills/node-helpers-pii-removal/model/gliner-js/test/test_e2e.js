'use strict';
/* End-to-end: JS predict() vs Python reference decoded entities. */
const fs = require('node:fs');
const { GLiNER } = require('../src/run');

function main() {
  const modelDir = process.argv[2];
  const refMeta = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')).meta;
  const g = new GLiNER(modelDir).loadWeights();
  const text = refMeta.text, labels = refMeta.labels;
  const refEnts = refMeta.decoded_entities;
  const t0 = Date.now();
  const ents = g.predict(text, labels, 0.5);
  console.log(`predict took ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.log('JS entities:');
  for (const e of ents) console.log('  ', JSON.stringify(e));
  console.log('REF entities:');
  for (const e of refEnts) console.log('  ', JSON.stringify(e));
  // compare (round score to ~6 decimals)
  const norm = (a) => a.map((e) => ({ start: e.start, end: e.end, text: e.text, label: e.label, score: Math.round(e.score * 1e5) / 1e5 }));
  const js = JSON.stringify(norm(ents)), ref = JSON.stringify(norm(refEnts));
  console.log('MATCH:', js === ref);
  if (js !== ref) console.log('  js :', js, '\n  ref:', ref);
}
main();
