'use strict';
/* Run JS GLiNER on the Nemotron-PII rows and compare to the Python reference output. */
const fs = require('node:fs');
const { GLiNER } = require('../src/run');

function main() {
  const modelDir = process.argv[2];
  const rows = fs.readFileSync('nemotron/rows.jsonl', 'utf8').trim().split('\n').map(JSON.parse);
  const refOut = fs.readFileSync('nemotron/ref_out.jsonl', 'utf8').trim().split('\n').map(JSON.parse);
  const g = new GLiNER(modelDir).loadWeights();
  const jsOut = [];
  let exactMatch = 0, totalRows = rows.length;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const labels = [...new Set(r.spans.map((s) => s.label))].sort();
    const t0 = Date.now();
    const ents = g.predict(r.text, labels, 0.5);
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    const js = ents.map((e) => ({ start: e.start, end: e.end, text: e.text, label: e.label, score: Math.round(e.score * 1e5) / 1e5 }));
    const refEnts = refOut[i].entities;
    const match = JSON.stringify(js) === JSON.stringify(refEnts);
    if (match) exactMatch++;
    jsOut.push({ text: r.text, labels, entities: js });
    console.log(`row ${i}: ${dt}s ents=${ents.length} ref=${refEnts.length} ${match ? 'EXACT' : 'DIFF'}`);
    if (!match) {
      const jsSet = new Set(js.map((e) => `${e.start}-${e.end}-${e.label}`));
      const refSet = new Set(refEnts.map((e) => `${e.start}-${e.end}-${e.label}`));
      const onlyJs = [...jsSet].filter((x) => !refSet.has(x));
      const onlyRef = [...refSet].filter((x) => !jsSet.has(x));
      if (onlyJs.length) console.log('    only JS:', onlyJs.slice(0, 5));
      if (onlyRef.length) console.log('    only REF:', onlyRef.slice(0, 5));
    }
  }
  fs.writeFileSync('nemotron/js_out.jsonl', jsOut.map((x) => JSON.stringify(x)).join('\n') + '\n');
  console.log(`\nEXACT match: ${exactMatch}/${totalRows} rows`);
}
main();
