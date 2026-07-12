'use strict';
const { Tokenizer } = require('../src/tokenizer');
const tok = new Tokenizer(process.argv[2]);
const prompt = ['<<ENT>>','first_name','<<ENT>>','date_of_birth','<<ENT>>','street_address','<<ENT>>','employment_status','<<ENT>>','phone_number','<<ENT>>','email','<<SEP>>'];
const words = ['I', ',', 'Jason', ',', 'am', 'applying', 'for', 'a', 'financial', 'services', 'account', '.', 'My', 'date', 'of', 'birth', 'is', '1987-05-22', '.', 'I', 'live', 'at', '87', 'Avenida', 'De', 'La', 'Estrella', '.'];
const { ids, wordIds } = tok.encodeWords(prompt.concat(words));
const refIds = [1,128002,362,616,8982,128002,1043,616,1580,616,31392,128002,2011,616,36163,128002,3192,616,35124,128002,932,616,24516,128002,871,128003,273,366,4887,366,481,4422,270,266,999,544,914,323,573,1043,265,2665,269,7405,271,3586,271,2944,323,273,685,288,9204,83495,2060,2025,92460,323,2];
const refWordIds = [null,0,1,1,1,2,3,3,3,3,3,4,5,5,5,6,7,7,7,8,9,9,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,30,30,30,30,31,32,33,34,35,36,37,38,39,40,null];
console.log('ids len:', ids.length, '(ref 59)');
console.log('IDS MATCH:', JSON.stringify(ids) === JSON.stringify(refIds));
console.log('WORD IDS MATCH:', JSON.stringify(wordIds) === JSON.stringify(refWordIds));
if (JSON.stringify(ids) !== JSON.stringify(refIds)) {
  console.log('mine:', JSON.stringify(ids));
  console.log('ref :', JSON.stringify(refIds));
  for (let i = 0; i < Math.max(ids.length, refIds.length); i++) {
    if (ids[i] !== refIds[i]) { console.log(`  diff @${i}: mine=${ids[i]} ref=${refIds[i]}`); break; }
  }
}
// also test a few raw words tokenized individually
const cases = [['Hello world.',['Hello','world','.'],[1,5365,447,260,2]], ['1987-05-22',['1987-05-22'],[1,7405,271,3586,271,2944,2]]];
// (single-word via encodeWords with [w] then strip CLS/SEP)
for (const [name, ws, ref] of cases) {
  const r = tok.encodeWords(ws);
  const got = r.ids;
  console.log(`${name}: ${JSON.stringify(got)} match=${JSON.stringify(got) === JSON.stringify(ref)}`);
}
