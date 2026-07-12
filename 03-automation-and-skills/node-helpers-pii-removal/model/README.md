# hf-model

Pure-Node.js-stdlib tooling for the HuggingFace model [`nvidia/gliner-PII`](https://huggingface.co/nvidia/gliner-PII):
download it, then **run it** (DeBERTa-v3-large + GLiNER span head) with zero
Python and zero npm dependencies at inference time.

Two pieces, each pure stdlib:

## 1. `download_model.js` — download the model

Resumable, with progress; skips files already complete; HF-token support.

```bash
node download_model.js --repo nvidia/gliner-PII --out-dir ./model
# or a subset:
node download_model.js --include 'pytorch_model.bin,gliner_config.json,tokenizer.json,spm.model'
```

## 2. `gliner-js/` — execute the model (pure stdlib)

A from-scratch, pure-JS reimplementation of the full `gliner` inference path —
PyTorch checkpoint parser, SentencePiece/Unigram tokenizer, DeBERTa-v3-large
encoder, GLiNER span head, and decoder — validated against the Python `gliner`
reference to **exact float agreement** (and **20/20 exact** on Nemotron-PII rows).

```bash
node --max-old-space-size=4096 gliner-js/src/run.js --model ./model \
  --text "My name is Jason, born 1987-05-22, phone 555-1234." \
  --labels person,date_of_birth,phone_number --threshold 0.5
```

See `gliner-js/README.md` for the validation matrix and architecture.

## End-to-end on Nemotron-PII

```bash
# 1. get a few dataset rows (text + gold PII spans) from the HF datasets-server
python3 - <<'PY'   # or any client
import json,ast,urllib.request
def get(off):
    u=f"https://datasets-server.huggingface.co/rows?dataset=nvidia/Nemotron-PII&config=default&split=train&offset={off}&length=100"
    return [r["row"] for r in json.load(urllib.request.urlopen(u))["rows"]]
rows=[]
for off in (0,100):
    for r in get(off):
        sp=r["spans"] if isinstance(r["spans"],list) else ast.literal_eval(r["spans"])
        if r["text"] and sp: rows.append({"text":r["text"],"labels":sorted(set(s["label"] for s in sp))})
        if len(rows)>=20: break
    if len(rows)>=20: break
open("rows.jsonl","w").write("\n".join(json.dumps(x) for x in rows)+"\n")
PY

# 2. run the pure-JS model on them
node --max-old-space-size=4096 gliner-js/src/run.js --model ./model \
  --input rows.jsonl --output out.jsonl
```

Validated result: 20/20 Nemotron-PII rows produce **byte-identical** entity
output to Python `gliner.predict_entities`, and F1 ≈ 0.96 vs the gold spans.
