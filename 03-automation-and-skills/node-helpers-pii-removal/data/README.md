# hf-data-dl (Node.js)

Download a **configurable subset** (first *N* rows) of a HuggingFace dataset
using **only the Node.js standard library** — no `npm install`, no
`node_modules`. Cross-platform (macOS / Linux / Windows).

Works out of the box with the example dataset
[`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII).

## Requirements

- **Node.js >= 18** (uses the built-in global `fetch`).

That's it — there are no dependencies.

## Quick start

```bash
# 100 rows of the train split, JSONL (auto-named file)
node download_subset.js --rows 100

# 1 000 rows as parquet (written by a pure-stdlib parquet writer)
node download_subset.js -n 1000 --split train --format parquet -o subset.parquet

# 50 rows as CSV
node download_subset.js -n 50 --format csv

# Discover available configs / splits / sizes, then exit
node download_subset.js --list-splits
```

## Usage

```
node download_subset.js [--dataset ID] [--config CONFIG] [--split SPLIT]
                        [-n ROWS] [--offset N] [--format jsonl|csv|parquet]
                        [-o OUT] [--token TOKEN] [--list-splits]
```

| Option        | Default                | Description                                  |
|---------------|------------------------|----------------------------------------------|
| `--dataset`   | `nvidia/Nemotron-PII`  | HF dataset id                                |
| `--config`    | `default`              | Dataset config name                          |
| `--split`     | `train`                | Split name (`train`/`test`/`validation`/…)  |
| `-n`/`--rows` | `100`                  | Number of rows to download                   |
| `--offset`    | `0`                    | Skip this many rows first                    |
| `--format`    | `jsonl`                | Output format (`jsonl`, `csv`, `parquet`)    |
| `-o`/`--output` | auto-named           | Output file path                             |
| `--token`     | `HF_TOKEN` env         | HF token, required for gated/private data    |
| `--list-splits` | —                    | Print configs/splits/sizes and exit          |

Auto-named outputs look like
`nvidia__Nemotron-PII_train_100rows_offset0.jsonl`.

## How it works

- Data is fetched from the **HuggingFace datasets-server REST API**:
  - `/rows` for the actual rows (paginated 100 rows/request), and
  - `/size` for `--list-splits`.
- HTTP uses Node's built-in `fetch` with retry/backoff on `429`/`5xx` and
  network errors.
- **Parquet** output is produced by a from-scratch writer (`parquet.js`) that
  uses only the standard library: a hand-written Thrift *compact-protocol*
  encoder for the footer, PLAIN value encoding, RLE definition levels
  (OPTIONAL columns, so NULLs are lossless), and UNCOMPRESSED pages. Output is
  verified readable by **pyarrow** and **pandas** (and any parquet-cpp reader).
- Type inference per column: `boolean` → BOOLEAN, safe integers → INT64,
  other numbers → DOUBLE, everything else (incl. strings/arrays/objects) →
  UTF8 BYTE_ARRAY (arrays/objects are JSON-serialized so nothing is lost).

## Files

- `download_subset.js` — CLI, HTTP, JSONL/CSV writers.
- `parquet.js` — pure-stdlib parquet writer (used only for `--format parquet`).
- `package.json` — metadata only; no `dependencies`.

## Gated / private datasets

```bash
export HF_TOKEN=hf_xxx
node download_subset.js --dataset some/gated-dataset --token $HF_TOKEN
```

## Inspecting the result

```js
const fs = require('fs');
const rows = fs.readFileSync('nvidia__Nemotron-PII_train_100rows_offset0.jsonl', 'utf8')
  .trim().split('\n').map(JSON.parse);
console.log(rows[0].text);
// `spans` is a Python-repr string -> not JSON; parse with care if needed.
```
