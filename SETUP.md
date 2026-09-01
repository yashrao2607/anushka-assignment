# SETUP — Run This Project on Any Machine

**Project:** Semantic Search / Intelligent Q&A Agent
**Time to first result:** ~10 minutes (most of it a one-time model download)
**Works on:** Windows, macOS, Linux · Python 3.10–3.12 · **CPU only, no GPU needed**

This guide is written so that someone who has never seen the project can clone it
and reproduce every number in the reports.

---

## TL;DR — the whole thing in six commands

```bash
git clone <your-repo-url>
cd "anushka assignment/semantic-qa-agent"

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt

python -m src.cli ingest        # documents -> clean, traceable chunks
python -m src.cli index         # chunks -> vector + BM25 indexes
python -m src.cli evaluate      # measure it (no API key needed)
```

That gets you the full retrieval system and every retrieval metric **with no API
key and no network access after install**. Only the `ask` / `judge` commands need
a key — see Step 4.

---

## Step 0 — Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10–3.12** | Check with `python --version`. 3.13 may not yet have wheels for every dependency. |
| **~2.5 GB free disk** | Mostly PyTorch, which `sentence-transformers` pulls in. |
| **Internet, on first run only** | To download two models (~110 MB total). After that the system runs fully offline. |
| **A Groq API key** | **Optional.** Only needed for `ask` and `judge`. Free at [console.groq.com](https://console.groq.com). |

You do **not** need a GPU, Docker, or a database server. The vector store is
embedded and writes to a local folder.

---

## Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd "anushka assignment/semantic-qa-agent"
```

> **Important:** every command below is run from inside the **`semantic-qa-agent/`**
> directory, not the repository root. If you see `No module named src`, you are
> one level too high.

---

## Step 2 — Create a virtual environment

Strongly recommended — this project installs PyTorch, which you do not want in
your global Python.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then retry.

**Windows (cmd)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

---

## Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes 3–8 minutes, dominated by PyTorch. Verify:

```bash
python -c "import torch, sentence_transformers, chromadb, rank_bm25; print('all good')"
```

---

## Step 4 — Add your Groq API key *(optional — skip to Step 5 to run without one)*

Create a file called **`.env`** inside `semantic-qa-agent/`:

```
GROQ_API_KEY=gsk_your_key_here
```

There is a template at `.env.example`. **`.env` is gitignored and must never be
committed.**

### What works without a key

| Command | Needs a key? |
|---|---|
| `ingest`, `index`, `stats`, `inspect` | ❌ No |
| `query` — semantic search | ❌ No |
| `evaluate` — all retrieval metrics and the full ablation | ❌ No |
| `calibrate` — refusal-threshold calibration | ❌ No |
| `scripts/failure_analysis.py` | ❌ No |
| `ask` — generate a written answer | ✅ Yes |
| `judge` — answer-quality evaluation | ✅ Yes |
| Streamlit UI | Partly — retrieval works; uncheck *"Generate an answer"* to run key-free |

**About 90% of this project — including every retrieval number and the entire
refusal calibration — is verifiable with no key at all.**

### Which model to use

The code defaults to `openai/gpt-oss-120b`. Groq periodically changes which
models an account can access, so if you get a **404 `model_not_found`**, list what
your key actually has:

```bash
python -c "import os,requests,sys; sys.path.insert(0,'.'); from pathlib import Path; from src.generate.groq_client import load_env; load_env(Path('.')); print('\n'.join(sorted(m['id'] for m in requests.get('https://api.groq.com/openai/v1/models', headers={'Authorization':'Bearer '+os.environ['GROQ_API_KEY']}).json()['data'])))"
```

Then set your choice in `config/default.yaml`:

```yaml
generation:
  provider: groq
  model: openai/gpt-oss-120b     # <- change to a model your key can access
```

---

## Step 5 — Build the index

A fresh clone contains the **source documents** but not the generated artefacts
(chunks, vectors, indexes) — those are gitignored because they are derived data.
Two commands regenerate everything:

```bash
python -m src.cli ingest
python -m src.cli index
```

**Expected output**

```
INGESTION SUMMARY                         INDEX BUILD
Files found                    16         chunks              48
Files ingested                 15         dim                 384
Files skipped (unsupported)     1         backend     ChromaStore
Files failed                    0         bm25_backend  BM25Okapi
Chunks produced                48
```

*The 1 skipped file is a deliberate `.rtf` decoy — it proves unsupported formats
are skipped with a warning rather than crashing the run.*

`index` downloads the embedding model (~80 MB) the first time. This is the only
slow step; it is cached in `~/.cache/huggingface` afterwards.

---

## Step 6 — Try it

### Semantic vs keyword, side by side

```bash
python -m src.cli query "Can I get my money back if a client cancels a trip?" --compare
```

Runs BM25, dense and hybrid on the same query. This is the fastest way to *see*
the vocabulary-mismatch problem the project exists to solve.

### Ask a question (needs a key)

```bash
python -m src.cli ask "How many casual leaves do interns get?"
```

```
Interns accrue casual leave at 0.5 days per completed month, up to a
maximum of 6 days per year[1].

confidence 0.965 | threshold 0.02 | answered

SOURCES
  [1] Hr Policy > 1. Leave Entitlement (p.1)
```

### Watch it refuse (costs **zero** API calls)

```bash
python -m src.cli ask "What is the policy on employees keeping pets in the office?"
```

```
I could not find information about this in the provided documents.
confidence 0.000 | threshold 0.02 | REFUSED
reason: top relevance 0.000 < threshold 0.02  (no LLM call was made)
```

The refusal gate sits **before** the model, so the system cannot hallucinate on
an out-of-corpus question — that is a structural guarantee, not a prompt request.

### The web UI

```bash
streamlit run app.py
```
Opens at <http://localhost:8501>. Uncheck *"Generate an answer"* in the sidebar to
use it with no API key.

---

## Step 7 — Reproduce the reported numbers

```bash
python -m src.cli evaluate            # retrieval metrics + full ablation (no key)
python -m src.cli calibrate           # refusal threshold sweep (no key)
python scripts/failure_analysis.py    # diagnose every remaining failure (no key)
python -m src.cli judge --n 12 --max-calls 40   # answer quality (needs a key)
python -m pytest tests/ -q            # 109 tests
```

**Expected from `evaluate`:**

```
A0  Keyword baseline (BM25 only)     P@3 0.314  R@10 0.853  MRR 0.738  nDCG 0.738
A1  Dense only (MiniLM)              P@3 0.366  R@10 0.971  MRR 0.898  nDCG 0.886
A4  Hybrid (dense + BM25, RRF)       P@3 0.379  R@10 0.980  MRR 0.852  nDCG 0.861
A5  Hybrid + cross-encoder re-rank   P@3 0.431  R@10 1.000  MRR 0.941  nDCG 0.942
```

Reports land in `reports/`. Anything there is regenerated by the command above,
so nothing in this repository is a hand-written number.

---

## Free-tier / API-cost notes

This project was built and tested against a **free** Groq key, and the cost
controls are part of the design:

- **Every response is cached to disk** (`.cache/groq_responses.json`) keyed on the
  prompt. Re-running any report costs **zero** calls. Generation is pinned to
  `temperature = 0.0`, so caching is semantically correct, not just a shortcut.
- **Refusals never call the API** — the gate runs before generation.
- **`--max-calls` is a hard budget.** Exceeding it raises a clear error rather
  than silently draining your daily quota.
- **Rate limiting with 429 backoff** honours the server's `retry-after` header.
- The full answer-quality evaluation used **22 live calls / 21k tokens** total.

To run with strict cost control: `python -m src.cli judge --n 6 --max-calls 15`

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named src` | You are in the repo root. `cd semantic-qa-agent` first. |
| `no chunks found -- run ingest first` | Run `python -m src.cli ingest`, then `index`. |
| `GROQ_API_KEY is not set` | Create `.env` in `semantic-qa-agent/` with `GROQ_API_KEY=gsk_...`. Or just use the key-free commands. |
| **404 `model_not_found`** | Your key cannot access that model. List available models (Step 4) and update `config/default.yaml`. |
| `UnicodeEncodeError ... charmap` on Windows | Should not happen — the CLI forces UTF-8. If a custom script hits it: `set PYTHONIOENCODING=utf-8`. |
| First query takes 10–80 seconds | One-time model download. Subsequent runs are milliseconds. |
| `429 rate limited` | The client backs off automatically. If persistent, wait a minute or lower `--n`. |
| `chromadb` fails to install | Not fatal. The code falls back to exact numpy search automatically and logs a warning. Or force it: `python -m src.cli index --backend numpy`. |
| `symlinks / Developer Mode` warning on Windows | Harmless HuggingFace cache warning. Ignore it. |
| Results differ slightly from the reports | Check the config fingerprint printed by `ingest`. It should be `a9cba3e539a3`. A different value means the chunking config changed. |
| Want to start completely fresh | Delete `storage/`, `data/processed/`, `data/manifest.csv`, `.cache/` and re-run `ingest` + `index`. |

---

## Using your own documents

```bash
# Drop your files into data/raw/ (or point at any folder)
python -m src.cli ingest --path /path/to/your/documents
python -m src.cli index
python -m src.cli ask "your question"
```

Supported: **PDF, DOCX, Markdown, TXT, HTML, CSV.** Unsupported files are skipped
with a warning, never a crash.

Tune chunking without touching code:

```bash
python -m src.cli ingest --chunk-size 512 --overlap 64
```

> **Note:** the golden set in `data/golden_set.jsonl` is written against the
> included handbook corpus. If you swap the corpus, `evaluate` numbers become
> meaningless until you write golden questions for the new documents. Retrieval
> and `ask` still work fine.

All settings live in [`config/default.yaml`](semantic-qa-agent/config/default.yaml).

---

## What gets created where

| Path | Contents | In git? |
|---|---|---|
| `data/raw/` | Source documents | ✅ yes |
| `data/golden_set.jsonl` | 60 hand-authored evaluation questions | ✅ yes |
| `config/default.yaml` | Every tunable setting | ✅ yes |
| `data/processed/chunks.jsonl` | Generated chunks | ❌ regenerated by `ingest` |
| `data/manifest.csv` | Per-document audit record | ❌ regenerated by `ingest` |
| `storage/` | Chroma vector store + BM25 index | ❌ regenerated by `index` |
| `.cache/` | Embedding + Groq response caches | ❌ local only |
| `.env` | **Your API key — never commit this** | ❌ gitignored |
| `reports/` | All measured results | ✅ yes |
| `logs/` | Structured JSONL traces | ❌ local only |

---

## Where to read next

| Document | What it covers |
|---|---|
| [`semantic-qa-agent/README.md`](semantic-qa-agent/README.md) | Architecture and headline results |
| [`PRD_Project1_Semantic_Search_QA_Agent.md`](PRD_Project1_Semantic_Search_QA_Agent.md) | Full product requirements |
| [`PHASE_PLAN_Project1.md`](PHASE_PLAN_Project1.md) | The 3-phase × 4-part execution plan |
| `reports/PHASE1_REPORT.md` | Ingestion — design decisions and verification |
| `reports/PHASE2_REPORT.md` | Retrieval — **includes a hypothesis that was refuted** |
| `reports/PHASE3_REPORT.md` | Re-ranking, generation, refusal, final results |
| `reports/ablation.md` | The A0–A6 comparison table |
| `reports/calibration.md` | How the refusal threshold was measured |
| `reports/failure_analysis.md` | Every remaining failure, diagnosed |

---

## Security note

`.env` is gitignored and was verified as untracked. If you ever suspect a key was
committed, rotate it at [console.groq.com](https://console.groq.com) — removing
the file from a later commit does **not** remove it from git history.
