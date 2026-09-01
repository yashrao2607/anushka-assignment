# Claude Code — Session History
**Project:** `anushka assignment`
**Date:** 1 September 2026
**Sessions:** 3
**Scope:** Two assignment projects — a Semantic Search / QA system and a Computer Vision object-detection + ReID system.

> Sanitised summary. Absolute paths, machine identifiers and credentials removed.

---

## Session 1 — "Detailed PRDs for two projects"
`759a3130` · 04:26 – 05:52 (1h 26m) · ~280 assistant turns

### What was asked
1. Read the requirements screenshot and write a highly detailed PRD for **both** projects as two markdown files.
2. Break Project 1 into **3 phases × 4 parts** and execute Phase 1 fully.
3. Continue through Phase 3, working within Groq free-tier API limits during testing.
4. Run the app for a live check.
5. Fix a crash hit while using the UI.

### What was produced
- `PRD_Project_1_Semantic_Search_QA_System.md`
- `PRD_Project2_Computer_Vision_ObjectDetection_ReID.md`
- `semantic-qa-agent/` — retrieval + generation pipeline, Streamlit UI (`app.py`), CLI
- Phases 1–3 implemented end to end

### Notable engineering outcome — threading bug fix
**Symptom:** `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread` — raised on the *second* UI interaction, never in the CLI.

**Root cause:** Streamlit caches the `Answerer` across reruns via `@st.cache_resource`, but executes each rerun on a **new script-runner thread**. The embedding cache's SQLite connection was created on the first thread. The CLI never surfaced it because the CLI is single-threaded.

**Fix — two parts, both required:**
```python
self.conn = sqlite3.connect(str(path), check_same_thread=False)  # lift sqlite's thread check
self._lock = threading.RLock()                                   # ...and actually serialise access
```
`check_same_thread=False` alone only removes the guard; the lock is what makes concurrent access correct.

The bug was reproduced deterministically before any code was changed, so the fix targeted the confirmed cause rather than a guess.

---

## Session 2 — "PRD Project 2: Computer Vision, Object Detection & ReID"
`c1dc077b` · 05:29 – 06:33 (1h 4m) · ~263 assistant turns

### What was asked
Read the Project 2 PRD, break it into 3 phases × 3–4 parts, then execute Phase 1 onward.

### What was delivered
All three phases built and running locally in `cv-detection-reid/`, with **121 tests passing**.

### Measured results (CPU only, no GPU on this machine)

| Metric | Result |
|---|---|
| Scene-level leakage gate | **PASS** — 540 frames, 18 scenes, split 360 / 90 / 90 |
| M1 — mAP@0.5 (B0 zero-shot) | 0.41 — the domain-gap baseline fine-tuning must beat |
| τ_reid calibration | **0.30**, F1 0.543 across 41 real occlusion events (curve published) |
| M14 / M15 / M16 — Rank-1 / Rank-5 / mAP | 0.741 / 0.935 / 0.640 |
| M18 — cross-camera match rate | **1.000** (8/8 identities, 2 camera pairs) — PASS |

Still in flight at session end: `reid-eval`, `failures`, `export --verify`, `ablate`, `demo`.

### Test quality note
These are not smoke tests. Every expected value in `test_metrics.py` and `test_tracking_metrics.py` is hand-derived in the docstring directly above it. `tests/test_no_leakage.py` feeds the validator a **deliberately leaked manifest** to prove the guard actually fires rather than merely passing on clean input.

### How to reproduce
```bash
cd cv-detection-reid
python -m pytest tests/ -q      # 121 tests
python -m src.cli env           # device + library versions
python -m src.cli validate      # leakage gate + label integrity
```

---

## Session 3 — "Remote control and chat history"
`8bafaa55` · 04:45 – 06:33 · housekeeping

Not project work. Covered disabling Claude Code Remote Control, auditing which sessions were visible over it, and producing this history document.

**Finding worth recording:** 24 peer sessions were reachable over Remote Control, and most belonged to other people's machines (several distinct `*-macbook-pro` / `*-macbook-air` hosts under other usernames). The Claude account is shared across the team, so Remote Control exposed this project's sessions to every other connected machine on that account. Remote Control was switched off; separate per-person logins are the durable fix.

---

## Commit trail

```
7103699  docs: add ethics and responsible AI policy documentation
4b52662  feat: implement phase 3 CLI module for ReID threshold calibration and evaluation
cca8fed  feat: add CLI phase 3 commands for ReID evaluation and threshold calibration
d9f43c2  test: add comprehensive pipeline and tracker unit tests with ReID documentation
e001cbd  test: add unit tests for Kalman filter, association metrics, and tracker logic
```

## Deliverables

| Path | Contents |
|---|---|
| `PRD_Project_1_Semantic_Search_QA_System.md` | Project 1 PRD |
| `PRD_Project2_Computer_Vision_ObjectDetection_ReID.md` | Project 2 PRD |
| `PHASE_PLAN_Project2.md` | 3-phase × 3–4-part execution plan |
| `semantic-qa-agent/` | Retrieval + QA pipeline, Streamlit UI, CLI |
| `cv-detection-reid/` | Detection + tracking + ReID pipeline, CLI, 121 tests |
| `cv-detection-reid/reports/` | Cross-camera evaluation results (JSON + Markdown) |
