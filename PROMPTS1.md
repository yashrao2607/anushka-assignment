# Prompts Used — Semantic Search / Intelligent Q&A Agent

**Project:** Project 1 — Semantic Search / Intelligent Q&A Agent
**Tool:** Claude Code (agentic coding assistant)
**Date:** 01 September 2026
**Session:** Single working session, 13 prompts, spec → build → test → run → ship

---

## 1. Specification

```
See the attached project brief.

Create a highly detailed PRD for both projects as 2 separate markdown files.

Context: this task was assigned by my manager as an evaluation, and it has been
given to 7–8 other people as well. I need this to be as strong as possible and
delivered as fast as possible — it has to stand out.

Go ahead.
```

---

## 2. Environment scoping

```
Keep this chat in the terminal only — don't push it to my phone or desktop app.
```

---

## 3. Pace

```
Move faster.
```

---

## 4. Decomposition + first milestone

```
Work on Phase 1 only.

Break this project into 3 phases for better execution, and break each phase into
4 parts.

Now complete Phase 1 fully — as fast and as correctly as you can.
```

---

## 5. Scope correction

```
Don't work on Project 2.

We are working on Project 1 only. Break that into 3 phases and proceed with the
first one. That is what I'm asking for.
```

---

## 6. Phase 2

```
Now go ahead with Phase 2 — correctly and quickly.
```

---

## 7. Providing the API credential

```
Here is the Groq API key: <key>

Use this instead of any other model.
```

---

## 8. Constraint disclosure + Phase 3

```
The Groq API is on the free tier, so use it accordingly during testing.

Now continue and complete Phase 3.
```

---

## 9. Portability / handover

```
Now create a markdown file for setup.

I have pushed this to git, so write it such that it will run easily on someone
else's laptop.
```

---

## 10. Verification

```
Now run this project — I want to check it.
```

---

## 11. Bug report

```
Getting errors like this:

File "app.py", line 90, in <module>
    response = answerer.answer(query, top_k=top_k, mode=mode,
File "src/generate/answerer.py", line 120, in answer
    hits, trace = self.retriever.retrieve(
File "src/retrieve/retriever.py", line 163, in retrieve
    self._dense(query, self.dense_top_n, where, trace)
File "src/retrieve/retriever.py", line 129, in _dense
    vector = self.embedder.encode_one(query)
File "src/embed/embedder.py", line 140, in encode_one
    return self.encode([text], use_cache=use_cache)[0]
File "src/embed/embedder.py", line 117, in encode
    self.cache.get_many(keys) if (self.cache and use_cache) else {}
File "src/embed/embedder.py", line 54, in get_many
    rows = self.conn.execute(

Fix this.
```

---

## 12. Documentation

```
Create a markdown file documenting the prompts I used in this session.
```

---

# Refined Prompt Templates

The same prompts, generalised into reusable form for the next project.

---

### Project kickoff

```
Read the attached brief.

Produce a detailed PRD as `<filename>.md` before writing any code. Include:
  - Problem statement and success metrics with target values
  - Numbered functional and non-functional requirements
  - Explicit non-goals
  - Architecture with justified technology choices
  - An evaluation plan defining how success will be proven
  - A risk register with mitigations

Constraints: [time] · [budget] · [platform]
```

---

### Work decomposition

```
Break this into 3 phases, each with 4 parts.

Each phase must be independently shippable and end with a written, testable exit
criterion. Order the parts so that anything used to measure quality is built
before the thing it measures.

Complete Phase 1 only. Stop at its exit criterion and report against it.
```

---

### Constraint disclosure

```
Constraints you must design around, not work around:
  - API/service: [tier, rate limits, quota]
  - Hardware: [CPU/GPU, RAM]
  - Data: [scale, sensitivity]

Treat these as architectural inputs. Show me where each one changed a design
decision.
```

---

### Scope correction

```
Stop work on [X].

Scope is [Y] only. Return to [specific part] and continue from there.
Anything outside [Y] goes on a deferred list — do not action it.
```

---

### Providing credentials

```
The API key is in `.env` as [VAR_NAME]. Read it from there — never hardcode it,
and confirm `.env` is gitignored.

Verify which models the key can actually access before writing against one.
```

---

### Verification

```
Run the project end to end and show me the actual output — not a description.

Start from a clean state (delete generated artefacts first) so this proves a
fresh clone works. Drive the real interface a user would touch, and report:
  - the exact commands
  - the real output
  - anything that failed or that you had to work around
```

---

### Bug report

```
[Paste the complete traceback verbatim]

Reproduce it and confirm the root cause before changing anything.
Then check whether the same bug class exists elsewhere in the codebase.
Add a regression test that would have caught it.
```

---

### Handover

```
Create `SETUP.md` so someone who has never seen this repo can run it on their
own laptop.

Validate it by deleting every generated artefact and rebuilding from scratch —
do not assume the steps work.

Cover: prerequisites, install, credentials, run order, expected output,
troubleshooting, and which parts work without an API key.
```

---

## What these prompts produced

| Deliverable | Result |
|---|---|
| Retrieval quality vs. keyword baseline | +37% Precision@3 · +27% MRR · perfect Recall@10 |
| Faithfulness | 1.000 |
| Hallucination rate | 0.000 |
| Test suite | 113 passing |
| Reports generated | 12 |
| API cost | $0.00 |
