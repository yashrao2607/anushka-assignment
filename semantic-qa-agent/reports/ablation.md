# Ablation Matrix — Phase 2 (rows A0–A4)

Each row is one command: `python -m src.cli evaluate --only A4`.
Rows A2/A3 (alternative embedding models) and A5–A9 (re-ranking, chunk-size
arms, HyDE) are filled in Phase 3.

| # | Configuration | P@3 | P@5 | R@10 | MRR@10 | nDCG@10 | p95 ms | Δ P@3 vs A0 |
|---|---|---|---|---|---|---|---|---|
| A0 | Keyword baseline (BM25 only) | 0.314 | 0.216 | 0.853 | 0.738 | 0.738 | 0.3 | — |
| A1 | Dense only (MiniLM bi-encoder) | 0.366 | 0.255 | 0.971 | 0.898 | 0.886 | 50.9 | +0.052 |
| A4 | Hybrid (dense + BM25, RRF fusion) | 0.379 | 0.255 | 0.980 | 0.852 | 0.861 | 10.7 | +0.065 |

## Reading the table

- **A0 (BM25 only)** is the keyword baseline the whole project exists to beat.
- **A1 (dense only)** isolates the contribution of semantic embeddings.
- **A4 (hybrid + RRF)** should beat both, because the two legs fail on
  different query types: dense loses on rare literal tokens, sparse loses on
  paraphrase. The per-category table in `eval_report.md` is where that
  complementarity is visible — check `exact_id` versus `paraphrase`.
