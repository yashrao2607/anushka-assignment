"""Command-line interface.

Phase 1 implements `ingest`, `stats` and `inspect`. `query`, `evaluate` and
`serve` are registered but report their target phase, so the CLI surface is
stable from day one and later phases only fill in behaviour.

    python -m src.cli ingest --path data/raw
    python -m src.cli stats
    python -m src.cli inspect --n 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .ingest.pipeline import load_chunks, run_ingestion
from .utils.logging import render_table, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-qa-agent",
        description="Semantic Search / Intelligent Q&A Agent -- Phase 1: ingestion",
    )
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="parse, clean and chunk a document folder")
    p_ingest.add_argument("--path", default=None, help="input directory (default: data/raw)")
    p_ingest.add_argument("--chunk-size", type=int, default=None, help="override chunk size in tokens")
    p_ingest.add_argument("--overlap", type=int, default=None, help="override overlap in tokens")
    p_ingest.add_argument("--force", action="store_true", help="rebuild from scratch")

    sub.add_parser("stats", help="show statistics for the current chunk set")

    p_inspect = sub.add_parser("inspect", help="print sample chunks for eyeball QA")
    p_inspect.add_argument("--n", type=int, default=3, help="number of chunks to show")
    p_inspect.add_argument("--doc", default=None, help="filter to one doc_id")

    p_index = sub.add_parser("index", help="embed chunks and build the dense + sparse indexes")
    p_index.add_argument("--backend", default=None, choices=["chroma", "numpy"])

    p_query = sub.add_parser("query", help="search the corpus")
    p_query.add_argument("text", help="the natural-language question")
    p_query.add_argument("-k", "--top-k", type=int, default=5)
    p_query.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    p_query.add_argument("--compare", action="store_true",
                         help="run every ablation configuration side by side")
    p_query.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    p_query.add_argument("--no-rerank", dest="rerank", action="store_false")

    p_eval = sub.add_parser("evaluate", help="run the golden set and write the reports")
    p_eval.add_argument("--only", default=None, help="run a single ablation arm, e.g. A4")

    p_ask = sub.add_parser("ask", help="ask a question and get a cited answer")
    p_ask.add_argument("text", help="the question")
    p_ask.add_argument("-k", "--top-k", type=int, default=5)
    p_ask.add_argument("--threshold", type=float, default=None,
                       help="override the refusal threshold")
    p_ask.add_argument("--no-llm", action="store_true",
                       help="retrieval only -- makes zero API calls")

    sub.add_parser("calibrate", help="calibrate the refusal threshold (0 API calls)")

    p_judge = sub.add_parser("judge", help="LLM-judged answer quality on a subset")
    p_judge.add_argument("--n", type=int, default=12,
                         help="stratified sample size (keeps free-tier usage low)")
    p_judge.add_argument("--max-calls", type=int, default=60,
                         help="hard cap on live API calls")

    sub.add_parser("serve", help="print the Streamlit launch command")
    return parser


def cmd_ingest(args, cfg) -> int:
    log = setup_logging(cfg.path("logs_dir"), args.verbose)
    log.info("=" * 62)
    log.info("INGESTION RUN -- %s", cfg.project)
    log.info("=" * 62)
    chunks, stats = run_ingestion(
        cfg, Path(args.path) if args.path else None, force=args.force
    )
    print()
    print(render_table(stats.as_rows(), "INGESTION SUMMARY"))

    if chunks:
        lengths = [c.token_count for c in chunks]
        print()
        print(render_table(
            [
                ("Chunks", len(chunks)),
                ("Tokens: min", min(lengths)),
                ("Tokens: median", int(statistics.median(lengths))),
                ("Tokens: mean", round(statistics.fmean(lengths), 1)),
                ("Tokens: max", max(lengths)),
                ("Chunks with a heading", sum(1 for c in chunks if c.section_heading)),
                ("Distinct documents", len({c.doc_id for c in chunks})),
                ("Config fingerprint", cfg.fingerprint()),
            ],
            "CHUNK QUALITY",
        ))
    if stats.files_failed:
        log.warning("%d file(s) failed to parse -- see reports/unparsed.csv",
                    stats.files_failed)
    return 0


def cmd_stats(args, cfg) -> int:
    setup_logging(cfg.path("logs_dir"), args.verbose)
    rows = load_chunks(cfg)
    if not rows:
        print("No chunks found. Run:  python -m src.cli ingest")
        return 1
    lengths = [r["token_count"] for r in rows]
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["doc_type"]] = by_type.get(r["doc_type"], 0) + 1
    print(render_table(
        [
            ("Total chunks", len(rows)),
            ("Distinct documents", len({r["doc_id"] for r in rows})),
            ("Total tokens", sum(lengths)),
            ("Median tokens/chunk", int(statistics.median(lengths))),
            ("Chunks with a heading", sum(1 for r in rows if r["section_heading"])),
            *[(f"Chunks from .{k}", v) for k, v in sorted(by_type.items())],
        ],
        "CHUNK SET STATISTICS",
    ))
    return 0


def cmd_inspect(args, cfg) -> int:
    setup_logging(cfg.path("logs_dir"), args.verbose)
    rows = load_chunks(cfg)
    if args.doc:
        rows = [r for r in rows if r["doc_id"] == args.doc]
    if not rows:
        print("No chunks to inspect. Run:  python -m src.cli ingest")
        return 1
    step = max(1, len(rows) // max(args.n, 1))
    for row in rows[::step][: args.n]:
        print("-" * 74)
        print(f"chunk_id : {row['chunk_id']}")
        print(f"source   : {Path(row['source_path']).name}  (page {row['page_no']})")
        print(f"heading  : {row['section_heading']}")
        print(f"span     : chars {row['char_start']}-{row['char_end']}  "
              f"| {row['token_count']} tokens")
        print(f"embedded : {row['embed_text'][:110]}...")
        print()
        print(row["text"][:420] + ("..." if len(row["text"]) > 420 else ""))
    print("-" * 74)
    return 0


def cmd_index(args, cfg) -> int:
    from .store.index import build_index

    log = setup_logging(cfg.path("logs_dir"), args.verbose)
    log.info("building indexes...")
    stats = build_index(cfg, backend=args.backend)
    print()
    print(render_table(list(stats.items()), "INDEX BUILD"))
    return 0


def _print_hits(hits, trace, header: str) -> None:
    print(f"\n{header}   ({trace.total_ms:.0f} ms  "
          f"embed {trace.embed_ms:.0f} / dense {trace.dense_ms:.0f} / "
          f"bm25 {trace.bm25_ms:.0f})")
    if not hits:
        print("  (no results)")
        return
    for h in hits:
        legs = []
        if h.dense_rank:
            legs.append(f"dense#{h.dense_rank} {h.dense_score:.3f}")
        if h.bm25_rank:
            legs.append(f"bm25#{h.bm25_rank} {h.bm25_score:.2f}")
        meta = h.metadata
        print(f"  {h.final_rank}. [{h.score:.4f}] {meta.get('doc_title')} "
              f"> {meta.get('section_heading')}   ({' | '.join(legs) or 'n/a'})")
        snippet = " ".join(h.text.split())[:150]
        print(f"     {snippet}...")


def cmd_query(args, cfg) -> int:
    from .store.index import load_retriever

    setup_logging(cfg.path("logs_dir"), args.verbose)
    retriever = load_retriever(cfg)
    print(f'\nQUERY: "{args.text}"')

    if args.compare:
        # These rows deliberately mirror the ablation arms in reports/ablation.md
        # exactly -- same modes, same rerank setting. A demo that quietly applied
        # re-ranking to every row would not be showing the reader the same system
        # the numbers describe.
        for mode, rerank, label in (
            ("sparse", False, "A0  BM25 keyword only"),
            ("dense", False, "A1  Dense semantic"),
            ("hybrid", False, "A4  Hybrid + RRF"),
            ("hybrid", True, "A5  Hybrid + cross-encoder re-rank  <- shipped"),
        ):
            hits, trace = retriever.retrieve(
                args.text, top_k=args.top_k, mode=mode, rerank=rerank
            )
            _print_hits(hits, trace, label)
    else:
        hits, trace = retriever.retrieve(
            args.text, top_k=args.top_k, mode=args.mode, rerank=args.rerank
        )
        _print_hits(hits, trace, f"mode={args.mode}")
    print()
    return 0


def cmd_evaluate(args, cfg) -> int:
    from .eval.runner import run_evaluation
    from .store.index import load_retriever

    setup_logging(cfg.path("logs_dir"), args.verbose)
    retriever = load_retriever(cfg)
    payload = run_evaluation(cfg, retriever, only=args.only)

    print()
    print(f"{'Config':<34} {'P@3':>7} {'P@5':>7} {'R@10':>7} "
          f"{'MRR':>7} {'nDCG':>7} {'p95ms':>7}")
    print("-" * 80)
    for r in payload["results"]:
        m = r["metrics"]
        print(f"{r['name'] + '  ' + r['description']:<34} "
              f"{m['precision@3']:>7.3f} {m['precision@5']:>7.3f} "
              f"{m['recall@10']:>7.3f} {m['mrr@10']:>7.3f} "
              f"{m['ndcg@10']:>7.3f} {r['latency_ms']['p95']:>7.1f}")
    print("-" * 80)
    print(f"reports/eval_report.md · reports/ablation.md · reports/eval_results.json")
    return 0


def cmd_ask(args, cfg) -> int:
    from .generate.answerer import build_answerer

    setup_logging(cfg.path("logs_dir"), args.verbose)
    answerer = build_answerer(cfg, threshold=args.threshold)
    response = answerer.answer(
        args.text, top_k=args.top_k, allow_llm=not args.no_llm
    )

    print(f'\nQ: {response.query}')
    print("=" * 74)
    print(response.answer)
    print("=" * 74)
    print(f"confidence {response.confidence:.3f} | threshold {answerer.threshold:.2f} "
          f"| {'REFUSED' if response.refused else 'answered'}"
          f"{' | cached' if response.from_cache else ''}")
    if response.refusal_reason:
        print(f"reason: {response.refusal_reason}  (no LLM call was made)")
    if response.citation_violations:
        print(f"citation violations: {response.citation_violations}")

    if response.citations:
        print("\nSOURCES")
        for c in response.citations:
            print(f"  [{c.marker}] {c.doc_title} > {c.section_heading} (p.{c.page_no})")
            print(f"      {c.quote[:150]}...")
    if response.latency_ms:
        print(f"\nlatency: {response.latency_ms}")
    print(f"api usage: {answerer.client.usage_summary()}")
    return 0


def cmd_calibrate(args, cfg) -> int:
    from .eval.calibrate import run_calibration
    from .generate.answerer import build_answerer

    setup_logging(cfg.path("logs_dir"), args.verbose)
    payload = run_calibration(cfg, build_answerer(cfg))
    best = payload["at_recommended"]
    print()
    print(render_table([
        ("Answerable questions", payload["n_answerable"]),
        ("Unanswerable questions", payload["n_unanswerable"]),
        ("Answerable conf (median)", payload["answerable_confidence"]["median"]),
        ("Unanswerable conf (median)", payload["unanswerable_confidence"]["median"]),
        ("Recommended threshold", payload["recommended_threshold"]),
        ("Answer correctness", best["answer_correctness"]),
        ("Refusal correctness", best["refusal_correctness"]),
        ("Balanced accuracy", best["balanced"]),
        ("LLM calls used", 0),
    ], "REFUSAL CALIBRATION"))
    print("\nreports/calibration.md · reports/calibration.json")
    return 0


def cmd_judge(args, cfg) -> int:
    from .eval.judge import run_judgement
    from .generate.answerer import build_answerer

    setup_logging(cfg.path("logs_dir"), args.verbose)
    answerer = build_answerer(cfg)
    answerer.client.max_calls = args.max_calls
    payload = run_judgement(cfg, answerer, sample_size=args.n)
    print()
    print(render_table(list(payload["summary"].items()), "ANSWER QUALITY"))
    print("\nreports/answer_quality.md")
    return 0


def cmd_serve(args, cfg) -> int:
    print("Launch the UI with:\n\n    streamlit run app.py\n")
    return 0


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode characters an LLM
    routinely emits (non-breaking hyphens, curly quotes, en dashes). Without this
    a perfectly good answer crashes on print. Reconfigure rather than strip, so
    the text stays intact wherever the terminal can render it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(
            args.config,
            **{
                "chunking.chunk_size_tokens": getattr(args, "chunk_size", None),
                "chunking.overlap_tokens": getattr(args, "overlap", None),
            },
        )
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    handlers = {
        "ingest": cmd_ingest, "stats": cmd_stats, "inspect": cmd_inspect,
        "index": cmd_index, "query": cmd_query, "evaluate": cmd_evaluate,
        "ask": cmd_ask, "calibrate": cmd_calibrate, "judge": cmd_judge,
        "serve": cmd_serve,
    }
    try:
        return handlers[args.command](args, cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
