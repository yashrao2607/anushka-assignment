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

    for name, phase in (("query", "Phase 2"), ("evaluate", "Phase 2"), ("serve", "Phase 3")):
        sub.add_parser(name, help=f"[{phase}] not yet implemented")
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


def cmd_not_ready(args, cfg) -> int:
    phase = {"query": "Phase 2", "evaluate": "Phase 2", "serve": "Phase 3"}[args.command]
    print(f"`{args.command}` is scheduled for {phase}. Phase 1 delivers: "
          f"ingest, stats, inspect.")
    return 2


def main(argv: list[str] | None = None) -> int:
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
        "query": cmd_not_ready, "evaluate": cmd_not_ready, "serve": cmd_not_ready,
    }
    try:
        return handlers[args.command](args, cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
