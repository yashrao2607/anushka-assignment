"""Typed configuration loading and validation.

PRD principle #4: "Config over code." Every tunable lives in `config/default.yaml`
and is surfaced here as a typed object, so a bad value fails loudly at startup
rather than silently producing wrong chunks three stages later.

The config also produces a `fingerprint()` -- a stable hash of the settings that
actually affect output. Every artefact records it, so any number in any report
can be traced back to the exact configuration that produced it (PRD NFR-8).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


class ConfigError(ValueError):
    """Raised when the configuration is structurally or semantically invalid."""


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    storage_dir: str = "storage"
    reports_dir: str = "reports"
    logs_dir: str = "logs"


@dataclass(frozen=True)
class IngestConfig:
    supported_extensions: tuple[str, ...] = (
        ".pdf", ".docx", ".md", ".txt", ".html", ".htm", ".csv",
    )
    recursive: bool = True
    max_file_mb: int = 200
    scanned_page_char_threshold: int = 50


@dataclass(frozen=True)
class CleaningConfig:
    normalize_unicode: bool = True
    fix_ligatures: bool = True
    dedupe_hyphenation: bool = True
    collapse_whitespace: bool = True
    strip_boilerplate: bool = True
    boilerplate_page_ratio: float = 0.6
    strip_page_numbers: bool = True


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive"
    chunk_size_tokens: int = 512
    overlap_tokens: int = 64
    min_chunk_chars: int = 30
    max_non_alnum_ratio: float = 0.8
    prefix_with_heading: bool = True
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass(frozen=True)
class Config:
    project: str = "semantic-qa-agent"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    # Sections declared for later phases are carried through untouched so that
    # Phase 2/3 code can read them without this file needing to change.
    extra: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # -- derived paths ------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a configured relative path against the project root."""
        return (self.root / getattr(self.paths, key)).resolve()

    def fingerprint(self) -> str:
        """Stable hash of the settings that change ingestion output.

        Only output-affecting sections are hashed: changing the reports
        directory must not invalidate a chunk set.
        """
        payload = {
            "ingest": asdict(self.ingest),
            "cleaning": asdict(self.cleaning),
            "chunking": asdict(self.chunking),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _validate(cfg: Config) -> None:
    """Fail loudly on impossible settings (PRD principle #6)."""
    ch = cfg.chunking
    if ch.chunk_size_tokens <= 0:
        raise ConfigError("chunking.chunk_size_tokens must be positive")
    if ch.overlap_tokens < 0:
        raise ConfigError("chunking.overlap_tokens must be >= 0")
    if ch.overlap_tokens >= ch.chunk_size_tokens:
        raise ConfigError(
            f"chunking.overlap_tokens ({ch.overlap_tokens}) must be smaller than "
            f"chunk_size_tokens ({ch.chunk_size_tokens}) -- otherwise chunking "
            f"never advances and would loop forever."
        )
    if ch.strategy not in {"recursive", "sentence"}:
        raise ConfigError(f"unknown chunking.strategy: {ch.strategy!r}")
    if not 0.0 < cfg.cleaning.boilerplate_page_ratio <= 1.0:
        raise ConfigError("cleaning.boilerplate_page_ratio must be in (0, 1]")
    if not 0.0 < ch.max_non_alnum_ratio <= 1.0:
        raise ConfigError("chunking.max_non_alnum_ratio must be in (0, 1]")
    if not cfg.ingest.supported_extensions:
        raise ConfigError("ingest.supported_extensions must not be empty")
    for ext in cfg.ingest.supported_extensions:
        if not ext.startswith("."):
            raise ConfigError(f"extension {ext!r} must start with a dot")


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load, validate and return the typed configuration.

    `overrides` accepts dotted keys, e.g. ``chunking.chunk_size_tokens=256``,
    so experiments can be driven from the CLI without editing the YAML file.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    for dotted, value in overrides.items():
        if value is None:
            continue
        node = raw
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value

    known = {"project", "seed", "paths", "ingest", "cleaning", "chunking"}
    cfg = Config(
        project=raw.get("project", "semantic-qa-agent"),
        seed=int(raw.get("seed", 42)),
        paths=PathsConfig(**raw.get("paths", {})),
        ingest=IngestConfig(
            **{
                **raw.get("ingest", {}),
                "supported_extensions": tuple(
                    raw.get("ingest", {}).get(
                        "supported_extensions", IngestConfig.supported_extensions
                    )
                ),
            }
        ),
        cleaning=CleaningConfig(**raw.get("cleaning", {})),
        chunking=ChunkingConfig(
            **{
                **raw.get("chunking", {}),
                "separators": tuple(
                    raw.get("chunking", {}).get(
                        "separators", ChunkingConfig.separators
                    )
                ),
            }
        ),
        extra={k: v for k, v in raw.items() if k not in known},
        root=cfg_path.resolve().parents[1],
    )
    _validate(cfg)
    return cfg
