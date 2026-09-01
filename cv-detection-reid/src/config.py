"""Typed configuration loading and validation.

PRD 9.7: "All hyperparameters live in `configs/*.yaml`." Every tunable is
surfaced here as a frozen, typed object so a bad value fails loudly at startup
rather than silently producing a wrong metric three stages later.

The config also produces a `fingerprint()` -- a stable hash over the settings
that actually affect output. Every artefact records it, so any number in any
report can be traced back to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when the configuration is structurally or semantically invalid."""


@dataclass(frozen=True)
class PathsConfig:
    raw_videos_dir: str = "data/raw_videos"
    frames_dir: str = "data/frames"
    labels_dir: str = "data/labels"
    splits_dir: str = "data/splits"
    sprites_dir: str = "data/sprites"
    manifest: str = "data/manifest.csv"
    reports_dir: str = "reports"
    runs_dir: str = "runs"
    logs_dir: str = "logs"


@dataclass(frozen=True)
class DatasetConfig:
    classes: tuple[str, ...] = ("person", "car", "motorcycle", "bus", "truck", "bicycle")
    coco_id_map: dict[int, str] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def class_id(self, name: str) -> int:
        try:
            return self.classes.index(name)
        except ValueError as exc:  # pragma: no cover - guarded by validation
            raise ConfigError(f"unknown class {name!r}") from exc


@dataclass(frozen=True)
class SamplingConfig:
    target_fps: float = 2.0
    max_frames_per_video: int = 400
    image_format: str = ".jpg"
    jpeg_quality: int = 92
    min_blur_score: float = 8.0


@dataclass(frozen=True)
class AttributesConfig:
    lighting_night_below: float = 60.0
    lighting_dusk_below: float = 110.0
    blur_hard_below: float = 60.0
    crowded_object_count: int = 15


@dataclass(frozen=True)
class SplitsConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    mode: str = "by_video"

    @property
    def ratios(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class DetectionConfig:
    weights: str = "yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    device: str = "auto"
    half: bool = False
    frame_skip: int = 1


@dataclass(frozen=True)
class EvalConfig:
    iou_thresholds: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
    primary_iou: float = 0.5
    recall_points: int = 101
    small_area_max: float = 1024.0
    medium_area_max: float = 9216.0
    operating_conf: float = 0.25


@dataclass(frozen=True)
class TrackingConfig:
    tracker_type: str = "botsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_hits: int = 3
    gmc_method: str = "sparseOptFlow"
    appearance_weight: float = 0.5


@dataclass(frozen=True)
class ReidGalleryConfig:
    threshold: float = 0.35
    ttl_frames: int = 300
    class_gated: bool = True


@dataclass(frozen=True)
class ReidConfig:
    enabled: bool = True
    model: str = "osnet_x0_25"
    embedding_dim: int = 512
    crop_size: tuple[int, int] = (256, 128)
    min_crop_wh: tuple[int, int] = (16, 32)
    embedding_ema: float = 0.9
    gallery: ReidGalleryConfig = field(default_factory=ReidGalleryConfig)


@dataclass(frozen=True)
class Config:
    project: str = "cv-detection-reid"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    attributes: AttributesConfig = field(default_factory=AttributesConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reid: ReidConfig = field(default_factory=ReidConfig)
    root: Path = REPO_ROOT

    # -- path helpers ------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a `paths.*` entry against the repository root."""
        value = getattr(self.paths, key)
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def fingerprint(self) -> str:
        """Stable 12-char hash of the output-affecting settings."""
        payload = {
            "seed": self.seed,
            "dataset": {"classes": list(self.dataset.classes)},
            "sampling": asdict(self.sampling),
            "attributes": asdict(self.attributes),
            "splits": asdict(self.splits),
            "detection": asdict(self.detection),
            "eval": asdict(self.eval),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------

def _sub(cls: type, data: dict[str, Any] | None, name: str) -> Any:
    """Build a frozen dataclass from a mapping, rejecting unknown keys.

    Unknown keys are an error, not a warning: a typo'd knob that is silently
    ignored is exactly how a config-driven project ends up reporting numbers
    that nobody can reproduce.
    """
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"section '{name}' must be a mapping, got {type(data).__name__}")
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in '{name}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        current = cls.__dataclass_fields__[key]
        if isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
        del current
    return cls(**kwargs)


def _validate(cfg: Config) -> None:
    d = cfg.dataset
    if not d.classes:
        raise ConfigError("dataset.classes must not be empty")
    if len(set(d.classes)) != len(d.classes):
        raise ConfigError(f"dataset.classes contains duplicates: {d.classes}")
    for coco_id, name in d.coco_id_map.items():
        if name not in d.classes:
            raise ConfigError(
                f"dataset.coco_id_map maps COCO id {coco_id} to unknown class {name!r}"
            )

    s = cfg.sampling
    if s.target_fps <= 0:
        raise ConfigError("sampling.target_fps must be > 0")
    if s.max_frames_per_video <= 0:
        raise ConfigError("sampling.max_frames_per_video must be > 0")
    if not 1 <= s.jpeg_quality <= 100:
        raise ConfigError("sampling.jpeg_quality must be in [1, 100]")

    a = cfg.attributes
    if not 0 <= a.lighting_night_below < a.lighting_dusk_below <= 255:
        raise ConfigError(
            "attributes: require 0 <= lighting_night_below < lighting_dusk_below <= 255"
        )

    sp = cfg.splits
    total = sp.train + sp.val + sp.test
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"splits must sum to 1.0, got {total:.4f}")
    if min(sp.ratios.values()) <= 0:
        raise ConfigError("every split ratio must be > 0")
    if sp.mode not in {"by_video", "random"}:
        raise ConfigError(f"splits.mode must be 'by_video' or 'random', got {sp.mode!r}")

    det = cfg.detection
    if not 0.0 < det.conf < 1.0:
        raise ConfigError("detection.conf must be in (0, 1)")
    if not 0.0 < det.iou < 1.0:
        raise ConfigError("detection.iou must be in (0, 1)")
    if det.imgsz % 32 != 0:
        raise ConfigError(f"detection.imgsz must be a multiple of 32, got {det.imgsz}")
    if det.device not in {"auto", "cpu", "cuda", "mps"} and not det.device.isdigit():
        raise ConfigError(f"detection.device must be auto|cpu|cuda|mps|<gpu index>, got {det.device!r}")
    if det.frame_skip < 1:
        raise ConfigError("detection.frame_skip must be >= 1")

    ev = cfg.eval
    if not ev.iou_thresholds:
        raise ConfigError("eval.iou_thresholds must not be empty")
    if any(not 0.0 < t < 1.0 for t in ev.iou_thresholds):
        raise ConfigError("every eval.iou_threshold must be in (0, 1)")
    if ev.primary_iou not in ev.iou_thresholds:
        raise ConfigError(
            f"eval.primary_iou {ev.primary_iou} must appear in eval.iou_thresholds"
        )
    if ev.small_area_max >= ev.medium_area_max:
        raise ConfigError("eval.small_area_max must be < eval.medium_area_max")

    tr = cfg.tracking
    if tr.tracker_type not in {"botsort", "bytetrack", "iou"}:
        raise ConfigError(f"tracking.tracker_type must be botsort|bytetrack|iou, got {tr.tracker_type!r}")
    if tr.track_low_thresh >= tr.track_high_thresh:
        raise ConfigError("tracking.track_low_thresh must be < track_high_thresh")
    if tr.track_buffer < 1:
        raise ConfigError("tracking.track_buffer must be >= 1")
    if not 0.0 <= tr.appearance_weight <= 1.0:
        raise ConfigError("tracking.appearance_weight must be in [0, 1]")

    rd = cfg.reid
    if not 0.0 < rd.gallery.threshold < 2.0:
        raise ConfigError("reid.gallery.threshold is a cosine distance in (0, 2)")
    if not 0.0 <= rd.embedding_ema <= 1.0:
        raise ConfigError("reid.embedding_ema must be in [0, 1]")
    if rd.gallery.ttl_frames < 1:
        raise ConfigError("reid.gallery.ttl_frames must be >= 1")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load, merge and validate the YAML configuration.

    `overrides` is a flat mapping of dotted keys, e.g. {"detection.conf": 0.4},
    which is how CLI flags reach the config without a second source of truth.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{cfg_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a top-level mapping")

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, key = dotted.partition(".")
        if not key:
            raw[section] = value
            continue
        node = raw.setdefault(section, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot override '{dotted}': '{section}' is not a mapping")
        node[key] = value

    reid_raw = dict(raw.get("reid") or {})
    gallery_raw = reid_raw.pop("gallery", None)
    reid = _sub(ReidConfig, reid_raw, "reid")
    if gallery_raw is not None:
        reid = ReidConfig(**{**asdict(reid), "gallery": _sub(ReidGalleryConfig, gallery_raw, "reid.gallery")})

    known_top = set(Config.__dataclass_fields__) - {"root"}
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ConfigError(f"unknown top-level config section(s): {sorted(unknown_top)}")

    cfg = Config(
        project=raw.get("project", "cv-detection-reid"),
        seed=int(raw.get("seed", 42)),
        paths=_sub(PathsConfig, raw.get("paths"), "paths"),
        dataset=_sub(DatasetConfig, raw.get("dataset"), "dataset"),
        sampling=_sub(SamplingConfig, raw.get("sampling"), "sampling"),
        attributes=_sub(AttributesConfig, raw.get("attributes"), "attributes"),
        splits=_sub(SplitsConfig, raw.get("splits"), "splits"),
        detection=_sub(DetectionConfig, raw.get("detection"), "detection"),
        eval=_sub(EvalConfig, raw.get("eval"), "eval"),
        tracking=_sub(TrackingConfig, raw.get("tracking"), "tracking"),
        reid=reid,
        root=cfg_path.resolve().parents[1],
    )
    _validate(cfg)
    return cfg
