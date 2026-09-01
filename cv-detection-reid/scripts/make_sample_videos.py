"""Generate the synthetic scene set with exact ground truth.

**What this is and is not.** PRD 7.1 calls for custom captured footage annotated
in CVAT. That is the real dataset and this does not replace it: drop real `.mp4`
files into `data/raw_videos/` and the entire pipeline consumes them unchanged.

What this *does* solve is the project's stated bottleneck. Annotation is the
critical path (PRD R1, "Annotation is the true bottleneck"), and until labels
exist, nothing downstream -- splitter, metrics harness, tracker, ReID gallery,
occlusion recovery -- can be exercised or tested at all. So the scenes here
composite **real object crops** (cut from photographs with the COCO-pretrained
detector) onto procedural backgrounds along known trajectories. That buys two
things a hand-annotated clip cannot:

  * **Pixel-exact ground truth, including identities and visibility.** Track
    ids and per-frame occlusion fractions are known by construction, so
    MOTA/IDF1/HOTA (Phase 2.4) and the post-occlusion recovery rate M17
    (Phase 3.1) have a ground truth no human annotator could produce.
  * **Controlled difficulty.** Each scene isolates one axis from PRD 13.3 --
    lighting, occlusion, camera motion, crowding, blur -- so a slice measures
    the thing it is named after instead of whatever the footage happened to
    contain.

The objects are real photographed objects, so detector scores here are
meaningful; the *backgrounds* are synthetic, so absolute mAP is not comparable
to a road-scene benchmark and is never presented as such.

Usage:
    python scripts/make_sample_videos.py            # all scenes
    python scripts/make_sample_videos.py --scenes scene01 scene04
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.mot import MotRow, write_mot  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("scripts.make_videos")

FPS = 30
W, H = 960, 540
DURATION_S = 15

# Photographs the sprites are cut from. Both are Ultralytics' public sample
# assets, redistributed under their repository licence and used here only as a
# source of real object pixels.
SPRITE_SOURCES = (
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
)


# ---------------------------------------------------------------------------
# Sprites: cut real objects out of real photographs
# ---------------------------------------------------------------------------


def build_sprites(cfg, force: bool = False) -> dict[str, list[np.ndarray]]:
    """Extract per-class object crops, caching them under `data/sprites/`."""
    sprite_dir = cfg.path("sprites_dir")
    sprite_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[str, list[np.ndarray]] = {}
    if not force:
        for cls_dir in sorted(p for p in sprite_dir.iterdir() if p.is_dir()):
            imgs = [cv2.imread(str(p)) for p in sorted(cls_dir.glob("*.png"))]
            imgs = [i for i in imgs if i is not None and i.size]
            if imgs:
                cached[cls_dir.name] = imgs
        if cached:
            log.info(f"using cached sprites: { {k: len(v) for k, v in cached.items()} }")
            return cached

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    wanted = {int(k): v for k, v in cfg.dataset.coco_id_map.items()}
    out: dict[str, list[np.ndarray]] = {}

    for url in SPRITE_SOURCES:
        try:
            results = model.predict(url, conf=0.45, verbose=False)
        except Exception as exc:
            log.warning(f"could not fetch sprite source {url}: {exc}")
            continue
        for res in results:
            img = res.orig_img
            if res.boxes is None:
                continue
            for (x1, y1, x2, y2), k, c in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
                res.boxes.conf.cpu().numpy(),
            ):
                name = wanted.get(int(k))
                if name is None or c < 0.45:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
                crop = img[max(0, y1):y2, max(0, x1):x2]
                # Too small to survive rescaling into a scene without becoming
                # an unrecognisable smear -- which would make the "detector
                # missed it" signal meaningless.
                if crop.size == 0 or crop.shape[0] < 48 or crop.shape[1] < 24:
                    continue
                out.setdefault(name, []).append(crop)

    if not out:
        raise RuntimeError(
            "no sprites could be extracted (no network?). Place real videos in "
            "data/raw_videos/ and skip this script, or add PNG crops under data/sprites/<class>/."
        )

    for name, crops in out.items():
        d = sprite_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(crops):
            cv2.imwrite(str(d / f"{name}_{i:02d}.png"), crop)
    log.info(f"extracted sprites: { {k: len(v) for k, v in out.items()} }")
    return out


# ---------------------------------------------------------------------------
# Scene description
# ---------------------------------------------------------------------------


@dataclass
class Actor:
    track_id: int
    cls_name: str
    sprite: np.ndarray
    x: float
    y: float
    vx: float
    vy: float
    height_px: float
    scale_rate: float = 0.0     # px of height gained per frame (approach/recede)

    def step(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.height_px = max(24.0, self.height_px + self.scale_rate)

    def box(self) -> tuple[int, int, int, int]:
        sh, sw = self.sprite.shape[:2]
        h = int(round(self.height_px))
        w = max(8, int(round(h * sw / sh)))
        return int(round(self.x)), int(round(self.y)), w, h


@dataclass
class Scene:
    name: str                  # e.g. scene04_camA -> scene_id "scene04"
    lighting: str = "day"      # day | dusk | night
    n_actors: int = 4
    occluder: bool = False
    camera_pan: float = 0.0    # px/frame of horizontal ego-motion
    motion_blur: int = 0       # blur kernel size, 0 = off
    noise: float = 0.0         # gaussian sensor noise sigma
    seed: int = 0
    classes: tuple[str, ...] = field(default_factory=lambda: ("person", "bus"))


SCENES: tuple[Scene, ...] = (
    Scene("scene01_camA", lighting="day", n_actors=4, seed=1),
    Scene("scene02_camA", lighting="dusk", n_actors=4, seed=2),
    Scene("scene03_camA", lighting="night", n_actors=3, noise=6.0, seed=3),
    Scene("scene04_camA", lighting="day", n_actors=4, occluder=True, seed=4),
    Scene("scene05_camA", lighting="day", n_actors=4, camera_pan=1.6, seed=5),
    Scene("scene06_camA", lighting="day", n_actors=4, seed=6),
    Scene("scene06_camB", lighting="day", n_actors=4, camera_pan=-0.8, seed=6),
    Scene("scene07_camA", lighting="day", n_actors=11, seed=7),
    Scene("scene08_camA", lighting="day", n_actors=4, motion_blur=9, seed=8),
)

LIGHTING_GAIN = {"day": 1.0, "dusk": 0.45, "night": 0.16}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def make_background(rng: np.random.Generator) -> np.ndarray:
    """A road-ish backdrop: sky gradient, asphalt, lane markings, texture."""
    bg = np.zeros((H, W, 3), np.uint8)
    horizon = int(H * 0.42)
    for y in range(horizon):
        t = y / max(1, horizon)
        bg[y, :] = (int(180 - 40 * t), int(150 - 30 * t), int(120 - 20 * t))
    for y in range(horizon, H):
        t = (y - horizon) / max(1, H - horizon)
        bg[y, :] = (int(70 + 40 * t), int(70 + 40 * t), int(72 + 40 * t))

    cv2.rectangle(bg, (0, horizon - 26), (W, horizon), (95, 110, 95), -1)
    for x in range(0, W, 90):
        cv2.line(bg, (x, H - 60), (x + 45, H - 60), (215, 215, 215), 4)
    cv2.line(bg, (0, horizon + 8), (W, horizon + 8), (200, 200, 200), 2)
    for x in range(60, W, 240):
        cv2.rectangle(bg, (x, horizon - 90), (x + 26, horizon), (110, 100, 92), -1)

    # Texture stops the background from being a flat colour, which would make
    # the blur and low-light slices trivially easy for the wrong reason.
    noise = rng.normal(0, 7, (H, W, 3))
    return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def build_actors(scene: Scene, sprites: dict[str, list[np.ndarray]], rng: np.random.Generator) -> list[Actor]:
    available = [c for c in scene.classes if sprites.get(c)] or [c for c in sprites if sprites[c]]
    actors: list[Actor] = []
    for i in range(scene.n_actors):
        cls_name = available[i % len(available)]
        pool = sprites[cls_name]
        sprite = pool[int(rng.integers(len(pool)))]
        height = float(rng.uniform(70, 150) if cls_name == "person" else rng.uniform(110, 190))
        left_to_right = bool(i % 2 == 0)
        speed = float(rng.uniform(1.8, 3.6)) * (1 if left_to_right else -1)
        actors.append(
            Actor(
                track_id=i + 1,
                cls_name=cls_name,
                sprite=sprite,
                x=float(rng.uniform(-120, 120) if left_to_right else W - rng.uniform(-120, 120)),
                y=float(rng.uniform(H * 0.42, H * 0.80)),
                vx=speed,
                vy=float(rng.uniform(-0.25, 0.25)),
                height_px=height,
                scale_rate=float(rng.uniform(-0.06, 0.10)),
            )
        )
    return actors


def render_scene(scene: Scene, sprites: dict[str, list[np.ndarray]], cfg, out_dir: Path,
                 gt_dir: Path) -> tuple[Path, Path, int]:
    """Render one scene to `<name>.mp4` plus `<name>_gt.txt` (MOT format)."""
    rng = np.random.default_rng(cfg.seed + scene.seed)
    base_bg = make_background(rng)
    actors = build_actors(scene, sprites, rng)
    class_ids = {name: cfg.dataset.classes.index(name) for name in cfg.dataset.classes}

    out_path = out_dir / f"{scene.name}.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise IOError(f"cannot open writer for {out_path}")

    gain = LIGHTING_GAIN[scene.lighting]
    rows: list[MotRow] = []
    n_frames = FPS * DURATION_S

    for f in range(n_frames):
        pan = int(round(scene.camera_pan * f)) % W if scene.camera_pan else 0
        frame = np.roll(base_bg, -pan, axis=1) if pan else base_bg.copy()

        # An id-buffer gives exact visibility: after painting everything in
        # depth order, an actor's visible fraction is simply how many of its
        # pixels still carry its own id.
        id_buf = np.zeros((H, W), np.int32)
        boxes: dict[int, tuple[int, int, int, int]] = {}

        # Painter's algorithm, far objects (higher on screen) first.
        for actor in sorted(actors, key=lambda a: a.y):
            x, y, w, h = actor.box()
            x -= pan if scene.camera_pan else 0
            if x + w <= 0 or x >= W or y + h <= 0 or y >= H:
                continue
            sprite = cv2.resize(actor.sprite, (w, h), interpolation=cv2.INTER_AREA)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            sx1, sy1 = x1 - x, y1 - y
            if x2 <= x1 or y2 <= y1:
                continue
            frame[y1:y2, x1:x2] = sprite[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
            id_buf[y1:y2, x1:x2] = actor.track_id
            boxes[actor.track_id] = (x, y, w, h)

        if scene.occluder:
            # A pole the actors walk behind: this is what creates the full
            # occlusion events that M17 (post-occlusion recovery) is scored on.
            ox = int(W * 0.5) - 44
            cv2.rectangle(frame, (ox, 0), (ox + 88, H), (58, 58, 62), -1)
            cv2.rectangle(frame, (ox + 10, 0), (ox + 20, H), (78, 78, 84), -1)
            id_buf[:, ox:ox + 88] = -1

        for tid, (x, y, w, h) in boxes.items():
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            visible = int(np.count_nonzero(id_buf[y1:y2, x1:x2] == tid))
            visibility = visible / float(w * h)
            actor = next(a for a in actors if a.track_id == tid)
            rows.append(
                MotRow(
                    frame=f + 1, track_id=tid,
                    left=float(x), top=float(y), width=float(w), height=float(h),
                    conf=1.0, cls_id=class_ids[actor.cls_name],
                    visibility=round(visibility, 3),
                )
            )

        if scene.motion_blur:
            k = scene.motion_blur
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k       # horizontal blur = camera pan smear
            frame = cv2.filter2D(frame, -1, kernel)
        if gain != 1.0:
            frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        if scene.noise:
            frame = np.clip(
                frame.astype(np.float32) + rng.normal(0, scene.noise, frame.shape), 0, 255
            ).astype(np.uint8)

        writer.write(frame)
        for actor in actors:
            actor.step()
            x, _, w, _ = actor.box()
            if x > W + 160:
                actor.x = -w - 40.0
            elif x + w < -160:
                actor.x = float(W + 40)

    writer.release()
    gt_path = gt_dir / f"{scene.name}_gt.txt"
    write_mot(gt_path, rows)
    return out_path, gt_path, n_frames


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic scenes with exact ground truth")
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenes", nargs="*", default=None, help="scene name prefixes to render")
    ap.add_argument("--force-sprites", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.path("logs_dir") / "make_videos.jsonl")

    sprites = build_sprites(cfg, force=args.force_sprites)
    out_dir = cfg.path("raw_videos_dir")
    gt_dir = cfg.root / "data" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        s for s in SCENES
        if not args.scenes or any(s.name.startswith(p) for p in args.scenes)
    ]
    total = 0
    for scene in selected:
        video, gt, n = render_scene(scene, sprites, cfg, out_dir, gt_dir)
        total += n
        log.info(
            f"{scene.name}: {n} frames -> {video.name}, gt -> {gt.name} "
            f"[{scene.lighting}, {scene.n_actors} actors"
            f"{', occluder' if scene.occluder else ''}"
            f"{', pan' if scene.camera_pan else ''}"
            f"{', blur' if scene.motion_blur else ''}]"
        )
    print(f"\nRendered {len(selected)} scenes, {total} frames -> {out_dir}")
    print("Next:  python -m src.cli sample  &&  python -m src.cli labels  &&  python -m src.cli split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
