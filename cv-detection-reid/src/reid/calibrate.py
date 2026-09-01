"""Calibrate `tau_reid` -- the gallery match threshold.

PRD 9.4: "`tau_reid` is **calibrated empirically**, not guessed: sweep tau over
the annotated occlusion events and pick the value maximising post-occlusion
recovery F1. The calibration curve is a published figure."

Why a swept threshold rather than a sensible-looking constant: the two failure
modes pull in opposite directions and the cost of each is different.

* **Too tight** -- the gallery rejects a genuine return, a new id is issued,
  and the unique-object count inflates. This is the failure the system is being
  built to fix, so it is the *expensive* one.
* **Too loose** -- the gallery restores the *wrong* id, merging two objects
  into one identity. This corrupts every downstream trajectory statistic
  silently, and a merged identity is harder to spot in a report than a
  duplicated one.

F1 over the occlusion events balances the two, and the curve is published so a
reviewer can see the shape rather than trust the argmax. The sweep runs on the
**validation** split; using test events to pick the threshold and then
reporting test recovery would be selecting on the test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config
from ..data.mot import IGNORE_VISIBILITY, MotRow, read_mot
from ..eval.detection_metrics import iou_matrix
from ..eval.reid_metrics import DEFAULT_MIN_GAP, find_occlusion_events
from ..utils.logging import get_logger

log = get_logger("reid.calibrate")

DEFAULT_TAUS = tuple(round(0.05 * i, 2) for i in range(1, 21))     # 0.05 ... 1.00


@dataclass
class CalibrationPoint:
    tau: float
    true_positive: int = 0      # correct identity restored
    false_positive: int = 0     # wrong identity restored -- an identity merge
    false_negative: int = 0     # genuine return rejected -> a new id issued
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CalibrationResult:
    points: list[CalibrationPoint] = field(default_factory=list)
    best_tau: float = 0.35
    best_f1: float = 0.0
    n_pairs: int = 0
    n_events: int = 0
    backend: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_tau": self.best_tau,
            "best_f1": self.best_f1,
            "n_pairs": self.n_pairs,
            "n_events": self.n_events,
            "backend": self.backend,
            "source": self.source,
            "curve": [p.as_dict() for p in self.points],
        }


def _crop_embeddings_at(
    cap, extractor, rows: Sequence[MotRow], frame_no: int
) -> dict[int, np.ndarray]:
    """Embed every visible ground-truth box in one frame, keyed by track id."""
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 1))
    ok, frame = cap.read()
    if not ok or frame is None:
        return {}
    visible = [r for r in rows if r.frame == frame_no and r.visibility >= IGNORE_VISIBILITY]
    if not visible:
        return {}
    feats = extractor.extract(frame, [r.xyxy for r in visible])
    return {visible[i].track_id: emb for i, emb in feats.items()}


def calibrate_tau(
    cfg: Config,
    videos: Sequence[Path],
    extractor=None,
    taus: Sequence[float] = DEFAULT_TAUS,
    min_gap: int = DEFAULT_MIN_GAP,
) -> CalibrationResult:
    """Sweep tau over real occlusion events and pick the F1-optimal value.

    For every occlusion event the embedding of the object *before* it vanished
    is compared against every candidate identity present when it returns. That
    is exactly the decision the live gallery makes, so the threshold chosen
    here is the threshold the runtime needs -- not a proxy for it.
    """
    import cv2

    if extractor is None:
        from .extractor import ReidExtractor

        extractor = ReidExtractor(cfg)

    pairs: list[tuple[float, bool]] = []      # (distance, is_same_identity)
    n_events = 0

    for video in videos:
        gt_file = cfg.root / "data" / "gt" / f"{Path(video).stem}_gt.txt"
        if not gt_file.exists():
            log.warning(f"no ground truth for {Path(video).name}; skipped",
                        extra={"event": "calib_no_gt"})
            continue
        rows = read_mot(gt_file)
        events = find_occlusion_events(rows, min_gap)
        if not events:
            continue

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        try:
            for ev in events:
                before = _crop_embeddings_at(cap, extractor, rows, ev.before_frame)
                after = _crop_embeddings_at(cap, extractor, rows, ev.after_frame)
                query = before.get(ev.gt_track_id)
                if query is None or not after:
                    continue
                n_events += 1
                q = np.asarray(query, dtype=np.float32).reshape(-1)
                q /= max(1e-12, float(np.linalg.norm(q)))
                for cand_id, cand in after.items():
                    c = np.asarray(cand, dtype=np.float32).reshape(-1)
                    c /= max(1e-12, float(np.linalg.norm(c)))
                    pairs.append((float(1.0 - c @ q), cand_id == ev.gt_track_id))
        finally:
            cap.release()

    result = CalibrationResult(
        n_pairs=len(pairs), n_events=n_events,
        backend=extractor.describe().get("reid_backend", "unknown"),
        source=", ".join(Path(v).name for v in videos),
    )
    if not pairs:
        log.warning(
            "no occlusion events with usable crops -- tau left at the config default",
            extra={"event": "calib_empty"},
        )
        result.best_tau = cfg.reid.gallery.threshold
        return result

    distances = np.array([d for d, _ in pairs])
    same = np.array([s for _, s in pairs], dtype=bool)

    for tau in taus:
        accepted = distances < tau
        tp = int(np.sum(accepted & same))
        fp = int(np.sum(accepted & ~same))
        fn = int(np.sum(~accepted & same))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result.points.append(
            CalibrationPoint(
                tau=float(tau), true_positive=tp, false_positive=fp, false_negative=fn,
                precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
            )
        )

    best = max(result.points, key=lambda p: (p.f1, -p.tau))
    result.best_tau = best.tau
    result.best_f1 = best.f1
    log.info(
        f"tau_reid calibrated to {best.tau} (F1 {best.f1:.3f}) over {n_events} occlusion "
        f"events / {len(pairs)} candidate pairs, backend {result.backend}",
        extra={"event": "calib_done", "tau": best.tau, "f1": best.f1},
    )
    return result


def plot_calibration(result: CalibrationResult, path: Path) -> Path | None:
    """Publish the curve (PRD 9.4 requires the figure, not just the number)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable; calibration figure skipped")
        return None

    if not result.points:
        return None
    taus = [p.tau for p in result.points]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(taus, [p.f1 for p in result.points], marker="o", label="F1")
    ax.plot(taus, [p.precision for p in result.points], marker=".", label="precision")
    ax.plot(taus, [p.recall for p in result.points], marker=".", label="recall")
    ax.axvline(result.best_tau, linestyle="--", color="crimson",
               label=f"chosen tau = {result.best_tau}")
    ax.set_xlabel("tau_reid (cosine distance)")
    ax.set_ylabel("score")
    ax.set_title(f"ReID gallery threshold calibration ({result.backend}, "
                 f"{result.n_events} occlusion events)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
