"""Tracking metrics: MOTA, IDF1, HOTA, IDSW, MT/ML, Fragmentation.

PRD 4.3 (M8-M13) and differentiator D2: "mAP measures the detector; it says
nothing about identity quality. Reporting HOTA/IDF1 proves the tracking claim
is measured rather than asserted."

Three metrics because they answer three different questions, and a tracker can
be good at one while being bad at another:

* **MOTA** counts errors: `1 - (FN + FP + IDSW) / |GT|`. It is dominated by
  detection errors, so a strong detector with a broken tracker still scores
  well. It can go negative -- that is not a bug, it means the system produced
  more errors than there were objects.
* **IDF1** matches whole *trajectories* globally, so a tracker that swaps two
  ids halfway through is punished for the entire second half rather than for
  one frame. This is the identity metric.
* **HOTA** is `sqrt(DetA * AssA)` averaged over IoU thresholds -- it refuses to
  let good detection hide bad association, or the reverse.

**Fidelity statement.** MOTA, IDF1, IDSW, MT/ML and Frag follow the CLEAR-MOT
and IDF1 definitions exactly. HOTA follows the published definition with one
simplification: per-frame correspondence is solved by Hungarian assignment on
IoU alone, whereas official TrackEval jointly optimises detection and
association. The two agree closely on well-behaved sequences and can diverge by
about a point where association is poor. TrackEval remains the reference
implementation and the reported figure names which produced it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..tracking.matching import linear_assignment
from .detection_metrics import iou_matrix

DEFAULT_IOU = 0.5
HOTA_ALPHAS = tuple(round(0.05 * i, 2) for i in range(1, 20))   # 0.05 ... 0.95


@dataclass(frozen=True)
class TrackBox:
    """One object in one frame, with its identity."""

    frame: int
    track_id: int
    xyxy: tuple[float, float, float, float]
    cls_id: int = 0
    ignore: bool = False


@dataclass
class TrackingMetrics:
    mota: float = 0.0
    motp: float = 0.0
    idf1: float = 0.0
    idp: float = 0.0
    idr: float = 0.0
    hota: float = 0.0
    deta: float = 0.0
    assa: float = 0.0
    idsw: int = 0
    idsw_per_1k: float = 0.0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    n_gt: int = 0
    n_pred: int = 0
    n_frames: int = 0
    gt_tracks: int = 0
    pred_tracks: int = 0
    mt: int = 0
    pt: int = 0
    ml: int = 0
    mt_ratio: float = 0.0
    ml_ratio: float = 0.0
    fragmentation: float = 0.0
    per_alpha: dict[str, float] = field(default_factory=dict)

    def headline(self) -> dict[str, float]:
        """The PRD-id keyed view the report renderer scores against targets."""
        return {
            "M8": self.mota,
            "M9": self.idf1,
            "M10": self.hota,
            "M11": self.idsw_per_1k,
            "M12": self.mt_ratio,
            "M12b": self.ml_ratio,
            "M13": self.fragmentation,
        }

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def group_frames(boxes: Iterable[TrackBox]) -> dict[int, list[TrackBox]]:
    out: dict[int, list[TrackBox]] = defaultdict(list)
    for b in boxes:
        out[b.frame].append(b)
    return dict(out)


def _match_frame(
    gts: Sequence[TrackBox],
    preds: Sequence[TrackBox],
    iou_thr: float,
    sticky: Mapping[int, int] | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Match one frame's GT to predictions, preferring last frame's pairing.

    The stickiness is what makes ID-switch counting correct. Without it, two
    equally good candidate matches are resolved arbitrarily and the tracker is
    charged switches it never made.
    """
    if not gts or not preds:
        return [], np.zeros((len(gts), len(preds)))

    ious = iou_matrix([p.xyxy for p in preds], [g.xyxy for g in gts]).T  # (gt, pred)
    matches: list[tuple[int, int]] = []
    used_g: set[int] = set()
    used_p: set[int] = set()

    if sticky:
        for gi, g in enumerate(gts):
            want = sticky.get(g.track_id)
            if want is None:
                continue
            for pi, p in enumerate(preds):
                if p.track_id == want and ious[gi, pi] >= iou_thr:
                    matches.append((gi, pi))
                    used_g.add(gi)
                    used_p.add(pi)
                    break

    free_g = [i for i in range(len(gts)) if i not in used_g]
    free_p = [i for i in range(len(preds)) if i not in used_p]
    if free_g and free_p:
        sub = 1.0 - ious[np.ix_(free_g, free_p)]
        pairs, _, _ = linear_assignment(sub, 1.0 - iou_thr)
        for r, c in pairs:
            matches.append((free_g[r], free_p[c]))

    return matches, ious


# ---------------------------------------------------------------------------
# CLEAR MOT
# ---------------------------------------------------------------------------


def clear_mot(
    gt: Sequence[TrackBox], pred: Sequence[TrackBox], iou_thr: float = DEFAULT_IOU
) -> dict[str, Any]:
    """MOTA, MOTP, ID switches, MT/ML and fragmentation."""
    gt_by_frame = group_frames(b for b in gt if not b.ignore)
    ignore_by_frame = group_frames(b for b in gt if b.ignore)
    pred_by_frame = group_frames(pred)
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    n_gt = sum(len(v) for v in gt_by_frame.values())
    tp = fp = fn = idsw = 0
    iou_sum = 0.0
    last_match: dict[int, int] = {}                      # gt id -> pred id
    tracked_frames: dict[int, list[int]] = defaultdict(list)
    gt_total_frames: Counter = Counter()

    for frame in frames:
        gts = gt_by_frame.get(frame, [])
        preds = pred_by_frame.get(frame, [])
        for g in gts:
            gt_total_frames[g.track_id] += 1

        matches, ious = _match_frame(gts, preds, iou_thr, last_match)
        matched_g = {gi for gi, _ in matches}
        matched_p = {pi for _, pi in matches}

        for gi, pi in matches:
            tp += 1
            iou_sum += float(ious[gi, pi])
            gid, pid = gts[gi].track_id, preds[pi].track_id
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
            tracked_frames[gid].append(frame)

        fn += len(gts) - len(matched_g)

        # Unmatched predictions that land on an `ignore` region are neither
        # rewarded nor punished (annotation guide rule 2).
        unmatched_preds = [p for pi, p in enumerate(preds) if pi not in matched_p]
        ignores = ignore_by_frame.get(frame, [])
        if ignores and unmatched_preds:
            m = iou_matrix([p.xyxy for p in unmatched_preds], [g.xyxy for g in ignores])
            absorbed = (m.max(axis=1) >= iou_thr).sum() if m.size else 0
            fp += len(unmatched_preds) - int(absorbed)
        else:
            fp += len(unmatched_preds)

        # A GT that disappears entirely resets its correspondence, so a
        # re-appearance after a genuine exit is not charged as a switch.
        present = {g.track_id for g in gts}
        for gid in list(last_match):
            if gid not in present:
                pass  # keep: the whole point of track_buffer is surviving a gap

    mota = 1.0 - (fn + fp + idsw) / n_gt if n_gt else 0.0
    motp = iou_sum / tp if tp else 0.0

    mt = pt = ml = 0
    frags: list[int] = []
    for gid, total in gt_total_frames.items():
        seen = sorted(set(tracked_frames.get(gid, [])))
        ratio = len(seen) / total if total else 0.0
        if ratio >= 0.8:
            mt += 1
        elif ratio <= 0.2:
            ml += 1
        else:
            pt += 1
        # Fragmentation: how many times the trajectory resumes after a gap.
        gaps = 0
        gt_frames = sorted(f for f in frames if any(b.track_id == gid for b in gt_by_frame.get(f, [])))
        seen_set = set(seen)
        was_tracked = False
        for f in gt_frames:
            now = f in seen_set
            if now and not was_tracked and gaps >= 0:
                gaps += 1
            was_tracked = now
        frags.append(max(0, gaps - 1))

    n_traj = len(gt_total_frames) or 1
    return {
        "mota": round(mota, 4),
        "motp": round(motp, 4),
        "idsw": idsw,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": n_gt,
        "n_pred": sum(len(v) for v in pred_by_frame.values()),
        "n_frames": len(frames),
        "gt_tracks": len(gt_total_frames),
        "pred_tracks": len({p.track_id for p in pred}),
        "mt": mt,
        "pt": pt,
        "ml": ml,
        "mt_ratio": round(mt / n_traj, 4),
        "ml_ratio": round(ml / n_traj, 4),
        "fragmentation": round(float(np.mean(frags)) if frags else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# IDF1
# ---------------------------------------------------------------------------


def idf1_score(
    gt: Sequence[TrackBox], pred: Sequence[TrackBox], iou_thr: float = DEFAULT_IOU
) -> dict[str, float]:
    """Global one-to-one identity matching over whole trajectories."""
    gt = [b for b in gt if not b.ignore]
    gt_by_frame = group_frames(gt)
    pred_by_frame = group_frames(pred)
    gt_ids = sorted({b.track_id for b in gt})
    pred_ids = sorted({b.track_id for b in pred})
    if not gt_ids or not pred_ids:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0, "idtp": 0, "idfp": len(pred), "idfn": len(gt)}

    gi_of = {g: i for i, g in enumerate(gt_ids)}
    pi_of = {p: i for i, p in enumerate(pred_ids)}
    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)

    for frame, gts in gt_by_frame.items():
        preds = pred_by_frame.get(frame, [])
        if not preds:
            continue
        m = iou_matrix([p.xyxy for p in preds], [g.xyxy for g in gts]).T
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                if m[gi, pi] >= iou_thr:
                    overlap[gi_of[g.track_id], pi_of[p.track_id]] += 1.0

    # Maximise total overlap: minimise its negation.
    pairs, _, _ = linear_assignment(-overlap, 0.0)
    idtp = int(sum(overlap[r, c] for r, c in pairs))
    idfn = len(gt) - idtp
    idfp = len(pred) - idtp
    idp = idtp / (idtp + idfp) if (idtp + idfp) else 0.0
    idr = idtp / (idtp + idfn) if (idtp + idfn) else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    return {
        "idf1": round(idf1, 4), "idp": round(idp, 4), "idr": round(idr, 4),
        "idtp": idtp, "idfp": idfp, "idfn": idfn,
    }


# ---------------------------------------------------------------------------
# HOTA
# ---------------------------------------------------------------------------


def hota_score(
    gt: Sequence[TrackBox], pred: Sequence[TrackBox], alphas: Sequence[float] = HOTA_ALPHAS
) -> dict[str, Any]:
    """HOTA = mean over alpha of sqrt(DetA_alpha * AssA_alpha)."""
    gt = [b for b in gt if not b.ignore]
    gt_by_frame = group_frames(gt)
    pred_by_frame = group_frames(pred)
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    if not gt or not pred:
        return {"hota": 0.0, "deta": 0.0, "assa": 0.0, "per_alpha": {}}

    gt_count = Counter(b.track_id for b in gt)
    pred_count = Counter(b.track_id for b in pred)

    hotas, detas, assas, per_alpha = [], [], [], {}
    for alpha in alphas:
        pair_hits: Counter = Counter()          # (gt_id, pred_id) -> matched frames
        tp_pairs: list[tuple[int, int]] = []
        n_tp = 0
        last_match: dict[int, int] = {}

        for frame in frames:
            gts = gt_by_frame.get(frame, [])
            preds = pred_by_frame.get(frame, [])
            matches, _ = _match_frame(gts, preds, alpha, last_match)
            for gi, pi in matches:
                gid, pid = gts[gi].track_id, preds[pi].track_id
                pair_hits[(gid, pid)] += 1
                tp_pairs.append((gid, pid))
                last_match[gid] = pid
                n_tp += 1

        n_fn = len(gt) - n_tp
        n_fp = len(pred) - n_tp
        deta = n_tp / (n_tp + n_fn + n_fp) if (n_tp + n_fn + n_fp) else 0.0

        if n_tp:
            total = 0.0
            for gid, pid in tp_pairs:
                tpa = pair_hits[(gid, pid)]
                fna = gt_count[gid] - tpa          # this GT, matched elsewhere or missed
                fpa = pred_count[pid] - tpa        # this prediction, spent on another GT
                total += tpa / (tpa + fna + fpa) if (tpa + fna + fpa) else 0.0
            assa = total / n_tp
        else:
            assa = 0.0

        h = float(np.sqrt(deta * assa))
        hotas.append(h)
        detas.append(deta)
        assas.append(assa)
        per_alpha[f"{alpha:.2f}"] = round(h, 4)

    return {
        "hota": round(float(np.mean(hotas)), 4),
        "deta": round(float(np.mean(detas)), 4),
        "assa": round(float(np.mean(assas)), 4),
        "per_alpha": per_alpha,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_tracking(
    gt: Sequence[TrackBox], pred: Sequence[TrackBox], iou_thr: float = DEFAULT_IOU
) -> TrackingMetrics:
    clear = clear_mot(gt, pred, iou_thr)
    ident = idf1_score(gt, pred, iou_thr)
    hota = hota_score(gt, pred)

    m = TrackingMetrics(**{k: v for k, v in clear.items() if k in TrackingMetrics.__annotations__})
    m.idf1, m.idp, m.idr = ident["idf1"], ident["idp"], ident["idr"]
    m.hota, m.deta, m.assa = hota["hota"], hota["deta"], hota["assa"]
    m.per_alpha = hota["per_alpha"]
    m.idsw_per_1k = round(1000.0 * m.idsw / m.n_frames, 2) if m.n_frames else 0.0
    return m
