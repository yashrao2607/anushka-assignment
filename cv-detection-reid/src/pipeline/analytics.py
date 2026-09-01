"""Trajectory analytics -- unique counts, dwell time, line crossing, UOCA.

PRD 1.2 states the business case plainly: "Detection alone answers *what is in
this frame?*... A detector without identity produces an unusable flood of
duplicate events; the identity layer is what converts raw pixels into
countable, auditable business events." This module is where that conversion
happens, and where the North Star metric is computed.

**UOCA** (PRD 4.1) is the North Star precisely because it cannot be gamed by
any single component: a perfect detector with a broken tracker inflates the
unique count, and a perfect tracker fed a blind detector deflates it. Only the
whole pipeline working produces the right number.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..eval.tracking_metrics import TrackBox


@dataclass
class TrackSummary:
    track_id: int
    cls_name: str
    first_frame: int
    last_frame: int
    n_frames: int

    def dwell_seconds(self, fps: float) -> float:
        return round((self.last_frame - self.first_frame + 1) / max(fps, 1e-6), 2)


@dataclass
class AnalyticsResult:
    unique_total: int = 0
    unique_by_class: dict[str, int] = field(default_factory=dict)
    tracks: list[TrackSummary] = field(default_factory=list)
    dwell_seconds: dict[int, float] = field(default_factory=dict)
    mean_dwell_by_class: dict[str, float] = field(default_factory=dict)
    line_crossings: dict[str, int] = field(default_factory=dict)
    min_track_frames: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "unique_total": self.unique_total,
            "unique_by_class": self.unique_by_class,
            "mean_dwell_by_class": self.mean_dwell_by_class,
            "line_crossings": self.line_crossings,
            "min_track_frames": self.min_track_frames,
            "n_tracks": len(self.tracks),
        }


def summarise_tracks(
    predictions: Sequence[TrackBox],
    class_names: Sequence[str],
    fps: float = 30.0,
    min_track_frames: int = 3,
) -> AnalyticsResult:
    """Collapse per-frame boxes into per-identity records.

    `min_track_frames` filters out one- and two-frame flickers before they are
    counted as unique objects. Without it the "unique count" reports detector
    noise, which is the very failure the identity layer is meant to remove --
    so the filter is applied and its value is reported, not hidden.
    """
    by_track: dict[int, list[TrackBox]] = defaultdict(list)
    for p in predictions:
        by_track[p.track_id].append(p)

    result = AnalyticsResult(min_track_frames=min_track_frames)
    per_class_dwell: dict[str, list[float]] = defaultdict(list)

    for track_id, boxes in sorted(by_track.items()):
        if len(boxes) < min_track_frames:
            continue
        boxes.sort(key=lambda b: b.frame)
        cls_id = max(set(b.cls_id for b in boxes), key=[b.cls_id for b in boxes].count)
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        summary = TrackSummary(
            track_id=track_id, cls_name=name,
            first_frame=boxes[0].frame, last_frame=boxes[-1].frame, n_frames=len(boxes),
        )
        result.tracks.append(summary)
        dwell = summary.dwell_seconds(fps)
        result.dwell_seconds[track_id] = dwell
        per_class_dwell[name].append(dwell)

    result.unique_total = len(result.tracks)
    counts: dict[str, int] = defaultdict(int)
    for t in result.tracks:
        counts[t.cls_name] += 1
    result.unique_by_class = dict(sorted(counts.items()))
    result.mean_dwell_by_class = {
        k: round(sum(v) / len(v), 2) for k, v in sorted(per_class_dwell.items())
    }
    return result


def count_line_crossings(
    predictions: Sequence[TrackBox],
    line: tuple[float, float, float, float],
    class_names: Sequence[str],
) -> dict[str, int]:
    """Count directional crossings of a virtual line by track centroid.

    Direction comes from the sign of the 2D cross product of the line vector
    with the object's position, so a track that crosses and comes back nets to
    zero rather than counting twice. Reported as `in` / `out` because "how many
    entered" is the question a supervisor actually asks.
    """
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1

    def side(cx: float, cy: float) -> int:
        cross = dx * (cy - y1) - dy * (cx - x1)
        return 1 if cross > 0 else (-1 if cross < 0 else 0)

    by_track: dict[int, list[TrackBox]] = defaultdict(list)
    for p in predictions:
        by_track[p.track_id].append(p)

    counts = {"in": 0, "out": 0}
    per_class: dict[str, int] = defaultdict(int)
    for track_id, boxes in by_track.items():
        boxes.sort(key=lambda b: b.frame)
        prev = None
        for b in boxes:
            bx1, by1, bx2, by2 = b.xyxy
            s = side((bx1 + bx2) / 2, (by1 + by2) / 2)
            if prev is not None and s != 0 and prev != 0 and s != prev:
                key = "in" if s > 0 else "out"
                counts[key] += 1
                name = class_names[b.cls_id] if 0 <= b.cls_id < len(class_names) else str(b.cls_id)
                per_class[f"{name}_{key}"] += 1
            if s != 0:
                prev = s
    counts.update(per_class)
    return counts


def uoca(reported: int, truth: int) -> float:
    """North Star (PRD 4.1): absolute percentage error of the unique count.

    Target is <= 8%. Returned as a percentage, so 0.0 is perfect.
    """
    if truth <= 0:
        return 0.0 if reported == 0 else 100.0
    return round(100.0 * abs(reported - truth) / truth, 2)


def ground_truth_unique_count(gt_rows: Iterable, min_frames: int = 3) -> int:
    """Unique identities in the ground truth, on the same terms as the prediction.

    The same `min_frames` filter is applied to both sides. Comparing a filtered
    prediction against an unfiltered truth would report a systematic deficit
    that is an artefact of the comparison, not of the system.
    """
    counts: dict[int, int] = defaultdict(int)
    for r in gt_rows:
        vis = getattr(r, "visibility", 1.0)
        if vis >= 0.30:
            counts[r.track_id] += 1
    return sum(1 for n in counts.values() if n >= min_frames)
