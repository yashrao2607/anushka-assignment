"""Build YOLO label files for sampled frames from MOT-format ground truth.

This is the seam where *any* annotation source plugs in. Today it reads the
MOT files that either the synthetic generator or a CVAT/Roboflow MOT export
produces; the rest of the pipeline never learns where labels came from.

Only the frames the sampler actually kept get labels -- writing labels for all
30 fps would put 15× more files on disk than the dataset contains and quietly
break the 1:1 image↔label invariant the validator checks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..config import Config
from ..utils.logging import get_logger
from .manifest import ManifestRow
from .mot import frame_index, group_by_frame, read_mot, to_yolo_lines

log = get_logger("data.labels")


@dataclass
class LabelBuildReport:
    videos: int = 0
    frames_written: int = 0
    frames_without_gt: int = 0
    boxes_written: int = 0
    boxes_ignored: int = 0        # visibility below the ignore threshold
    class_counts: Counter = field(default_factory=Counter)
    missing_gt_files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "videos": self.videos,
            "frames_written": self.frames_written,
            "frames_without_gt": self.frames_without_gt,
            "boxes_written": self.boxes_written,
            "boxes_ignored": self.boxes_ignored,
            "class_counts": dict(self.class_counts),
            "missing_gt_files": self.missing_gt_files,
        }


def gt_path_for(cfg: Config, source_video: str) -> Path:
    return cfg.root / "data" / "gt" / f"{Path(source_video).stem}_gt.txt"


def build_labels(rows: Sequence[ManifestRow], cfg: Config) -> LabelBuildReport:
    """Write one `.txt` per sampled frame; returns an auditable report."""
    report = LabelBuildReport()
    by_video: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_video.setdefault(row.source_video, []).append(row)

    names = cfg.dataset.classes
    for source_video, video_rows in sorted(by_video.items()):
        gt_file = gt_path_for(cfg, source_video)
        if not gt_file.exists():
            report.missing_gt_files.append(source_video)
            log.warning(
                f"no ground truth for {source_video}; its frames stay unlabelled "
                "(annotate them, or they will be excluded from every metric)",
                extra={"event": "missing_gt", "video": source_video},
            )
            continue

        report.videos += 1
        frames = group_by_frame(read_mot(gt_file))
        for row in video_rows:
            mot_rows = frames.get(frame_index(row.frame_no), [])
            if not mot_rows:
                report.frames_without_gt += 1
            report.boxes_ignored += sum(1 for r in mot_rows if r.ignore)
            lines = to_yolo_lines(mot_rows, row.width, row.height)

            label_path = cfg.root / row.label_path
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

            report.frames_written += 1
            report.boxes_written += len(lines)
            for line in lines:
                cls_id = int(line.split()[0])
                if 0 <= cls_id < len(names):
                    report.class_counts[names[cls_id]] += 1
    return report
