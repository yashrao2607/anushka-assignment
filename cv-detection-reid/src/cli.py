"""Command-line interface.

Phase 1 (data + detection baseline):
    python -m src.cli env
    python -m src.cli sample
    python -m src.cli labels
    python -m src.cli split
    python -m src.cli validate
    python -m src.cli eval-det --split test

Phase 2 (training + tracking):
    python -m src.cli train --model yolo11n.pt --epochs 60
    python -m src.cli track --video data/raw_videos/scene04_camA.mp4 --save
    python -m src.cli eval-track --tracker botsort
    python -m src.cli ablate

The surface is stable from day one; Phase 3 commands (`reid`, `demo`, `export`)
are registered and report their target phase rather than failing with an
unknown-command error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .utils.logging import get_logger, render_table, setup_logging

log = get_logger("cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cv-detection-reid",
        description="Object detection, tracking and re-identification pipeline",
    )
    p.add_argument("--config", default=None, help="path to a YAML config file")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="print the environment and resolved device")

    s = sub.add_parser("sample", help="sample frames from raw videos into the manifest")
    s.add_argument("--path", default=None, help="video directory (default: data/raw_videos)")
    s.add_argument("--force", action="store_true", help="re-write frames that already exist")

    sub.add_parser("labels", help="build YOLO labels for sampled frames from MOT ground truth")

    sp = sub.add_parser("split", help="assign scene-level train/val/test splits")
    sp.add_argument("--no-copy", action="store_true", help="write listings only, do not copy files")

    sub.add_parser("validate", help="validate labels, images and split integrity")

    ed = sub.add_parser("eval-det", help="detection metrics + difficulty slices")
    ed.add_argument("--split", default="test", choices=["train", "val", "test"])
    ed.add_argument("--weights", default=None, help="default: detection.weights (B0 baseline)")
    ed.add_argument("--limit", type=int, default=None)
    ed.add_argument("--label", default=None, help="row label for the report, e.g. B0")

    tr = sub.add_parser("train", help="fine-tune the detector")
    tr.add_argument("--model", default="yolo11n.pt")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--imgsz", type=int, default=640)
    tr.add_argument("--freeze", type=int, default=None, help="EXP-7: freeze N backbone layers")
    tr.add_argument("--name", default=None)

    tk = sub.add_parser("track", help="detect + track one video")
    tk.add_argument("--video", required=True)
    tk.add_argument("--tracker", default=None, choices=["iou", "bytetrack", "botsort"])
    tk.add_argument("--weights", default=None)
    tk.add_argument("--save", action="store_true", help="write annotated mp4 + results csv")
    tk.add_argument("--max-frames", type=int, default=None)
    tk.add_argument("--reid", action="store_true", help="appearance-fused association (Phase 3)")

    et = sub.add_parser("eval-track", help="MOTA / IDF1 / HOTA on the test scenes")
    et.add_argument("--tracker", default=None, choices=["iou", "bytetrack", "botsort"])
    et.add_argument("--weights", default=None)
    et.add_argument("--split", default="test", choices=["train", "val", "test"])
    et.add_argument("--max-frames", type=int, default=None)

    ab = sub.add_parser("ablate", help="fill the ablation matrix rows that exist today")
    ab.add_argument("--weights", default=None, help="fine-tuned weights for the B2+ rows")
    ab.add_argument("--max-frames", type=int, default=None)
    ab.add_argument("--split", default="test", choices=["train", "val", "test"])

    sub.add_parser("runs", help="list registered training runs")

    for name, phase in (("reid", "3.1"), ("demo", "3.2"), ("export", "3.3")):
        sub.add_parser(name, help=f"(Phase {phase}) not implemented yet")
    return p


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_env(cfg) -> int:
    from .utils.device import environment_report

    info = environment_report()
    print(render_table(["Key", "Value"], [[k, v] for k, v in info.items()]))
    print(f"\n  config fingerprint : {cfg.fingerprint()}")
    print(f"  classes            : {', '.join(cfg.dataset.classes)}")
    if not info["cuda_available"]:
        print(
            "\n  NOTE: no GPU. PRD risk R9 applies -- train on Colab/Kaggle and pull\n"
            "  best.pt back into runs/. CPU numbers are reported as the honest worst case."
        )
    return 0


def cmd_sample(cfg, args) -> int:
    from .data.manifest import write_manifest
    from .data.sampler import find_videos, sample_video

    directory = Path(args.path) if args.path else cfg.path("raw_videos_dir")
    videos = find_videos(directory)
    if not videos:
        print(
            f"no videos found in {directory}\n"
            "  Put .mp4 files there, or generate the synthetic scene set:\n"
            "    python scripts/make_sample_videos.py"
        )
        return 1

    rows, table = [], []
    for video in videos:
        res = sample_video(video, cfg, force=args.force)
        rows.extend(res.rows)
        table.append([res.source_video, res.scene_id, f"{res.fps:.1f}",
                      res.total_frames, res.kept, res.dropped_blur, res.unreadable])

    n = write_manifest(cfg.path("manifest"), rows)
    print(render_table(
        ["video", "scene", "fps", "frames", "kept", "dropped(blur)", "unreadable"], table))
    print(f"\n  {n} frames -> {cfg.path('manifest').relative_to(cfg.root)}")
    print("  next: python -m src.cli labels")
    return 0


def cmd_labels(cfg) -> int:
    from .data.labels_builder import build_labels
    from .data.manifest import read_manifest, write_manifest

    rows = read_manifest(cfg.path("manifest"))
    report = build_labels(rows, cfg)
    write_manifest(cfg.path("manifest"), rows)

    print(render_table(["Metric", "Value"], [
        ["videos with ground truth", report.videos],
        ["label files written", report.frames_written],
        ["frames with no GT rows", report.frames_without_gt],
        ["boxes written", report.boxes_written],
        ["boxes ignored (occluded > 70%)", report.boxes_ignored],
    ]))
    if report.class_counts:
        print("\n" + render_table(
            ["class", "instances"],
            sorted(report.class_counts.items(), key=lambda kv: -kv[1])))
    if report.missing_gt_files:
        print(f"\n  WARNING: no ground truth for: {', '.join(report.missing_gt_files)}")
    print("\n  next: python -m src.cli split")
    return 0


def cmd_split(cfg, args) -> int:
    from .data.manifest import read_manifest, write_manifest
    from .data.splitter import assert_no_leakage, materialise_splits, split_dataset
    from .data.validate_labels import validate_and_enrich

    rows = read_manifest(cfg.path("manifest"))
    validate_and_enrich(rows, cfg)          # n_objects/classes feed the split report
    report = split_dataset(rows, cfg)
    assert_no_leakage(rows)
    write_manifest(cfg.path("manifest"), rows)
    written = materialise_splits(rows, cfg, copy=not args.no_copy)

    print(render_table(
        ["split", "frames", "realised", "target", "scenes"],
        [[s, report.counts[s], f"{report.realised[s]:.3f}", f"{report.target[s]:.2f}",
          ", ".join(report.scenes.get(s, [])) or "-"] for s in ("train", "val", "test")]))
    print("\n" + render_table(
        ["class"] + list(("train", "val", "test")),
        [[c] + [report.class_counts[s].get(c, 0) for s in ("train", "val", "test")]
         for c in cfg.dataset.classes]))
    for w in report.warnings:
        print(f"\n  WARNING: {w}")
    print(f"\n  leakage check: PASS (no scene or video spans two splits)")
    print(f"  materialised: {written}")
    return 0


def cmd_validate(cfg) -> int:
    from .data.manifest import read_manifest
    from .data.splitter import LeakageError, assert_no_leakage
    from .data.validate_labels import check_images_readable, validate_and_enrich

    rows = read_manifest(cfg.path("manifest"))
    report = validate_and_enrich(rows, cfg)
    bad_images = check_images_readable(rows, cfg)

    try:
        assert_no_leakage(rows)
        leak = "PASS"
    except LeakageError as exc:
        leak = f"FAIL -- {exc}"

    print(render_table(["Check", "Result"], [
        ["label files checked", report.files_checked],
        ["label files missing", report.files_missing],
        ["empty label files (hard negatives)", report.files_empty],
        ["boxes valid / total", f"{report.boxes_valid} / {report.boxes_total}"],
        ["label issues", len(report.issues)],
        ["unreadable images", len(bad_images)],
        ["scene-level leakage", leak],
    ]))
    if report.class_counts:
        print("\n" + render_table(["class", "instances"],
                                  sorted(report.class_counts.items(), key=lambda kv: -kv[1])))
    if report.issues:
        print("\n" + render_table(
            ["code", "count"],
            sorted({i.code: sum(1 for j in report.issues if j.code == i.code)
                    for i in report.issues}.items(), key=lambda kv: -kv[1])))
    ok = report.ok and not bad_images and leak == "PASS"
    print(f"\n  overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_eval_det(cfg, args) -> int:
    from .eval.report import (DETECTION_TARGETS, confusion_table, git_sha, per_class_table,
                              slice_table, targets_table, write_json, write_markdown)
    from .eval.runner import run_detection_eval

    run = run_detection_eval(cfg, split=args.split, weights=args.weights, limit=args.limit)
    o = run.overall
    label = args.label or ("B0" if run.detector["coco_pretrained"] else "fine-tuned")

    headline = targets_table(run.headline(), DETECTION_TARGETS, markdown=False)
    print("\n" + headline)
    print("\n" + per_class_table(o.per_class_ap50, o.per_class_ap50_95, o.support, markdown=False))
    if run.slices:
        print("\n" + slice_table({k: v.as_dict() for k, v in run.slices.items()}, markdown=False))
    print("\n" + render_table(["Timing", "Value"],
                              [[k, v] for k, v in run.timing.items()]))

    sections = [
        (f"Headline metrics — configuration `{label}`, split `{args.split}`",
         targets_table(run.headline(), DETECTION_TARGETS, markdown=True)),
        ("Per-class AP (M6)", per_class_table(o.per_class_ap50, o.per_class_ap50_95, o.support)),
        ("Object-size bands (M7)", render_table(
            ["Band", "AP@0.5"], [[k, f"{v:.4f}"] for k, v in o.size_ap50.items()], markdown=True)),
        ("Difficulty slices (PRD §13.3)", slice_table({k: v.as_dict() for k, v in run.slices.items()})),
        ("Confusion matrix", confusion_table(o.confusion)),
        ("Counts", render_table(["Key", "Value"], [
            ["images", o.n_images], ["ground-truth boxes", o.n_gt], ["predictions", o.n_pred],
            ["TP / FP / FN", f"{o.tp} / {o.fp} / {o.fn}"]], markdown=True)),
        ("Throughput", render_table(["Key", "Value"],
                                    [[k, v] for k, v in run.timing.items()], markdown=True)),
    ]
    provenance = {
        "config fingerprint": cfg.fingerprint(),
        "git commit": git_sha(cfg.root),
        "split": args.split,
        **{k: v for k, v in run.detector.items()},
    }
    out = cfg.path("reports_dir") / "eval_report.md"
    write_markdown(out, f"Detection Evaluation — {label}", sections, provenance)
    write_json(cfg.path("reports_dir") / "eval_results.json", {
        "label": label, "split": args.split, "overall": o.as_dict(),
        "slices": {k: v.as_dict() for k, v in run.slices.items()},
        "detector": run.detector, "timing": run.timing, "provenance": provenance,
    })
    print(f"\n  wrote {out.relative_to(cfg.root)}")
    return 0


def cmd_train(cfg, args) -> int:
    from .models.train import TrainArgs, train_detector

    targs = TrainArgs(
        model=args.model, epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz, freeze=args.freeze,
    )
    record = train_detector(cfg, targs, run_name=args.name)
    print(render_table(["Key", "Value"], [
        ["run id", record.run_id], ["device", record.device],
        ["duration (s)", record.duration_s], ["best weights", record.best_weights or "n/a"],
        ["dataset hash", record.dataset_hash], ["git", record.git_sha],
    ]))
    if record.metrics:
        print("\n" + render_table(["metric", "value"],
                                  [[k, round(v, 4)] for k, v in record.metrics.items()]))
    return 0


def cmd_track(cfg, args) -> int:
    from .pipeline.track_video import track_video

    video = Path(args.video)
    save_dir = cfg.path("reports_dir") / "tracks"
    result = track_video(
        cfg, video, tracker_type=args.tracker, weights=args.weights,
        save_video=(save_dir / f"{video.stem}_{args.tracker or cfg.tracking.tracker_type}.mp4")
        if args.save else None,
        save_csv=(save_dir / f"{video.stem}_{args.tracker or cfg.tracking.tracker_type}.csv")
        if args.save else None,
        max_frames=args.max_frames, with_reid=args.reid,
    )
    print(render_table(["Key", "Value"], [[k, v] for k, v in result.summary().items()]))
    if args.save:
        print(f"\n  wrote {save_dir.relative_to(cfg.root)}/")
    return 0


def _test_videos(cfg, split: str) -> list[Path]:
    from .data.manifest import read_manifest

    rows = read_manifest(cfg.path("manifest"))
    names = sorted({r.source_video for r in rows if r.split == split})
    return [cfg.path("raw_videos_dir") / n for n in names]


def _eval_tracker_on(cfg, videos, tracker_type, weights, max_frames):
    from .eval.tracking_metrics import TrackBox, evaluate_tracking
    from .pipeline.track_video import load_gt_tracks, track_video

    all_gt: list[TrackBox] = []
    all_pred: list[TrackBox] = []
    per_video, offset = [], 0
    total_fps: list[float] = []

    for video in videos:
        gt_file = cfg.root / "data" / "gt" / f"{video.stem}_gt.txt"
        if not gt_file.exists():
            log.warning(f"no tracking ground truth for {video.name}; skipped",
                        extra={"event": "no_track_gt"})
            continue
        res = track_video(cfg, video, tracker_type=tracker_type, weights=weights,
                          max_frames=max_frames)
        gt = [b for b in load_gt_tracks(gt_file) if not max_frames or b.frame <= max_frames]

        # Offset frame ids and track ids so several sequences can be pooled into
        # one score without a track from clip A ever matching one from clip B.
        shift = offset
        span = max([b.frame for b in gt] + [b.frame for b in res.predictions] + [0])
        all_gt += [TrackBox(b.frame + shift, b.track_id + shift * 1000, b.xyxy, b.cls_id, b.ignore)
                   for b in gt]
        all_pred += [TrackBox(b.frame + shift, b.track_id + shift * 1000, b.xyxy, b.cls_id)
                     for b in res.predictions]
        offset += span + 10

        m = evaluate_tracking(gt, res.predictions)
        total_fps.append(res.fps)
        per_video.append((video.name, m, res))

    pooled = evaluate_tracking(all_gt, all_pred) if all_gt else None
    return pooled, per_video, (sum(total_fps) / len(total_fps) if total_fps else 0.0)


def cmd_eval_track(cfg, args) -> int:
    from .eval.report import (TRACKING_TARGETS, git_sha, targets_table, write_json,
                              write_markdown)

    videos = _test_videos(cfg, args.split)
    if not videos:
        print(f"no videos in split '{args.split}' -- run `python -m src.cli split` first")
        return 1

    name = args.tracker or cfg.tracking.tracker_type
    pooled, per_video, mean_fps = _eval_tracker_on(cfg, videos, name, args.weights, args.max_frames)
    if pooled is None:
        print("no sequence had tracking ground truth")
        return 1

    print("\n" + targets_table(pooled.headline(), TRACKING_TARGETS, markdown=False))
    print("\n" + render_table(
        ["video", "MOTA", "IDF1", "HOTA", "IDSW", "MT", "ML", "FPS"],
        [[v, f"{m.mota:.3f}", f"{m.idf1:.3f}", f"{m.hota:.3f}", m.idsw, m.mt, m.ml, r.fps]
         for v, m, r in per_video]))

    sections = [
        (f"Headline tracking metrics — tracker `{name}`, split `{args.split}` (pooled)",
         targets_table(pooled.headline(), TRACKING_TARGETS, markdown=True)),
        ("Per-sequence", render_table(
            ["Sequence", "MOTA", "IDF1", "HOTA", "IDSW", "MT", "ML", "Frag", "FPS"],
            [[v, f"{m.mota:.3f}", f"{m.idf1:.3f}", f"{m.hota:.3f}", m.idsw, m.mt, m.ml,
              f"{m.fragmentation:.2f}", r.fps] for v, m, r in per_video], markdown=True)),
        ("Error breakdown", render_table(["Key", "Value"], [
            ["TP", pooled.tp], ["FP", pooled.fp], ["FN", pooled.fn],
            ["ID switches", pooled.idsw], ["GT trajectories", pooled.gt_tracks],
            ["predicted trajectories", pooled.pred_tracks],
            ["MOTP (mean matched IoU)", f"{pooled.motp:.4f}"],
            ["DetA / AssA", f"{pooled.deta:.4f} / {pooled.assa:.4f}"],
        ], markdown=True)),
    ]
    out = cfg.path("reports_dir") / "tracking_report.md"
    write_markdown(out, f"Tracking Evaluation — {name}", sections, {
        "config fingerprint": cfg.fingerprint(), "git commit": git_sha(cfg.root),
        "tracker": name, "weights": args.weights or cfg.detection.weights,
        "mean FPS": round(mean_fps, 2),
    })
    write_json(cfg.path("reports_dir") / "tracking_results.json", {
        "tracker": name, "pooled": pooled.as_dict(),
        "per_video": {v: m.as_dict() for v, m, _ in per_video},
    })
    print(f"\n  wrote {out.relative_to(cfg.root)}")
    return 0


def cmd_ablate(cfg, args) -> int:
    """PRD §13.2 -- fill every row that the current phase can measure."""
    from .eval.report import git_sha, write_json, write_markdown
    from .eval.runner import run_detection_eval

    videos = _test_videos(cfg, args.split)
    base = cfg.detection.weights
    tuned = args.weights

    plan = [
        ("B0", "COCO-pretrained, zero fine-tuning, no tracker", base, None),
        ("B1", "COCO-pretrained + IoU-only tracker", base, "iou"),
        ("B3", "COCO-pretrained + ByteTrack", base, "bytetrack"),
        ("B4", "COCO-pretrained + BoT-SORT (GMC, no ReID)", base, "botsort"),
    ]
    if tuned:
        plan += [
            ("B2", "fine-tuned + IoU-only tracker", tuned, "iou"),
            ("B3f", "fine-tuned + ByteTrack", tuned, "bytetrack"),
            ("B4f", "fine-tuned + BoT-SORT (GMC, no ReID)", tuned, "botsort"),
        ]

    rows = []
    for row_id, note, weights, tracker in plan:
        log.info(f"ablation row {row_id}: {note}", extra={"event": "ablate_row", "row": row_id})
        det = run_detection_eval(cfg, split=args.split, weights=weights,
                                 limit=args.limit if hasattr(args, "limit") else None,
                                 with_slices=False)
        if tracker is None:
            rows.append([row_id, note, f"{det.overall.map50:.3f}", f"{det.overall.map50_95:.3f}",
                         "-", "-", "-", f"{det.timing['fps']:.1f}"])
            continue
        pooled, _, fps = _eval_tracker_on(cfg, videos, tracker, weights, args.max_frames)
        rows.append([
            row_id, note, f"{det.overall.map50:.3f}", f"{det.overall.map50_95:.3f}",
            f"{pooled.idf1:.3f}" if pooled else "-",
            f"{pooled.hota:.3f}" if pooled else "-",
            pooled.idsw if pooled else "-",
            f"{fps:.1f}",
        ])

    headers = ["#", "Configuration", "mAP50", "mAP50-95", "IDF1", "HOTA", "IDSW", "FPS"]
    table = render_table(headers, rows, markdown=False)
    print("\n" + table)

    notes = (
        "Rows B2/B3f/B4f require fine-tuned weights (`--weights runs/<id>/weights/best.pt`); "
        "B5-B10 (ReID gallery, imgsz 960, YOLO11m, hard negatives, TensorRT) land in Phase 3.\n\n"
        "FPS is measured on this machine's **CPU** (PRD R9: no GPU present), so it is the "
        "honest worst case rather than the M19 GPU target."
    )
    out = cfg.path("reports_dir") / "ablation.md"
    write_markdown(out, "Ablation Matrix (PRD §13.2)",
                   [("Measured rows", render_table(headers, rows, markdown=True)),
                    ("Notes", notes)],
                   {"config fingerprint": cfg.fingerprint(), "git commit": git_sha(cfg.root),
                    "split": args.split})
    write_json(cfg.path("reports_dir") / "ablation.json",
               {"headers": headers, "rows": rows})
    print(f"\n  wrote {out.relative_to(cfg.root)}")
    return 0


def cmd_runs(cfg) -> int:
    from .models.train import list_runs

    runs = list_runs(cfg)
    if not runs:
        print("no registered runs yet -- `python -m src.cli train`")
        return 0
    print(render_table(
        ["run id", "started", "device", "duration s", "best weights"],
        [[r.get("run_id"), r.get("started"), r.get("device"),
          r.get("duration_s"), r.get("best_weights") or "-"] for r in runs]))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    setup_logging(cfg.path("logs_dir") / "cvdr.jsonl", verbose=args.verbose)

    try:
        if args.command == "env":
            return cmd_env(cfg)
        if args.command == "sample":
            return cmd_sample(cfg, args)
        if args.command == "labels":
            return cmd_labels(cfg)
        if args.command == "split":
            return cmd_split(cfg, args)
        if args.command == "validate":
            return cmd_validate(cfg)
        if args.command == "eval-det":
            return cmd_eval_det(cfg, args)
        if args.command == "train":
            return cmd_train(cfg, args)
        if args.command == "track":
            return cmd_track(cfg, args)
        if args.command == "eval-track":
            return cmd_eval_track(cfg, args)
        if args.command == "ablate":
            return cmd_ablate(cfg, args)
        if args.command == "runs":
            return cmd_runs(cfg)
        if args.command in {"reid", "demo", "export"}:
            print(f"`{args.command}` lands in Phase 3 -- see PHASE_PLAN_Project2.md")
            return 0
    except (FileNotFoundError, ValueError, IOError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
