"""Phase 3 command implementations: ReID, cross-camera, demo, export, failures.

Kept in their own module so `cli.py` stays a readable dispatch table rather
than a thousand-line file. Imported into `cli.py`'s namespace at the bottom of
that module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils.logging import get_logger, render_table

log = get_logger("cli.phase3")

REID_TARGETS = (
    ("M14", "Rank-1 accuracy", 0.75, "ge"),
    ("M15", "Rank-5 accuracy", 0.90, "ge"),
    ("M16", "ReID mAP", 0.65, "ge"),
    ("M17", "Post-occlusion recovery", 0.70, "ge"),
    ("M18", "Cross-camera match rate", 0.60, "ge"),
)


def _videos_for(cfg, split: str) -> list[Path]:
    """Videos belonging to a split. `trainval` is train + val, never test.

    Calibrating a threshold on the test split and then reporting test-split
    recovery would be selecting on the test set -- the same class of error as
    a leaked split, one stage later.
    """
    from .data.manifest import read_manifest

    wanted = {"train", "val"} if split == "trainval" else {split}
    rows = read_manifest(cfg.path("manifest"))
    names = sorted({r.source_video for r in rows if r.split in wanted})
    return [cfg.path("raw_videos_dir") / n for n in names]


def _extractor(cfg, backend: str | None):
    from .reid.extractor import ReidExtractor

    return ReidExtractor(cfg, backend=backend)


# ---------------------------------------------------------------------------
# 3.1 -- tau calibration
# ---------------------------------------------------------------------------


def cmd_calibrate(cfg, args) -> int:
    from .eval.report import git_sha, write_json, write_markdown
    from .reid.calibrate import calibrate_tau, plot_calibration

    videos = _videos_for(cfg, args.split)
    if not videos:
        print(f"no videos in split '{args.split}' -- run `python -m src.cli split` first")
        return 1

    extractor = _extractor(cfg, args.backend)
    result = calibrate_tau(cfg, videos, extractor=extractor)

    if not result.points:
        print(
            "no scorable occlusion events found.\n"
            "  The sweep needs ground truth with a visibility column and gaps of at\n"
            "  least 15 frames. Render the occluder scenes:\n"
            "    python scripts/make_sample_videos.py --scenes scene04 scene09 scene11"
        )
        return 1

    print(render_table(
        ["tau", "TP", "FP(merge)", "FN(new id)", "precision", "recall", "F1"],
        [[f"{p.tau:.2f}", p.true_positive, p.false_positive, p.false_negative,
          f"{p.precision:.3f}", f"{p.recall:.3f}", f"{p.f1:.3f}"] for p in result.points]))
    print(f"\n  chosen tau_reid = {result.best_tau}  (F1 {result.best_f1:.3f})")
    print(f"  backend         = {result.backend}")
    print(f"  events / pairs  = {result.n_events} / {result.n_pairs}")

    figure = plot_calibration(result, cfg.path("reports_dir") / "figures" / "tau_calibration.png")
    sections = [
        ("Chosen threshold", render_table(["Key", "Value"], [
            ["tau_reid", result.best_tau], ["F1 at tau", result.best_f1],
            ["ReID backend", result.backend], ["occlusion events", result.n_events],
            ["candidate pairs", result.n_pairs], ["calibration split", args.split],
        ], markdown=True)),
        ("Sweep", render_table(
            ["tau", "TP", "FP (identity merge)", "FN (new id issued)", "precision", "recall", "F1"],
            [[f"{p.tau:.2f}", p.true_positive, p.false_positive, p.false_negative,
              f"{p.precision:.3f}", f"{p.recall:.3f}", f"{p.f1:.3f}"] for p in result.points],
            markdown=True)),
        ("Reading this", (
            "A **false positive** here is an identity *merge*: two different objects "
            "collapsed into one id. A **false negative** is a genuine return rejected, "
            "which issues a new id and inflates the unique count. F1 balances the two; "
            "the curve is published so the shape, not only the argmax, is auditable.\n\n"
            + (f"![calibration]({figure.relative_to(cfg.path('reports_dir')).as_posix()})"
               if figure else "*(matplotlib unavailable; figure not rendered)*"))),
    ]
    out = cfg.path("reports_dir") / "calibration.md"
    write_markdown(out, "ReID Gallery Threshold Calibration (PRD §9.4)", sections,
                   {"config fingerprint": cfg.fingerprint(), "git commit": git_sha(cfg.root),
                    "split": args.split, "videos": result.source})
    write_json(cfg.path("reports_dir") / "calibration.json", result.as_dict())
    print(f"\n  wrote {out.relative_to(cfg.root)}")

    if args.apply:
        _apply_tau(cfg, result.best_tau)
        print(f"  applied tau_reid = {result.best_tau} to configs/default.yaml")
    return 0


def _apply_tau(cfg, tau: float) -> None:
    """Write the calibrated threshold back into the config file.

    Config stays the single source of truth (PRD 9.7): the runtime must read
    the same number the calibration report published, and a value typed by hand
    into two places will eventually differ in one of them.
    """
    import re

    path = cfg.root / "configs" / "default.yaml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"(\n    threshold:\s*)([0-9.]+)",
        lambda m: f"{m.group(1)}{tau:g}",
        text, count=1,
    )
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 3.1 -- ReID evaluation
# ---------------------------------------------------------------------------


def cmd_reid_eval(cfg, args) -> int:
    from .data.mot import read_mot
    from .eval.report import git_sha, targets_table, write_json, write_markdown
    from .eval.reid_metrics import cmc_and_map, post_occlusion_recovery
    from .pipeline.track_video import track_video

    videos = _videos_for(cfg, args.split)
    if not videos:
        print(f"no videos in split '{args.split}'")
        return 1

    extractor = _extractor(cfg, args.backend)
    tau = args.tau if args.tau is not None else cfg.reid.gallery.threshold
    from .reid.gallery import ReidGallery

    rows_out, all_detail = [], []
    total_scored = total_recovered = 0

    for video in videos:
        gt_file = cfg.root / "data" / "gt" / f"{video.stem}_gt.txt"
        if not gt_file.exists():
            continue
        res = track_video(
            cfg, video, tracker_type="botsort", weights=args.weights,
            with_reid=True, with_gallery=True, max_frames=args.max_frames,
            extractor=extractor, gallery=ReidGallery(cfg, threshold=tau),
        )
        gt_rows = read_mot(gt_file)
        if args.max_frames:
            gt_rows = [r for r in gt_rows if r.frame <= args.max_frames]
        rate, scored, recovered, detail = post_occlusion_recovery(gt_rows, res.predictions)
        total_scored += scored
        total_recovered += recovered
        all_detail.extend(detail)
        rows_out.append([video.name, scored, recovered,
                         "n/a" if rate is None else f"{rate:.3f}",
                         len(res.restorations), res.fps])

    # Rank-k / CMC on a query-gallery protocol built from the ground truth
    # crops, which scores the embedding space itself independently of any
    # tracker -- M14-M16 are a property of the descriptor, not of association.
    cmc = _rank_protocol(cfg, videos, extractor, args.max_frames)
    overall_recovery = round(total_recovered / total_scored, 4) if total_scored else None

    headline = {
        "M14": cmc.rank1, "M15": cmc.rank5, "M16": cmc.map,
        "M17": overall_recovery, "M18": None,
    }
    headline = {k: v for k, v in headline.items() if v is not None}
    print("\n" + targets_table(headline, REID_TARGETS, markdown=False))
    print("\n" + render_table(
        ["video", "occlusion events", "recovered", "M17 rate", "gallery restores", "FPS"],
        rows_out))

    switches = [d for d in all_detail if d.get("outcome") == "id_switch"]
    sections = [
        (f"Headline ReID metrics — backend `{extractor.describe()['reid_backend']}`, "
         f"tau {tau}, split `{args.split}`",
         targets_table(headline, REID_TARGETS, markdown=True)),
        ("Post-occlusion recovery per sequence (M17)", render_table(
            ["Sequence", "Occlusion events scored", "Recovered", "Rate", "Gallery restores", "FPS"],
            rows_out, markdown=True)),
        ("Query/gallery protocol (M14–M16)", render_table(["Key", "Value"], [
            ["queries", cmc.n_queries], ["gallery items", cmc.n_gallery],
            ["identities", cmc.n_identities],
            ["CMC (rank 1..10)", ", ".join(f"{v:.3f}" for v in cmc.cmc)],
        ], markdown=True)),
        ("Unrecovered occlusions (the failures worth reading)", render_table(
            ["GT track", "gap (frames)", "id before", "id after"],
            [[d["gt_track"], d["gap"], d["before"], d["after"]] for d in switches[:20]],
            markdown=True) if switches else "*No identity was lost across a scored occlusion.*"),
    ]
    out = cfg.path("reports_dir") / "reid_report.md"
    write_markdown(out, "Re-Identification Evaluation (PRD §4.4)", sections, {
        "config fingerprint": cfg.fingerprint(), "git commit": git_sha(cfg.root),
        "split": args.split, "tau_reid": tau, **extractor.describe(),
    })
    write_json(cfg.path("reports_dir") / "reid_results.json", {
        "headline": headline, "cmc": cmc.as_dict(),
        "recovery_detail": all_detail, "tau": tau, **extractor.describe(),
    })
    print(f"\n  wrote {out.relative_to(cfg.root)}")
    return 0


# A gallery sample within this many frames of the query is excluded. Without
# it the protocol degenerates: matching an object to itself one frame later is
# trivially easy and drives Rank-1 towards 1.0 while measuring nothing about
# re-identification. This plays the role the same-camera exclusion plays in
# Market-1501, which single-view footage cannot provide.
RANK_MIN_FRAME_GAP = 90          # 3 s at 30 fps


def _rank_protocol(cfg, videos, extractor, max_frames=None):
    """Build a query/gallery set from ground-truth crops and score Rank-k / mAP."""
    import cv2

    from .data.mot import IGNORE_VISIBILITY, read_mot
    from .eval.reid_metrics import cmc_and_map

    embeddings, ids, cams = [], [], []
    for video in videos:
        gt_file = cfg.root / "data" / "gt" / f"{video.stem}_gt.txt"
        if not gt_file.exists():
            continue
        rows = [r for r in read_mot(gt_file) if r.visibility >= IGNORE_VISIBILITY]
        if max_frames:
            rows = [r for r in rows if r.frame <= max_frames]
        by_frame: dict[int, list] = {}
        for r in rows:
            by_frame.setdefault(r.frame, []).append(r)

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        try:
            # Sample sparsely: near-duplicate crops from adjacent frames would
            # make Rank-1 trivially high and stop measuring re-identification.
            for frame_no in sorted(by_frame)[::30]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
                ok, frame = cap.read()
                if not ok:
                    continue
                visible = by_frame[frame_no]
                for idx, emb in extractor.extract(frame, [r.xyxy for r in visible]).items():
                    embeddings.append(emb)
                    # Identity is scoped per video: track 1 in two different
                    # scenes is two different objects.
                    ids.append(hash((video.stem, visible[idx].track_id)) % 10**8)
                    # "Camera" here is a temporal bucket: samples in the same
                    # bucket are excluded from each other's gallery, enforcing
                    # the minimum gap above.
                    cams.append(f"{video.stem}_b{frame_no // RANK_MIN_FRAME_GAP}")
        finally:
            cap.release()

    if len(embeddings) < 4:
        from .eval.reid_metrics import ReidMetrics

        return ReidMetrics()
    # Every sample acts as a query against all the others; the same-camera
    # exclusion (here, same-frame) removes the trivial self-match.
    return cmc_and_map(embeddings, ids, embeddings, ids, cams, cams)


# ---------------------------------------------------------------------------
# 3.2 -- cross-camera
# ---------------------------------------------------------------------------


def cmd_cross_camera(cfg, args) -> int:
    from .eval.report import git_sha, targets_table, write_json, write_markdown
    from .pipeline.cross_camera import find_camera_pairs, run_cross_camera

    if args.cam_a and args.cam_b:
        pairs = [(Path(args.cam_a), Path(args.cam_b))]
    else:
        pairs = find_camera_pairs(cfg)
    if not pairs:
        print("no two-view scene found. Name the files `<scene>_camA.mp4` / `<scene>_camB.mp4`.")
        return 1

    extractor = _extractor(cfg, args.backend)
    tau = args.tau if args.tau is not None else cfg.reid.gallery.threshold
    results = [run_cross_camera(cfg, a, b, extractor=extractor, threshold=tau) for a, b in pairs]

    rows = [[r.cam_a, r.cam_b, r.scored, r.matched,
             "n/a" if r.match_rate is None else f"{r.match_rate:.3f}"] for r in results]
    print("\n" + render_table(["camera A", "camera B", "identities", "matched", "M18 rate"], rows))

    scored = sum(r.scored for r in results)
    matched = sum(r.matched for r in results)
    overall = round(matched / scored, 4) if scored else None
    if overall is not None:
        print("\n" + targets_table({"M18": overall}, REID_TARGETS, markdown=False))

    detail_rows = [
        [d["cam_a_track"], d["expected_cam_b_track"], d["predicted"], d["distance"],
         "yes" if d["matched"] else "no"]
        for r in results for d in r.detail
    ]
    sections = [
        ("Headline (M18)", targets_table({"M18": overall} if overall is not None else {},
                                         REID_TARGETS, markdown=True)),
        ("Per camera pair", render_table(
            ["Camera A", "Camera B", "Identities scored", "Matched", "Rate"], rows, markdown=True)),
        ("Per identity", render_table(
            ["Cam-A track", "Expected cam-B track", "Predicted", "Cosine distance", "Matched"],
            detail_rows, markdown=True)),
        ("Design note", (
            "Cross-camera matching reuses the occlusion-recovery gallery unchanged — same "
            "cosine distance, same class gate, same calibrated `tau_reid`, with a `camera_id` "
            "field and a wider temporal window (PRD §9.4). No new machinery was added for "
            "the hand-off, which is the design claim this table tests.")),
    ]
    out = cfg.path("reports_dir") / "cross_camera.md"
    write_markdown(out, "Cross-Camera Re-Identification (PRD §9.4, M18)", sections, {
        "config fingerprint": cfg.fingerprint(), "git commit": git_sha(cfg.root),
        "tau_reid": tau, **extractor.describe(),
    })
    write_json(cfg.path("reports_dir") / "cross_camera.json",
               {"overall_M18": overall, "pairs": [r.as_dict() for r in results]})
    print(f"\n  wrote {out.relative_to(cfg.root)}")
    return 0


# ---------------------------------------------------------------------------
# 3.2 -- live demo
# ---------------------------------------------------------------------------


def cmd_demo(cfg, args) -> int:
    from .pipeline.demo import run_demo

    line = None
    if args.line:
        try:
            line = tuple(float(v) for v in args.line.split(","))
            if len(line) != 4:
                raise ValueError
        except ValueError:
            print("--line expects four comma-separated numbers: x1,y1,x2,y2")
            return 1

    result = run_demo(
        cfg, args.source, tracker_type=args.tracker, weights=args.weights,
        with_reid=args.reid, with_gallery=args.gallery, save=args.save,
        show=args.show, max_frames=args.max_frames, blur_faces=args.blur_faces,
        line=line,
    )
    print(render_table(["Key", "Value"], [[k, v] for k, v in result.items()]))
    return 0


# ---------------------------------------------------------------------------
# 3.3 -- export
# ---------------------------------------------------------------------------


def cmd_export(cfg, args) -> int:
    from .eval.report import git_sha, write_json, write_markdown
    from .models.export import PARITY_TOLERANCE, export_model, verify_parity

    weights = args.weights or cfg.detection.weights
    result = export_model(cfg, weights, fmt=args.format, half=args.half)
    if not result.ok:
        print(f"export failed: {result.error}")
        return 1

    rows = [["source weights", result.weights], ["format", result.fmt],
            ["output", result.output], ["size (MB)", result.size_mb],
            ["M22 target", "<= 25 MB"],
            ["M22 verdict", "PASS" if result.size_mb <= 25 else "FAIL"]]

    parity = None
    if args.verify:
        parity = verify_parity(cfg, weights, result.output, split="test", limit=args.limit)
        if parity.ok:
            rows += [
                ["PyTorch mAP@0.5", parity.pytorch_map50],
                ["exported mAP@0.5", parity.exported_map50],
                ["absolute delta", parity.delta],
                ["NFR-6 tolerance", PARITY_TOLERANCE],
                ["NFR-6 verdict", "PASS" if parity.parity_ok else "FAIL"],
                ["PyTorch FPS", parity.timing.get("pytorch_fps")],
                ["exported FPS", parity.timing.get("exported_fps")],
                ["speedup", parity.timing.get("speedup")],
            ]
        else:
            rows.append(["parity check", f"failed: {parity.error}"])

    print(render_table(["Key", "Value"], rows))
    out = cfg.path("reports_dir") / "export.md"
    write_markdown(out, "Export & Parity (PRD EXP-8, NFR-6, M22)",
                   [("Result", render_table(["Key", "Value"], rows, markdown=True)),
                    ("Note", "FPS here is CPU-only (PRD R9). The TensorRT FP16/INT8 arm of "
                             "EXP-8 needs CUDA and is documented rather than measured.")],
                   {"config fingerprint": cfg.fingerprint(), "git commit": git_sha(cfg.root)})
    write_json(cfg.path("reports_dir") / "export.json",
               {"export": result.as_dict(), "parity": parity.as_dict() if parity else None})
    print(f"\n  wrote {out.relative_to(cfg.root)}")
    return 0 if (parity is None or parity.parity_ok) else 1


# ---------------------------------------------------------------------------
# 3.4 -- failure gallery
# ---------------------------------------------------------------------------


def cmd_failures(cfg, args) -> int:
    from .data.manifest import read_manifest
    from .eval.failures import REMEDIATION, collect_detection_failures, render_failure_images
    from .eval.report import git_sha, write_json, write_markdown
    from .eval.runner import load_ground_truth
    from .models.detector import Detector

    rows = [r for r in read_manifest(cfg.path("manifest")) if r.split == args.split]
    if not rows:
        print(f"no frames in split '{args.split}'")
        return 1

    detector = Detector(cfg, args.weights)
    preds = []
    for row in rows:
        path = cfg.root / row.image_path
        if path.exists():
            preds.extend(detector.predict(str(path), row.image_id))
    gts = load_ground_truth(rows, cfg)

    report = collect_detection_failures(preds, gts, rows, cfg, max_cases=args.max_cases)
    out_dir = cfg.path("reports_dir") / "failures"
    report.saved = render_failure_images(report.cases, gts, preds, cfg, out_dir)

    freq = [[cause, n, REMEDIATION.get(cause, "")] for cause, n in report.counts.most_common()]
    print("\n" + render_table(["root cause", "count", "remediation"], freq))
    print(f"\n  {report.saved} annotated failure images -> "
          f"{out_dir.relative_to(cfg.root)}  (PRD 13.5 requires >= 20)")

    sections = [
        ("Root-cause frequency — this table is the next iteration's work plan",
         render_table(["Root cause", "Count", "Remediation"], freq, markdown=True)),
        ("Saved cases", render_table(
            ["#", "Cause", "Image", "Detail", "Lighting", "Blur", "IoU", "Conf"],
            [[i, c.cause, c.image_id, c.detail, c.lighting, c.blur_level, c.iou, c.conf]
             for i, c in enumerate(report.cases)], markdown=True)),
        ("How causes are assigned", (
            "Automatically, from the geometry of each failure — size band, lighting and blur "
            "attributes from the manifest, and whether an overlapping-but-below-threshold box "
            "existed. Hand-labelled causes drift with the labeller, and the frequency table is "
            "only useful across iterations if the labelling rule is fixed.")),
    ]
    write_markdown(cfg.path("reports_dir") / "failures.md",
                   "Failure Gallery (PRD §13.5)", sections,
                   {"config fingerprint": cfg.fingerprint(), "git commit": git_sha(cfg.root),
                    "split": args.split, "weights": args.weights or cfg.detection.weights})
    write_json(cfg.path("reports_dir") / "failures.json",
               {**report.as_dict(), "cases": [c.as_dict() for c in report.cases]})
    print(f"  wrote reports/failures.md")
    return 0


# ---------------------------------------------------------------------------
# 3.4 -- assemble the final report
# ---------------------------------------------------------------------------


def cmd_report(cfg) -> int:
    """Stitch every generated report into one document for the reviewer."""
    from .eval.report import git_sha, write_markdown

    reports_dir = cfg.path("reports_dir")
    order = [
        ("eval_report.md", "Detection"),
        ("tracking_report.md", "Tracking"),
        ("calibration.md", "ReID threshold calibration"),
        ("reid_report.md", "Re-identification"),
        ("cross_camera.md", "Cross-camera hand-off"),
        ("ablation.md", "Ablation matrix"),
        ("export.md", "Export & parity"),
        ("failures.md", "Failure gallery"),
    ]
    sections, missing = [], []
    for filename, title in order:
        path = reports_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        body = path.read_text(encoding="utf-8")
        body = "\n".join(
            line for line in body.splitlines()
            if not line.startswith("# ") and not line.startswith("*Generated")
        ).strip()
        sections.append((f"{title}  —  `reports/{filename}`", body))

    if missing:
        sections.append(("Not yet generated", "\n".join(f"- `reports/{m}`" for m in missing)))

    out = write_markdown(reports_dir / "FINAL_REPORT.md",
                         "cv-detection-reid — Consolidated Results", sections,
                         {"config fingerprint": cfg.fingerprint(),
                          "git commit": git_sha(cfg.root)})
    print(f"  assembled {len(sections)} sections -> {out.relative_to(cfg.root)}")
    if missing:
        print(f"  missing: {', '.join(missing)}")
    return 0
