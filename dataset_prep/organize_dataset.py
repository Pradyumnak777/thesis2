#!/usr/bin/env python3
"""Assemble the ExpertAF training samples from the drive pair-files + local EgoExo4D media.
(See earlier docstring -- unchanged behavior, now parallelized across entries.)
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

FPS = 30
SPLITS = ("train", "val", "test")

_TAKES_META = None
_ROOT = None
_DRY = None


def load_takes_meta(root):
    with open(Path(root) / "takes.json") as f:
        return {t["take_name"]: t for t in json.load(f)}


def find_pose_file(root, uid):
    for kind in ("automatic", "annotation"):
        for split in SPLITS:
            p = Path(root) / "annotations" / "ego_pose" / split / "body" / kind / f"{uid}.json"
            if p.exists():
                return p, kind
    return None, None


@lru_cache(maxsize=32)
def _load_pose_json(pose_path_str):
    with open(pose_path_str) as f:
        return json.load(f)


def _keypoints(frame_ann):
    a = frame_ann[0] if isinstance(frame_ann, list) and frame_ann else frame_ann
    if not isinstance(a, dict):
        return {}
    kp = a.get("annotation3D") or {}
    out = {}
    for joint, v in kp.items():
        if isinstance(v, dict) and "x" in v:
            out[joint] = [v["x"], v["y"], v["z"]]
    return out


def slice_pose(pose_path, start_f, end_f):
    data = _load_pose_json(str(pose_path))
    out = {}
    for frame_str, ann in data.items():
        try:
            fr = int(frame_str)
        except (ValueError, TypeError):
            continue
        if start_f <= fr < end_f:
            kp = _keypoints(ann)
            if kp:
                out[fr] = kp
    return out


def video_paths(root, meta):
    vdir = Path(root) / "takes" / meta["take_name"] / "frame_aligned_videos" / "downscaled" / "448"
    ego = next(vdir.glob("aria*_214-1.mp4"), None) or next(vdir.glob("aria*rgb*.mp4"), None)
    exo = None
    if meta.get("best_exo"):
        cand = vdir / f"{meta['best_exo']}.mp4"
        exo = cand if cand.exists() else next(vdir.glob(f"*{meta['best_exo']}*.mp4"), None)
    return ego, exo


def cut_video(src, start_f, end_f, dst):
    if dst.exists() or src is None:
        return
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-ss", f"{start_f / FPS:.3f}", "-i", str(src),
         "-t", f"{(end_f - start_f) / FPS:.3f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-threads", "2",  # cap per-clip threads so N parallel workers don't oversubscribe the node
         "-an", str(dst)],
        check=True)


def build_side(role, take, rng, takes_meta, root, out_dir, dry_run):
    start_f, end_f = json.loads(rng) if isinstance(rng, str) else rng
    meta = takes_meta[take]
    uid = meta["take_uid"]

    pose_path, pose_kind = find_pose_file(root, uid)
    pose = slice_pose(pose_path, start_f, end_f) if pose_path else {}
    coverage = len(pose) / max(end_f - start_f, 1)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        ego, exo = video_paths(root, meta)
        cut_video(ego, start_f, end_f, out_dir / f"{role}_ego.mp4")
        cut_video(exo, start_f, end_f, out_dir / f"{role}_exo.mp4")
        with open(out_dir / f"{role}_pose.json", "w") as f:
            json.dump({"take": take, "uid": uid, "range": [start_f, end_f],
                       "pose_source": pose_kind, "pose": pose}, f)

    side_meta = {"take": take, "uid": uid, "range": [start_f, end_f],
                 "best_exo": meta.get("best_exo"), "pose_source": pose_kind,
                 "pose_coverage": round(coverage, 3)}
    return side_meta, coverage


def _init_worker(takes_meta, root, dry_run):
    global _TAKES_META, _ROOT, _DRY
    _TAKES_META, _ROOT, _DRY = takes_meta, root, dry_run


def _process_entry(entry, split_dir):
    out_dir = split_dir / entry["id"]
    learner, cov_l = build_side("learner", entry["video1"], entry["video1_range"],
                                _TAKES_META, _ROOT, out_dir, _DRY)
    expert, cov_e = build_side("expert", entry["video2"], entry["video2_range"],
                               _TAKES_META, _ROOT, out_dir, _DRY)
    rec = {
        "id": entry["id"],
        "body_part": entry.get("corrected_body_part"),
        "feedback": entry.get("text"),
        "good_execution": entry.get("good_execution_commentary"),
        "expert_id": entry.get("expert_id"),
        "commentary_idx": entry.get("commentary_idx"),
        "best_dtw_score": entry.get("best_dtw_score"),
        "learner": learner,
        "expert": expert,
        "conversations": entry.get("conversations"),
    }
    if not _DRY:
        with open(out_dir / "meta.json", "w") as f:
            json.dump(rec, f, indent=2)
    return rec, cov_l, cov_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--egoexo-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel entries processed at once (each may spawn ffmpeg with -threads 2)")
    args = ap.parse_args()

    takes_meta = load_takes_meta(args.egoexo_root)
    with open(args.pairs) as f:
        entries = json.load(f)
    if args.limit:
        entries = entries[:args.limit]

    split_dir = Path(args.out) / args.split
    manifest_path = Path(args.out) / f"{args.split}_manifest.jsonl"
    if not args.dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    n_low, n_missing, records = 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                              initargs=(takes_meta, args.egoexo_root, args.dry_run)) as ex:
        futures = [ex.submit(_process_entry, e, split_dir) for e in entries]
        for i, fut in enumerate(as_completed(futures)):
            rec, cov_l, cov_e = fut.result()
            records.append(rec)
            for cov in (cov_l, cov_e):
                if cov == 0:
                    n_missing += 1
                elif cov < 0.9:
                    n_low += 1
            if i % 20 == 0:
                print(f"  ... {i}/{len(entries)}", flush=True)

    if not args.dry_run:
        with open(manifest_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    print(f"[{args.split}] {len(entries)} entries | "
          f"{n_missing} clip-sides with NO pose file | "
          f"{n_low} sides with <90% (partial) pose coverage")
    if not args.dry_run:
        print(f"  wrote samples to {split_dir}/ and manifest {manifest_path}")


if __name__ == "__main__":
    main()