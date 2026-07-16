#!/usr/bin/env python3
"""Cut the 4-second learner/expert clips defined by the ExpertAF pair files.

For every entry:
  video1 + video1_range -> learner clip (ego + best_exo views)
  video2 + video2_range -> expert  clip (ego + best_exo views)
plus the matching slice of the automatic 3D body pose.

Frame indices are in the take's 30 fps frame-aligned timeline (120 frames = 4 s).
Run with --dry-run first to get a pose-coverage report without cutting video.
"""
import argparse
import json
import subprocess
from pathlib import Path

FPS = 30
SPLITS = ("train", "val", "test")


def load_takes_meta(egoexo_root):
    with open(Path(egoexo_root) / "takes.json") as f:
        return {t["take_name"]: t for t in json.load(f)}


def find_pose_file(egoexo_root, take_uid):
    """ExpertAF uses the triangulated pseudo-GT, so prefer automatic/."""
    for kind in ("automatic", "annotation"):
        for split in SPLITS:
            p = (Path(egoexo_root) / "annotations" / "ego_pose" / split
                 / "body" / kind / f"{take_uid}.json")
            if p.exists():
                return p, kind
    return None, None


def video_paths(egoexo_root, meta):
    vdir = (Path(egoexo_root) / "takes" / meta["take_name"]
            / "frame_aligned_videos" / "downscaled" / "448")
    ego = next(vdir.glob("aria*_214-1.mp4"), None)  # Aria RGB stream
    exo = vdir / f"{meta['best_exo']}.mp4" if meta.get("best_exo") else None
    return ego, (exo if exo and exo.exists() else None)


def cut_video(src, start_f, end_f, dst):
    if dst.exists():
        return
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-ss", f"{start_f / FPS:.3f}", "-i", str(src),
         "-t", f"{(end_f - start_f) / FPS:.3f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", str(dst)],
        check=True)


def slice_pose(pose_path, start_f, end_f):
    """Return {frame_idx: {joint: [x,y,z]}} for frames in [start_f, end_f)."""
    with open(pose_path) as f:
        data = json.load(f)
    out = {}
    for frame_str, ann in data.items():
        fr = int(frame_str)
        if not (start_f <= fr < end_f):
            continue
        a = ann[0] if isinstance(ann, list) else ann
        kp3d = a.get("annotation3D") or {}
        out[fr] = {j: [v["x"], v["y"], v["z"]] for j, v in kp3d.items()}
    return out


def process_side(entry, side, takes_meta, egoexo_root, out_dir, dry_run):
    take_name = entry[side]
    start_f, end_f = json.loads(entry[f"{side}_range"])
    meta = takes_meta[take_name]

    pose_path, pose_kind = find_pose_file(egoexo_root, meta["take_uid"])
    pose = slice_pose(pose_path, start_f, end_f) if pose_path else {}
    coverage = len(pose) / max(end_f - start_f, 1)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        ego, exo = video_paths(egoexo_root, meta)
        if ego:
            cut_video(ego, start_f, end_f, out_dir / f"{side}_ego.mp4")
        if exo:
            cut_video(exo, start_f, end_f, out_dir / f"{side}_exo.mp4")
        with open(out_dir / f"{side}_pose.json", "w") as f:
            json.dump({"take": take_name, "range": [start_f, end_f],
                       "pose_source": pose_kind, "pose": pose}, f)
    return coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="ExpertAF train or test json")
    ap.add_argument("--egoexo-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="only report pose coverage, cut nothing")
    args = ap.parse_args()

    takes_meta = load_takes_meta(args.egoexo_root)
    with open(args.pairs) as f:
        entries = json.load(f)

    low_coverage = 0
    for entry in entries:
        out_dir = Path(args.out) / entry["id"]
        for side in ("video1", "video2"):
            cov = process_side(entry, side, takes_meta,
                               args.egoexo_root, out_dir, args.dry_run)
            if cov < 0.9:
                low_coverage += 1
                print(f"low pose coverage {cov:.0%}: {entry['id']} [{side}]")

    print(f"{len(entries)} entries processed; "
          f"{low_coverage} clip sides with <90% pose coverage")


if __name__ == "__main__":
    main()