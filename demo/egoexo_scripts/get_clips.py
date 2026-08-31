'''
fromt he untrimmed videos, get the clips by utilizing the sentence encoder + delta frames
'''

import json, re, subprocess
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EGOEXO_ROOT = Path("demo/egoexo4d_v2") 
SCRIPT_DIR = Path(__file__).resolve().parent
META_FILE = SCRIPT_DIR / "jumpshot_late_expert_takes.json"
OUT_DIR = Path("demo/basketball_expert_clips/Mid-range jump shot")

FPS = 30
DELTA_MINUS = 45   # frames before t*, gather + rise
DELTA_PLUS = 44    # frames after t*, release + follow-through/landing
T = DELTA_MINUS + DELTA_PLUS + 1  # 90

OPERATIVE_PHRASES = [
    "player performs the jump and shoots",
    "shot is released",
]
SIM_THRESHOLD = 0.5
MIN_SEP_FRAMES = T  # don't let two accepted windows from the same take overlap


def get_video_path(take_meta, cam_id):
    return EGOEXO_ROOT / take_meta["root_dir"] / "frame_aligned_videos" / "downscaled" / "448" / f"{cam_id}.mp4"


def cut_window(src_video, start_frame, end_frame, dst, fps=FPS):
    # cuts [start_frame, end_frame] inclusive straight from the untrimmed take video
    if dst.exists():
        return
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame + 1) / fps
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.4f}", "-i", str(src_video),
        "-t", f"{duration_sec:.4f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an",
        str(dst),
    ], check=True)


def load_narrations(take_uid):
    for split in ("train", "val"):
        p = EGOEXO_ROOT / "annotations" / f"atomic_descriptions_{split}.json"
        d = json.load(open(p))
        ann = d.get("annotations", d)
        if isinstance(ann, dict):
            if take_uid in ann:
                return ann[take_uid]
        else:
            hits = [a for a in ann if a.get("take_uid") == take_uid]
            if hits:
                return hits
    return []


def flatten_descriptions(narr_entries):
    # each take_uid maps to a list of annotator passes, each with its own "descriptions" list
    # flatten all passes into one list of individual description dicts
    out = []
    for pass_ in narr_entries:
        out.extend(pass_.get("descriptions", []))
    return out


def find_operative_moments(descriptions, encoder, phrases=OPERATIVE_PHRASES, threshold=SIM_THRESHOLD):
    # returns a list of (timestamp_sec, score, text, best_exo_cam_id) for every description
    # that clears the threshold, NOT just the single best one -- a take has multiple reps,
    # so we want a candidate per rep, deduped below
    usable = [d for d in descriptions if not d.get("unsure", False)]
    if not usable:
        return []

    texts = [d["text"] for d in usable]
    narr_emb = encoder.encode(texts, normalize_embeddings=True)
    phrase_emb = encoder.encode(phrases, normalize_embeddings=True)

    sim = narr_emb @ phrase_emb.T
    best_score = sim.max(axis=1)

    candidates = []
    for d, score in zip(usable, best_score):
        if score >= threshold:
            # cam_id = d.get("best_exo", {}).get("cam_id")
            #if best_exo is None,
            cam_id = (d.get("best_exo") or {}).get("cam_id")
            candidates.append((d["timestamp"], float(score), d["text"], cam_id))
    return candidates


def dedupe_candidates(candidates, fps=FPS, min_sep_frames=MIN_SEP_FRAMES):
    # greedy: keep highest-scoring candidates first, drop anything too close in time
    # to something already accepted
    candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
    accepted = []
    for t_sec, score, text, cam_id in candidates:
        t_frame = round(t_sec * fps)
        if all(abs(t_frame - round(a[0] * fps)) >= min_sep_frames for a in accepted):
            accepted.append((t_sec, score, text, cam_id))
    return sorted(accepted, key=lambda c: c[0])  # chronological order for saving


def main():
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    takes = json.load(open(META_FILE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_clips, total_skipped_bounds, total_skipped_no_cam = 0, 0, 0

    for take_name, meta in takes.items():
        narr_entries = load_narrations(meta["take_uid"])
        descriptions = flatten_descriptions(narr_entries)
        candidates = find_operative_moments(descriptions, encoder)
        accepted = dedupe_candidates(candidates)

        if not accepted:
            print(f"{take_name}: no operative moments found above threshold")
            continue

        total_frames = round(meta["duration_sec"] * FPS)

        for i, (t_sec, score, text, cam_id) in enumerate(accepted):
            t_frame = round(t_sec * FPS)
            start = t_frame - DELTA_MINUS
            end = t_frame + DELTA_PLUS

            if start < 0 or end >= total_frames:
                total_skipped_bounds += 1
                print(f"  {take_name} rep{i}: window [{start},{end}] out of bounds (take has {total_frames} frames), skipping")
                continue

            # cam = cam_id or meta["chosen_cam"]
            cam = cam_id or "cam01"
            src = get_video_path(meta, cam)
            if not src.exists():
                total_skipped_no_cam += 1
                print(f"  {take_name} rep{i}: {src} not found, skipping")
                continue

            dst = OUT_DIR / f"{take_name}__rep{i}__t{t_sec:.2f}.mp4"
            cut_window(src, start, end, dst)
            total_clips += 1
            print(f"  {take_name} rep{i}: t*={t_sec:.2f}s score={score:.2f} cam={cam} "
                  f"\"{text.strip()}\" -> frames[{start},{end}] -> {dst.name}")

    print(f"\ndone: {total_clips} clips extracted, "
          f"{total_skipped_bounds} skipped (out of bounds), "
          f"{total_skipped_no_cam} skipped (missing camera file)")


if __name__ == "__main__":
    main()