'''
1. read "expertedit_eval_benchmark_v1.0/expert_clips.csv"
2. from that info, take data from- "dataset_prep/egoexo_new/takes/{take_name}/frame_aligned_videos/downscaled/448" 
[labelled cam 0 thru cam 5, but only take cam0], only for the basketball data, and create a new folder "demo/demo_basketball_expert/" and store these mp4 files.
3. in that same folder, also create a json file which has a map to which specific "action" each vid is doing 
'''

import csv
import json
import subprocess
from pathlib import Path

CLIPS_CSV = "expertedit_eval_benchmark_v1.0/expert_clips.csv"
EGOEXO_ROOT = "dataset_prep/egoexo_new"
OUT_DIR = Path("demo/demo_basketball_expert")

BASKETBALL_ACTIONS = {"Mid-range jump shot", "Mikan layup", "Reverse layup"}

OUT_DIR.mkdir(exist_ok=True)
actions = {}

with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        action = row["action"]
        if action not in BASKETBALL_ACTIONS:
            continue

        take = row["take"]
        timeframe = row["timeframe"]
        if "frames" in timeframe:  # karate rows, no source video
            continue
        start_str, end_str = timeframe.split("-")
        start, end = float(start_str), float(end_str)

        video_dir = Path(EGOEXO_ROOT) / "takes" / take / "frame_aligned_videos" / "downscaled" / "448"
        if not video_dir.is_dir():
            print(f"skipping {take}, no video folder")
            continue

        for cam_video in sorted(video_dir.glob("cam0*.mp4")):
            cam = cam_video.stem
            out_name = f"{take}__{start:.3f}-{end:.3f}__{cam}.mp4"
            out_path = OUT_DIR / out_name

            if not out_path.exists():
                subprocess.run([
                    "ffmpeg", "-loglevel", "error", "-y",
                    "-ss", f"{start:.3f}", "-i", str(cam_video),
                    "-t", f"{end - start:.3f}",
                    "-c:v", "libx264", "-crf", "20", "-an",
                    str(out_path),
                ], check=True)
                print("wrote", out_name)

            actions[out_name] = {"action": action, "take": take, "cam": cam}

with open(OUT_DIR / "actions.json", "w") as f:
    json.dump(actions, f, indent=2)

print(f"done: {len(actions)} clips in {OUT_DIR}/")
