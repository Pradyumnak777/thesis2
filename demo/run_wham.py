'''
in demo/ (pwd)

1. read the videos in demo_basketball_expert/
2. run wham for each video
3. inspect its output, and check the number of poses generated
4. if its more than 1, use a simple heuristic to select one
5; the heursitic: "movement"- select the pose that "moves" the most. This could be a term that is a sum of
the pelvis translation across frames.
    a) mid range jump shot- the pelvis moves vertically, shooting arm velocity/acceleration
    b) Mikan Layup- lower body should move?
    c) reverse layup- again lower body and shoulder?

these are some example hehuristics of what could be checked. or simply, translation of the pelvis,
which seems to be a unified heuristic could be better to start off with
'''
import argparse
import os
import joblib
import time
from tqdm import tqdm
import subprocess
from pathlib import Path
from inspect_pose_demo import jumphot_heuristic

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True, help="physical GPU id this shard runs on")
ap.add_argument("--shard-id", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
args = ap.parse_args()

WHAM_ROOT = Path("WHAM")
VIDEO_DIR = Path("demo/basketball_expert_clips/Mid-range jump shot")
OUT_ROOT = Path("demo/basketball_expert_smpl_v2")

OUT_ROOT.mkdir(exist_ok=True)

videos = sorted(VIDEO_DIR.glob("*.mp4"))
videos = videos[args.shard_id::args.num_shards]  # interleaved slice -- this shard's share
print(f"shard {args.shard_id}/{args.num_shards} on gpu {args.gpu}: {len(videos)} clips")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

for video_path in tqdm(videos, desc=f"WHAM[gpu{args.gpu}]"):
    video_path = video_path.resolve()
    out_dir = (OUT_ROOT / video_path.stem).resolve()

    if (out_dir / "wham_output.pkl").exists():
        tqdm.write(f"skipping (already done): {video_path.name}")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    subprocess.run(
        ["python", "demo.py", "--video", str(video_path),
         "--output_pth", str(OUT_ROOT.resolve()), "--save_pkl"],
        cwd=WHAM_ROOT, env=env, check=True,
    )
    wham_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_idx, wham_output = jumphot_heuristic(str(out_dir))
    selected_person = wham_output[best_idx]
    joblib.dump(selected_person, out_dir / "wham_output_selected.pkl")
    select_time = time.perf_counter() - t0

    tqdm.write(f"{video_path.name}: wham {wham_time:.1f}s, select {select_time:.1f}s")

print(f"shard {args.shard_id} done")