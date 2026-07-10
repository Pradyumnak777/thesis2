import json, subprocess, os, re

DRIVE_VAL = "expertaf_drive/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json"
VIDEO_ROOT = "egoexo_basketball/takes"

data = json.load(open(DRIVE_VAL))

# grab first 3 basketball entries
bball = [r for r in data if "basketball" in r["video1"].lower()][:3]

def ffprobe_fps_frames(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,nb_frames,duration",
           "-of", "default=noprint_wrappers=1", path]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
    fps_raw = re.search(r"r_frame_rate=(\d+)/(\d+)", out)
    fps = int(fps_raw.group(1)) / int(fps_raw.group(2)) if fps_raw else None
    frames = re.search(r"nb_frames=(\d+)", out)
    dur = re.search(r"duration=([\d.]+)", out)
    return fps, int(frames.group(1)) if frames else None, float(dur.group(1)) if dur else None

for row in bball:
    take = row["video1"]
    v1_range = json.loads(row["video1_range"])
    take_dir = os.path.join(VIDEO_ROOT, take, "frame_aligned_videos")
    print(f"\n--- {take} ---")
    print(f"  video1_range: {v1_range}  (span={v1_range[1]-v1_range[0]} frames)")
    if not os.path.isdir(take_dir):
        print(f"  NOT FOUND on disk: {take_dir}")
        continue
    for fname in sorted(os.listdir(take_dir)):
        if not fname.endswith(".mp4"):
            continue
        fpath = os.path.join(take_dir, fname)
        fps, nframes, dur = ffprobe_fps_frames(fpath)
        clip_dur = (v1_range[1] - v1_range[0]) / fps if fps else "?"
        in_range = "OK" if nframes and v1_range[1] <= nframes else "OUT OF RANGE"
        print(f"  {fname:<35} fps={fps}  total_frames={nframes}  dur={dur:.1f}s  clip={clip_dur:.2f}s  [{in_range}]")