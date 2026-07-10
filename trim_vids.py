import json, os, glob, subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

VIDEO_ROOT = "egoexo_combined/takes"
TAKES      = "egoexo_survey/takes.json"
OUT_ROOT   = "dataset/clips"
STREAM     = "exo"            # "ego" (aria RGB) or "exo" (best_exo cam)

CLIPS = json.load(open("dataset/clips_to_extract.json"))
best_exo = {t["take_name"]: t.get("best_exo") for t in json.load(open(TAKES))}

def stream_path(take):
    d = os.path.join(VIDEO_ROOT, take, "frame_aligned_videos", "downscaled", "448")
    if STREAM == "ego":
        hits = glob.glob(os.path.join(d, "aria*_214-1.mp4"))   # 214-1 = RGB cam
        return hits[0] if hits else None
    be = best_exo.get(take)
    p = os.path.join(d, f"{be}.mp4")
    return p if be and os.path.exists(p) else None

def extract(clip):
    cid, a, b = clip["clip_id"], clip["start"], clip["end"]
    out_dir = os.path.join(OUT_ROOT, STREAM)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, cid + ".mp4")
    if os.path.exists(out):           return ("skip", cid)
    src = stream_path(clip["take"])
    if not src:                       return ("missing_src", cid)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-vf", f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS",
           "-an", out]
    try:
        subprocess.run(cmd, check=True)
        return ("ok", cid)
    except subprocess.CalledProcessError:
        return ("fail", cid)

res = Counter()
with ProcessPoolExecutor(max_workers=4) as ex:
    for i, (status, cid) in enumerate(ex.map(extract, CLIPS)):
        res[status] += 1
        if status in ("fail", "missing_src"):
            print(f"[{i+1}] {status}: {cid}")
        elif i % 10 == 0:
            print(f"[{i+1}/{len(CLIPS)}] {status}: {cid[-40:]}")
print(dict(res))