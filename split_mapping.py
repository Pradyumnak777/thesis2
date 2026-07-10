import json, os

DRIVE = "expertaf_drive"
TAKES = "egoexo_survey/takes.json"
OUT   = "dataset"
os.makedirs(OUT, exist_ok=True)

FILES = {
    "train": f"{DRIVE}/pose_dataset_concise_NE_DTW_with_soccer_train_N5.json",
    "val":   f"{DRIVE}/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json",
}

def cat_of(name):
    n = name.lower()
    if "basketball" in n: return "basketball"
    if "bouldering" in n or "climb" in n: return "rock_climbing"
    if "soccer" in n: return "soccer"
    return "other"

unique_clips = {}                       # (take, r0, r1) -> clip_id

for split, path in FILES.items():
    rows_out = []
    for r in json.load(open(path)):
        l_take, l_rng = r["video1"], json.loads(r["video1_range"])
        e_take, e_rng = r["video2"], json.loads(r["video2_range"])
        l_cid = f"{l_take}__{l_rng[0]}_{l_rng[1]}"
        e_cid = f"{e_take}__{e_rng[0]}_{e_rng[1]}"
        unique_clips[(l_take, l_rng[0], l_rng[1])] = l_cid
        unique_clips[(e_take, e_rng[0], e_rng[1])] = e_cid
        rows_out.append({
            "pair_id":       r["id"],
            "split":         split,
            "scenario":      cat_of(l_take),
            "body_part":     r.get("corrected_body_part"),
            "text":          r.get("text"),            # concise target
            "original_text": r.get("original_text"),   # raw commentary
            "label":         r.get("original_label"),  # {summary, needs-improvement, good}
            "learner": {"take": l_take, "range": l_rng, "clip_id": l_cid},
            "expert":  {"take": e_take, "range": e_rng, "clip_id": e_cid},
        })
    json.dump(rows_out, open(f"{OUT}/{split}.json", "w"), indent=1)
    print(f"{split}: {len(rows_out)} rows")

clips = [{"take": t, "start": a, "end": b, "clip_id": cid}
         for (t, a, b), cid in unique_clips.items()]
json.dump(clips, open(f"{OUT}/clips_to_extract.json", "w"), indent=1)
print(f"Unique clips to extract: {len(clips)}")