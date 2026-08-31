# builds the final late-expert jumpshot take list: proficiency = Late Expert,
# and actually covered by atomic_descriptions (checked directly against the file)

import json, re
from pathlib import Path

ROOT = Path("demo/egoexo4d_v2")
TASK_NAME = "Basketball Drills - Mid-Range Jump Shooting"
TARGET_PROFICIENCY = "Late Expert"
PREFERRED_CAM = "cam01"

OUT_DIR = Path(__file__).resolve().parent
UID_FILE = OUT_DIR / "jumpshot_late_expert_uids.txt"
META_FILE = OUT_DIR / "jumpshot_late_expert_takes.json"

takes = json.load(open(ROOT / "takes.json"))

prof, split_of = {}, {}
for split in ("train", "val"):
    p = ROOT / "annotations" / f"proficiency_demonstrator_{split}.json"
    for a in json.load(open(p))["annotations"]:
        prof[a["take_uid"]] = a["proficiency_score"]
        split_of[a["take_uid"]] = split

narr = {}  # take_uid -> narration entries
for split in ("train", "val"):
    p = ROOT / "annotations" / f"atomic_descriptions_{split}.json"
    d = json.load(open(p))
    ann = d.get("annotations", d)
    if isinstance(ann, dict):
        narr.update(ann)
    else:
        for a in ann:
            narr.setdefault(a["take_uid"], []).append(a)

js = [t for t in takes if t.get("task_name") == TASK_NAME]
labelled = [t for t in js if t["take_uid"] in prof]
sel = [t for t in labelled if prof[t["take_uid"]] == TARGET_PROFICIENCY]
covered = [t for t in sel if t["take_uid"] in narr]
print(f"{TASK_NAME} / {TARGET_PROFICIENCY}: {len(sel)} takes, {len(covered)} have narrations")


def exo_cams(t):
    return sorted(c for c in t.get("frame_aligned_videos", {}) if re.fullmatch(r"cam\d+", c))


def pick_cam(t, cams):
    return PREFERRED_CAM if PREFERRED_CAM in cams else t.get("best_exo")


meta = {}
for t in covered:
    cams = exo_cams(t)
    cam = pick_cam(t, cams)
    if cam is None:
        continue
    meta[t["take_name"]] = {
        "take_uid": t["take_uid"],
        "root_dir": t["root_dir"],
        "chosen_cam": cam,
        "used_fallback": cam != PREFERRED_CAM,
        "duration_sec": t.get("duration_sec"),
        "task_start_sec": t.get("task_start_sec"),
        "task_end_sec": t.get("task_end_sec"),
        "split": split_of[t["take_uid"]],
    }

fallback_count = sum(m["used_fallback"] for m in meta.values())
print(f"cam: {len(meta)-fallback_count} used cam01, {fallback_count} fell back to best_exo")

UID_FILE.write_text("\n".join(m["take_uid"] for m in meta.values()) + "\n")
META_FILE.write_text(json.dumps(meta, indent=2))
print(f"wrote {len(meta)} takes -> {UID_FILE}, {META_FILE}")