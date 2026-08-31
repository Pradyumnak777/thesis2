import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "egoexo4d_v2"
TASK_NAME = "Basketball Drills - Mid-Range Jump Shooting"
TARGET_PROFICIENCY = "Late Expert"

takes = json.load(open(ROOT / "takes.json"))

prof = {}
for split in ("train", "val"):
    p = ROOT / "annotations" / f"proficiency_demonstrator_{split}.json"
    for a in json.load(open(p))["annotations"]:
        prof[a["take_uid"]] = a["proficiency_score"]

# just find ONE take_uid that's both Late Expert and has an atomic_descriptions entry
candidates = [t for t in takes
              if t.get("task_name") == TASK_NAME
              and prof.get(t["take_uid"]) == TARGET_PROFICIENCY]

for split in ("train", "val"):
    p = ROOT / "annotations" / f"atomic_descriptions_{split}.json"
    d = json.load(open(p))
    ann = d.get("annotations", d)

    for t in candidates:
        uid = t["take_uid"]
        entry = None
        if isinstance(ann, dict):
            entry = ann.get(uid)
        else:
            matches = [a for a in ann if a.get("take_uid") == uid]
            entry = matches if matches else None

        if entry:
            print(f"take_name: {t['take_name']}")
            print(f"take_uid:  {uid}")
            print(f"found in:  atomic_descriptions_{split}.json")
            print(f"top-level type: {type(entry).__name__}")
            print()
            print(json.dumps(entry, indent=2))
            raise SystemExit  # stop at the first hit