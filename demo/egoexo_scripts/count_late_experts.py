'''
count late-expert takes (and narration coverage) across the four
basketball/soccer techniques used in ExpertEdit's Ego-Exo4D training set
'''

import json, re
from pathlib import Path

ROOT = Path("demo/egoexo4d_v2")
TARGET_PROFICIENCY = "Late Expert"

# NOTE: only "Mid-range jumpshot" is a string confirmed from find_late_experts.py.
# The other three are guesses based on that naming pattern -- verify against the
# "candidate task_names" printout below before trusting these counts.
TECHNIQUES = {
    "Mikan layup":        "Basketball Drills - Mikan Layup",
    "Reverse layup":      "Basketball Drills - Reverse Layup",
    "Mid-range jumpshot": "Basketball Drills - Mid-Range Jump Shooting",
    "Penalty kick":       "Soccer Drills - Penalty Kick",
}


def load_proficiency_and_narrations():
    prof = {}
    for split in ("train", "val"):
        p = ROOT / "annotations" / f"proficiency_demonstrator_{split}.json"
        for a in json.load(open(p))["annotations"]:
            prof[a["take_uid"]] = a["proficiency_score"]

    narr = {}
    for split in ("train", "val"):
        p = ROOT / "annotations" / f"atomic_descriptions_{split}.json"
        d = json.load(open(p))
        ann = d.get("annotations", d)
        if isinstance(ann, dict):
            narr.update(ann)
        else:
            for a in ann:
                narr.setdefault(a["take_uid"], []).append(a)

    return prof, narr


def main():
    takes = json.load(open(ROOT / "takes.json"))
    prof, narr = load_proficiency_and_narrations()

    # sanity check: print candidate task_names so you can verify/fix the guessed
    # strings above before trusting the counts below
    all_task_names = sorted(set(t.get("task_name") for t in takes))
    candidates = [n for n in all_task_names if n and re.search(r"layup|jump|shot|kick|penalty", n, re.I)]
    print("=== candidate task_names found in this dataset (verify TECHNIQUES dict against these) ===")
    for n in candidates:
        print(f"  {n!r}")
    print()

    print(f"=== Late Expert take counts (proficiency == {TARGET_PROFICIENCY!r}) ===")
    total_takes, total_narrated = 0, 0
    for label, task_name in TECHNIQUES.items():
        js = [t for t in takes if t.get("task_name") == task_name]
        labelled = [t for t in js if t["take_uid"] in prof]
        sel = [t for t in labelled if prof[t["take_uid"]] == TARGET_PROFICIENCY]
        covered = [t for t in sel if t["take_uid"] in narr]

        total_takes += len(sel)
        total_narrated += len(covered)

        flag = "" if task_name in all_task_names else "  <-- task_name NOT FOUND in dataset, check spelling above"
        print(f"{label:20s} ({task_name!r}): {len(sel)} takes, {len(covered)} have narrations{flag}")

    print(f"\nTOTAL: {total_takes} late-expert takes, {total_narrated} with narration coverage")


if __name__ == "__main__":
    main()