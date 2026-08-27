'''
reorganize basketball_expert_smpl/ so each clip's folder lives under its
action's own subfolder, using demo_basketball_expert/actions.json for the
clip -> action mapping.

before: basketball_expert_smpl/<clip_name>/wham_output.pkl, etc.
after:  basketball_expert_smpl/<action>/<clip_name>/wham_output.pkl, etc.
'''
import json
import shutil
from pathlib import Path

SMPL_ROOT = Path("demo/basketball_expert_smpl")
ACTIONS_JSON = Path("demo/demo_basketball_expert/actions.json")

with open(ACTIONS_JSON) as f:
    actions = json.load(f)

# actions.json keys are "<clip_name>.mp4"; smpl folders are named "<clip_name>"
clip_to_action = {Path(mp4_name).stem: meta["action"] for mp4_name, meta in actions.items()}

clip_dirs = sorted(d for d in SMPL_ROOT.iterdir() if d.is_dir())
print(f"found {len(clip_dirs)} clip folders under {SMPL_ROOT}/")

moved, skipped, missing = 0, 0, 0

for clip_dir in clip_dirs:
    action = clip_to_action.get(clip_dir.name)
    if action is None:
        print(f"no action found in actions.json for: {clip_dir.name}, skipping")
        missing += 1
        continue

    action_dir = SMPL_ROOT / action
    dest = action_dir / clip_dir.name

    if dest.exists():
        print(f"already moved, skipping: {clip_dir.name}")
        skipped += 1
        continue

    action_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(clip_dir), str(dest))
    print(f"moved {clip_dir.name} -> {action}/")
    moved += 1

print(f"done: {moved} moved, {skipped} already in place, {missing} had no action match")