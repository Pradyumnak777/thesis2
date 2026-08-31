#verifies the splice actually happened as intended
import joblib
import numpy as np
from pathlib import Path

ACTOR_DIR = Path("demo/arch/inference_test/learner_exo")

original = joblib.load(ACTOR_DIR / "wham_output_selected.pkl")
edited   = joblib.load(ACTOR_DIR / "edited_motion_smpl.pkl")

orig_body = original['pose_world'][:, 3:66]
edit_body = edited['pose_world'][:, 3:66]

s, e = edited['edit_start_frame'], edited['edit_end_frame']
mask = edited['edit_mask']

outside_diff = np.abs(edit_body[:s] - orig_body[:s]).max() if s > 0 else 0.0
outside_diff = max(outside_diff, np.abs(edit_body[e:] - orig_body[e:]).max() if e < len(orig_body) else 0.0)

window_orig = orig_body[s:e]
window_edit = edit_body[s:e]
unmasked_diff = np.abs(window_edit[~mask] - window_orig[~mask]).max()
masked_diff   = np.abs(window_edit[mask]  - window_orig[mask]).mean()

print(f"outside the 90-frame window entirely -- should be exactly 0.0: {outside_diff:.8f}")
print(f"inside window, UNMASKED frames vs learner -- should be exactly 0.0: {unmasked_diff:.8f}")
print(f"inside window, MASKED frames vs learner -- should be > 0 if infiller changed anything: {masked_diff:.6f}")
print(f"masked frame count: {mask.sum()}/{len(mask)}")