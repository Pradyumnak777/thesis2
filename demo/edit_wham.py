'''
in demo/ (pwd)

wham has already been run for every clip under basketball_expert_smpl/;
this just (re)runs the heuristic person-selector over each existing
wham_output.pkl and saves wham_output_selected.pkl. use this to iterate
on the heuristic without re-running wham itself.
'''
import time
import joblib
from pathlib import Path
from inspect_pose_demo import jumphot_heuristic

OUT_ROOT = Path("demo/basketball_expert_smpl/Mid-range jump shot")

result_dirs = sorted(d for d in OUT_ROOT.iterdir() if d.is_dir())
n = len(result_dirs)
print(f"found {n} processed clips in {OUT_ROOT}/")

batch_start = time.perf_counter()
skipped, missing, done = 0, 0, 0

for i, result_dir in enumerate(result_dirs, 1):
    if not (result_dir / "wham_output.pkl").exists():
        print(f"[{i}/{n}] missing wham_output.pkl, skipping: {result_dir.name}")
        missing += 1
        continue

    t0 = time.perf_counter()
    best_idx, wham_output = jumphot_heuristic(str(result_dir))
    selected_person = wham_output[best_idx]
    joblib.dump(selected_person, result_dir / "wham_output_selected.pkl")
    select_time = time.perf_counter() - t0

    elapsed = time.perf_counter() - batch_start
    print(f"[{i}/{n}] {result_dir.name}: select {select_time:.2f}s, "
          f"elapsed {elapsed/60:.1f}min")
    done += 1

print(f"done: {done} selected, {missing} missing wham_output.pkl")