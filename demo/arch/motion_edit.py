'''
this is inference visualization. run an actual pose and get the edit pose?
1. load video
3. run wham and get the pose pkl
3. get the operative moment using the formula (pelvis movement)
4. pass thru model
'''
import os
import subprocess
import joblib
from ..inspect_pose_demo import heuristic_pose_selector

VID_PATH = "dataset_prep/dataset_out/val/sfu_basketball012_12-----18-----uniandes_basketball_001_28-----32-----Hands/learner_exo.mp4"
OUT_DIR = "demo/arch/inference_test"
# OUT_DIR = (OUT_DIR / VID_PATH.stem).resolve()

if __name__ == "__main__":
    subprocess.run(
        [
            "python", "demo.py",
            "--video", str(VID_PATH),
            "--output_pth", str(OUT_DIR.resolve()),
            "--save_pkl",
        ],
        cwd=VID_PATH,
        check=True,
    )
    '''
    simple heuristic to select only actor pose
    '''
    best_idx, wham_output = heuristic_pose_selector(str(OUT_DIR))
    selected_person = wham_output[best_idx]
    joblib.dump(selected_person, OUT_DIR / "wham_output_selected.pkl")
    
    '''
    now, run this through the model
    '''