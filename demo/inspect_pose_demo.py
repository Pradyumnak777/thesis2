'''
inspect the WHAM output, get pose in accordance with GMR requirement for retargeting
'''
import joblib
import pickle
import numpy as np
'''
some fix
'''
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'unicode'):
    np.unicode = str


from scipy.spatial.transform import Rotation as R
import smplx
import debugpy
import torch
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger attach on port 5678...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

# BASE = "demo/basketball_expert_smpl/sfu_basketball_02_19__10.233-12.233__cam01/sfu_basketball_02_19__10.233-12.233__cam01"

model = smplx.create(
    model_path="demo/GMR/assets/body_models",
    model_type="smpl",
    gender="neutral"
)

def basic_heuristic(BASE):#base is the path to the smpl directory, NOT the smpl file itself!
    wham_output      = joblib.load(f"{BASE}/wham_output.pkl")
    slam_results     = joblib.load(f"{BASE}/slam_results.pth")
    tracking_results = joblib.load(f"{BASE}/tracking_results.pth")

    track_ids = list(wham_output.keys())
    people = list(wham_output.values())

    if len(people) == 1:
        best_id = track_ids[0]
    else:
        '''
        some heuristic needs to be defined
        -> for now relying on pelvis movement
        '''
        pelvis_trans = []
        for person in people:
            trans_world = person["trans_world"]  # (frames, 3)
            diffs = np.diff(trans_world, axis=0)
            step_dist = np.linalg.norm(diffs, axis=1)
            pelvis_trans.append(step_dist.sum())
        best_id = track_ids[int(np.argmax(pelvis_trans))]

    return best_id, wham_output


def jumphot_heuristic(BASE):#base is the path to the smpl directory, NOT the smpl file itself!
    wham_output      = joblib.load(f"{BASE}/wham_output.pkl")
    slam_results     = joblib.load(f"{BASE}/slam_results.pth")
    tracking_results = joblib.load(f"{BASE}/tracking_results.pth")

    track_ids = list(wham_output.keys())
    people = list(wham_output.values())

    if len(people) == 1:
        best_id = track_ids[0]
    else:
        '''
        some heuristic needs to be defined
        -> jumpshot: pelvis in y axis(?)
        '''
        pelvis_trans = []
        for person in people:
            trans_world = person["trans_world"]  # (frames, 3)
            vertical_trans = trans_world[:,1] # y axis
            
            #find difference between min and max
            jump_displacement = np.percentile(vertical_trans, 95) - np.percentile(vertical_trans, 5)

            pelvis_trans.append(jump_displacement.sum())
        
        best_id = track_ids[int(np.argmax(pelvis_trans))]

    return best_id, wham_output



    # #get the first person pose information (over all frames)
    # pose_info = wham_output[0]
    # # joblib.dump(pose_info, "pose_info.joblib")
    
    # pose_world = pose_info["pose_world"]
    # trans_world = pose_info["trans_world"]
    # betas = pose_info["betas"]
    
    # N = pose_world.shape[0]
    
    # if betas.ndim == 2:
    #     betas_mean = betas.mean(axis=0)
    # else:
    #     betas_mean = betas
        

    # print("debug to inspect!")