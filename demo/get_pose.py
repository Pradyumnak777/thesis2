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

BASE = "smpl_info/expert_exo"

model = smplx.create(
    model_path="GMR/assets/body_models",
    model_type="smpl",
    gender="neutral"
)


if __name__ == "__main__":
    
    wham_output      = joblib.load(f"{BASE}/wham_output.pkl")
    slam_results     = joblib.load(f"{BASE}/slam_results.pth")
    tracking_results = joblib.load(f"{BASE}/tracking_results.pth")

    
    #inspecting the files
    print("\nPKL Type: ", type(wham_output))
    print("\nPTH(slam) Type:", type(slam_results))
    print("\nPTH(tracking) Type:", type(tracking_results))
    
    
    #get the first person pose information (over all frames)
    pose_info = wham_output[0]
    # joblib.dump(pose_info, "pose_info.joblib")
    
    pose_world = pose_info["pose_world"]
    trans_world = pose_info["trans_world"]
    betas = pose_info["betas"]
    
    N = pose_world.shape[0]
    
    if betas.ndim == 2:
        betas_mean = betas.mean(axis=0)
    else:
        betas_mean = betas
        
    '''
    coordinate axes change
    '''
    # R_yup_to_zup = R.from_euler("x", 90, degrees=True)

    # # 1. Transform global root orientation
    # root_orient_rot = R.from_rotvec(pose_world[:, :3])
    # aligned_root_orient = (R_yup_to_zup * root_orient_rot).as_rotvec()

    # # 2. Transform global translation and lift pelvis onto the ground plane
    # aligned_trans = R_yup_to_zup.apply(trans_world)

    # # Elevate pelvis so standing height aligns (~0.78m for G1)
    # min_z = aligned_trans[:, 2].min()
    # aligned_trans[:, 2] += 0.78 - min_z
    
    global_orient = torch.tensor(pose_world[:, :3], dtype=torch.float32)   # Shape: (1, 3)
    body_pose     = torch.tensor(pose_world[:, 3:], dtype=torch.float32)   # Shape: (1, 69)
    transl        = torch.tensor(trans_world, dtype=torch.float32)
    
    #forward. kinematics
    # output.joints has shape (Batch_Size, Num_Joints, 3) -> (1, 24, 3) (or 45 if including extra SMPL markers)
    output = model(
    global_orient=global_orient,
    body_pose=body_pose,
    transl=transl
    )    
    
    '''
    need to know which way pelvis is facing in world view #1
    '''
    pelvis_world = pose_world[0][:3]
    
    rot_matrix = R.from_rotvec(pelvis_world).as_matrix() #converting 'pelvis_world' to rotation matrix
    #in cancnoical frame (before cam or world), SMPL pelvis faces along +Z
    canonical_forward = np.array([0.0, 0.0, 1.0])
    
    #rotate the canonical forward vector into world coordinates
    facing_direction_world = rot_matrix @ canonical_forward
    # print("Facing vector in world (X, Y, Z):", facing_direction_world)
    
    '''
    pelvis facing #2
    '''
    left_hip = output.joints[0, 1].detach().numpy()
    right_hip = output.joints[0, 2].detach().numpy()
    pelvis = output.joints[0, 0].detach().numpy()
    neck = output.joints[0, 12].detach().numpy()

    # 1. UP vector (spine line)
    up_vec = neck - pelvis
    up_vec /= np.linalg.norm(up_vec)

    # 2. LATERAL vector (across hips: left to right)
    hip_vec = right_hip - left_hip
    hip_vec /= np.linalg.norm(hip_vec)

    # 3. FACING vector (perpendicular to both UP and LATERAL)
    # In standard right-handed systems, Cross(Up, Right) points forward
    true_facing = np.cross(up_vec, hip_vec)
    true_facing /= np.linalg.norm(true_facing)

    print("Computed Facing Direction:", true_facing.round(3))
    
    
    '''
    apply transform
    '''
    R_mat = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) #rotation matrix
    R_basis = R.from_matrix(R_mat)
    root_orient_rot = R.from_rotvec(pose_world[:, :3])
    aligned_root_orient = (R_basis * root_orient_rot).as_rotvec()
    aligned_trans = R_basis.apply(trans_world) #applying transform to position vectors too.
    
    #ground elevatoin adjustment
    min_z = aligned_trans[:, 2].min()
    aligned_trans[:, 2] += 0.78 - min_z
        

    gmr_smpl_data = {
        "mocap_frame_rate": np.array(30, dtype=np.int32),
        "gender": "neutral",
        "root_orient": np.ascontiguousarray(
            aligned_root_orient, dtype=np.float32
        ),  # (N, 3)
        "pose_body": np.ascontiguousarray(
            pose_world[:, 3:72], dtype=np.float32
        ),  # (N, 69)
        "trans": np.ascontiguousarray(
            aligned_trans, dtype=np.float32
        ),  # (N, 3)
        "betas": np.ascontiguousarray(
            betas_mean[:10], dtype=np.float32
        ),  # (10,)
    }

    with open("gmr_input_motion_new.pkl", "wb") as f:
        pickle.dump(gmr_smpl_data, f)

    print("Saved gmr_input_motion.pkl successfully!")