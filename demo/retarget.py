'''
take SMPL pose motion and retarget it onto a skeleton of a robot (which likely has different DOFs)

step 1: recover 3D joint positions from SMPL parameters and sanity-check them
'''
import matplotlib
matplotlib.use("Agg") 

import shutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import numpy as np

# chumpy (pulled in transitively by smplx/pickle when loading legacy SMPL .pkl
# files) still does `from numpy import bool, int, float, ...` aliases numpy
# removed in 1.24+. Restore them before chumpy is imported anywhere below.
for _name, _builtin in [("bool", bool), ("int", int), ("float", float),
                         ("complex", complex), ("object", object),
                         ("str", str), ("unicode", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import joblib
import torch
import smplx

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger attach on port 5678...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")


SMPL_MODEL_DIR = "/scratch/pbk5339/thesis_new/WHAM/dataset/body_models"

# SMPL's 24-joint kinematic tree: parent[i] is the parent joint of joint i (-1 = root)
# NOTE: verify this against body.parents at runtime (done in __main__ below) before trusting it
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                          9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def smpl_joints_from_wham(pose_info, model_dir=SMPL_MODEL_DIR, use_world=True):
    """Run SMPL forward kinematics to turn pose parameters into 3D joint positions.

    Returns (N, 24, 3) float32 array of joint positions in metres.
    """
    pose  = pose_info["pose_world"]  if use_world else pose_info["pose"]
    trans = pose_info["trans_world"] if use_world else pose_info["trans"]
    betas = pose_info["betas"]
    N = pose.shape[0]

    # WHAM's betas are estimated per-frame but describe a single body;
    # averaging gives one stable shape for the whole clip.
    betas_mean = torch.from_numpy(betas.astype(np.float32)).mean(dim=0, keepdim=True)
    betas_mean = betas_mean.expand(N, -1)

    pose  = torch.from_numpy(pose.astype(np.float32))    # (N, 72) axis-angle
    trans = torch.from_numpy(trans.astype(np.float32))   # (N, 3)

    body = smplx.create(model_dir, model_type="smpl", gender="neutral", batch_size=N)

    with torch.no_grad():
        out = body(
            global_orient=pose[:, :3],   # joint 0: orientation of the whole body
            body_pose=pose[:, 3:],       # joints 1..23, relative rotations
            betas=betas_mean,
            transl=trans,
        )

    # smplx returns 45 joints for SMPL (24 body + 21 extra landmarks); keep the body ones
    return out.joints[:, :24].numpy(), body.parents.numpy()


def animate_skeleton(joints, parents, fps=30, save_path="smpl_motion.mp4"):
    """joints: (N, J, 3). Draws one line segment per bone, saved to disk (no display)."""
    bones = [(p, i) for i, p in enumerate(parents) if p >= 0]
    N = joints.shape[0]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    lines = [ax.plot([], [], [], "-o", lw=2, ms=3, color="tab:blue")[0] for _ in bones]

    flat = joints.reshape(-1, 3)
    center = flat.mean(axis=0)
    radius = np.abs(flat - center).max()
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - radius, c + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    def update(t):
        for line, (p, c) in zip(lines, bones):
            seg = joints[t, [p, c]]          # (2, 3): the bone's two endpoints
            line.set_data(seg[:, 0], seg[:, 2]) # screen x <- data x, screen y <- data z
            line.set_3d_properties(seg[:, 1]) # screen z (vertical) <- data y
        ax.set_title(f"frame {t}/{N-1}")
        return lines

    anim = FuncAnimation(fig, update, frames=N, interval=1000 / fps, blit=False)

    # ffmpeg may not be installed on the cluster; fall back to GIF (pillow ships with matplotlib)
    if save_path.endswith(".mp4") and shutil.which("ffmpeg") is None:
        save_path = save_path[:-4] + ".gif"
        print(f"ffmpeg not found, saving as GIF instead: {save_path}")

    writer = FFMpegWriter(fps=fps) if save_path.endswith(".mp4") else PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"saved animation to {save_path}")
    return save_path


if __name__ == "__main__":

    pose_info = joblib.load("pose_info.joblib")
    joints, parents = smpl_joints_from_wham(pose_info)

    # sanity checks before trusting this array for retargeting
    print("joints:", joints.shape)                      # expect (119, 24, 3)
    print("parents matches hardcoded SMPL_PARENTS:", np.array_equal(parents, SMPL_PARENTS))

    print("frame 0 pelvis:", joints[0, 0])
    flat = joints.reshape(-1, 3)
    print("bbox min:", flat.min(axis=0))
    print("bbox max:", flat.max(axis=0))

    # which axis is "up"? the axis where head (15) differs most from ankles (7, 8)
    head  = joints[:, 15].mean(axis=0)
    ankle = joints[:, [7, 8]].mean(axis=(0, 1))
    print("head - ankle per axis (x, y, z):", head - ankle)

    animate_skeleton(joints, parents, fps=30, save_path="smpl_motion.mp4")