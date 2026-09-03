'''
take an smpl pkl file and visualize it framewise, as a video, and save to disk
'''
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # must be set BEFORE importing pyrender

import joblib
import torch
import smplx
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


import trimesh
import pyrender
import imageio


# SMPL_FILE = "demo/basketball_expert_smpl_v2/sfu_basketball_03_18__rep4__t23.01/wham_output_selected.pkl"
# SMPL_FILE = "demo/arch/recon_test_motion.pkl"
SMPL_FILE = "demo/arch/inference_test_2/learner_exo/edited_motion_smpl.pkl"
# SMPL_FILE = "demo/arch/inference_test_2/learner_exo/input_90frame_smpl.pkl"

OUT_VIDEO = "demo/edited.mp4"
FPS = 30
W, H = 640, 640

def look_at(eye, target, up=np.array([0, 1, 0])): #y axis is up..
    eye, target, up = map(np.asarray, (eye, target, up)) #eye, target and up are now 
    z = (eye - target); z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


if __name__ == "__main__":
    person = joblib.load(SMPL_FILE)

    pose_world = person["pose_world"]    # (N, 72)
    trans_world = person["trans_world"]  # (N, 3)
    betas = person["betas"]
    betas = betas.mean(axis=0) if betas.ndim == 2 else betas

    model = smplx.create(
        model_path="demo/GMR/assets/body_models",
        model_type="smpl",
        gender="neutral",
    )

    N = pose_world.shape[0]
    global_orient = torch.tensor(pose_world[:, :3], dtype=torch.float32)
    body_pose = torch.tensor(pose_world[:, 3:], dtype=torch.float32)
    transl = torch.tensor(trans_world, dtype=torch.float32)
    betas_batch = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).repeat(N, 1)

    with torch.no_grad():
        output = model(global_orient=global_orient, body_pose=body_pose,
                        transl=transl, betas=betas_batch)
    all_verts = output.vertices.numpy()  # (N, 6890, 3)
    faces = model.faces                  # (13776, 3), constant across frames

    # camera framing: fit the whole trajectory's extent, not just one frame
    flat = all_verts.reshape(-1, 3)
    lo, hi = flat.min(0), flat.max(0)
    center = (lo + hi) / 2
    radius = np.linalg.norm(hi - lo) * 0.75

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = look_at(eye=center + np.array([0, radius * 0.3, radius]), target=center)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    frames = []
    for t in range(N):
        mesh = trimesh.Trimesh(vertices=all_verts[t], faces=faces, process=False)
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4]*3)
        scene.add(py_mesh)
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)

        color, _ = renderer.render(scene)
        frames.append(color)

        if (t + 1) % 20 == 0:
            print(f"  rendered {t+1}/{N}")

    renderer.delete()
    imageio.mimwrite(OUT_VIDEO, frames, fps=FPS)
    print("saved", OUT_VIDEO)
