"""
"dataset_prep/egoexo_new/takes/iiith_soccer_002_6/trajectory" check inside this directory-

1. "dataset_prep/egoexo_new/takes/iiith_soccer_002_6/trajectory/semidense_observations.csv.gz"
2. "dataset_prep/egoexo_new/takes/iiith_soccer_002_6/trajectory/semidense_points.csv.gz"

the above 2 files are the point clouds for a specific take. inspect them..
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ["OPEN3D_CPU_RENDERING"] = "true"

import pandas as pd
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

TAKE_DIR = "dataset_prep/egoexo_new/takes/iiith_soccer_002_6/trajectory"

closed_loop_path = f"{TAKE_DIR}/closed_loop_trajectory.csv"
closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

points_df = pd.read_csv(f"{TAKE_DIR}/semidense_points.csv.gz", compression="gzip")
clean_df = points_df[(points_df["inv_dist_std"] < 0.005) & (points_df["dist_std"] < 0.15)]

TARGET_FRAME_IDX = 500
RENDER_SOURCE = "cpf"   # options: "cpf" (recommended), "device", "rgb_camera", "custom_third_person"

selected_pose = closed_loop_traj[TARGET_FRAME_IDX]
T_world_device = selected_pose.transform_world_device.to_matrix()

# --- sanity check: is "Device" actually upright? (it usually isn't -- Device is
# canonically camera-slam-left, which is physically canted on the glasses) ---
traj_df = pd.read_csv(closed_loop_path)
row = traj_df.iloc[TARGET_FRAME_IDX]
g_world = row[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].to_numpy(dtype=float)
g_dev = T_world_device[:3, :3].T @ g_world
print("gravity in device frame:", g_dev, "-- if this isn't close to [0, +9.8, 0], "
      "'device' is not upright and RENDER_SOURCE='device' will look rolled/tilted")

if RENDER_SOURCE == "device":
    # NOTE: this is literally camera-slam-left's frame, canted on the glasses --
    # not a symmetric "midpoint" of the aria cams. Expect a tilted view.
    T_world_camera = T_world_device

elif RENDER_SOURCE == "cpf":
    # CPF = Central Pupil Frame, the human-eye-centric frame -- this is what
    # "device" was assumed to be. Use this for an upright, head-like view.
    vrs_file_path = f"dataset_prep/egoexo_new/takes/iiith_soccer_002_6/video.vrs"
    vrs_provider = data_provider.create_vrs_data_provider(vrs_file_path)
    device_calib = vrs_provider.get_device_calibration()
    T_device_cpf = device_calib.get_transform_device_cpf().to_matrix()
    T_world_camera = T_world_device @ T_device_cpf

elif RENDER_SOURCE == "rgb_camera":
    vrs_file_path = f"dataset_prep/egoexo_new/takes/iiith_soccer_002_6/video.vrs"
    vrs_provider = data_provider.create_vrs_data_provider(vrs_file_path)
    rgb_stream_id = StreamId("214-1")
    rgb_label = vrs_provider.get_label_from_stream_id(rgb_stream_id)
    device_calib = vrs_provider.get_device_calibration()
    rgb_calib = device_calib.get_camera_calib(rgb_label)
    T_device_rgb_camera = rgb_calib.get_transform_device_camera().to_matrix()
    T_world_camera = T_world_device @ T_device_rgb_camera
    # NOTE: Aria's RGB sensor is mounted rotated ~90 deg -- raw frames need a
    # rotation to display upright; the render below won't account for that.

elif RENDER_SOURCE == "custom_third_person":
    # X right, Y down, Z forward (camera convention) -- so "up" is -Y, "back" is -Z
    offset = np.array([
        [1, 0, 0, 0.0],
        [0, 1, 0, -2.0],   # -Y = 2m up
        [0, 0, 1, -3.0],   # -Z = 3m back
        [0, 0, 0, 1.0],
    ])
    T_world_camera = T_world_device @ offset

eye = T_world_camera[:3, 3]
forward = T_world_camera[:3, 2]
up = -T_world_camera[:3, 1]
center = eye + forward

# --- cull to a radius around the camera: without this you're rendering the ENTIRE
# capture's map (all ~50 takes of this session) with no occlusion
xyz = clean_df[["px_world", "py_world", "pz_world"]].to_numpy()
d = np.linalg.norm(xyz - eye, axis=1)
near_df = clean_df[d < 15.0]
print(f"{len(near_df)}/{len(clean_df)} points within 15m of camera")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(near_df[["px_world", "py_world", "pz_world"]].to_numpy())

z_coords = near_df["pz_world"].to_numpy()
z_min, z_max = np.percentile(z_coords, 2), np.percentile(z_coords, 98)
z_norm = (np.clip(z_coords, z_min, z_max) - z_min) / (z_max - z_min + 1e-6)
colors = plt.get_cmap("viridis")(z_norm)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

renderer = rendering.OffscreenRenderer(1920, 1080)
mat = rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 3.0
renderer.scene.add_geometry("soccer_point_cloud", pcd, mat)

vertical_fov = 100.0   # was 60 -- too narrow, crops most of the scene
aspect_ratio = 1920 / 1080
renderer.scene.camera.set_projection(vertical_fov, aspect_ratio, 0.1, 1000.0, rendering.Camera.FovType.Vertical)
renderer.scene.camera.look_at(center, eye, up)

rendered_image = renderer.render_to_image()
output_filename = f"render_frame_{TARGET_FRAME_IDX}_{RENDER_SOURCE}.png"
o3d.io.write_image(output_filename, rendered_image)
print(f"Render saved to: {output_filename}")