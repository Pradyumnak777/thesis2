'''
retargeted unitree g1 motion is at: "demo/GMR/g1_retargeted_expert.pkl"

bringing that into the mujoco simulator..
'''
import mujoco
import numpy as np
import joblib
import imageio

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger attach on port 5678...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")


def mujoco_model(xml_path):
    #load xml_file
    model = mujoco.MjModel.from_xml_path(xml_path)
    return model

def kinematic_playback(model, data, motion_data, output_path="demo/simulator/g1_mujoco_no-phys.mp4", fps=30):
    q_ref = motion_data["dof_pos"]
    num_frames = q_ref.shape[0]
    
    has_root = ("root_pos" in motion_data) and ("root_rot" in motion_data)
    
    if has_root:
        root_pos = motion_data["root_pos"]
        root_rot = motion_data["root_rot"]
        
        #sm fix- GMR output is [x, y, z, w], reorder columns to [w, x, y, z]:
        root_rot = root_rot[:, [3, 0, 1, 2]]
        
    
    renderer = mujoco.Renderer(model, height=720, width=1280) #offscreen renderer?
    scene_option = mujoco.MjvOption() #visual rendering options(?)
    
    frames = []
    print(f"Rendering {num_frames} frames to {output_path} at {fps} FPS...")
    
    '''
    below loop is for "teleportation"
    info is directly assigned to "data" from the "q_ref" variable.
    
    '''
    for i in range(num_frames):
        if model.nq == 29 + 7 and has_root: #nq is the number of q-coordinates (so body-29 plus 7(orientation and translation))
            #first 3: Root position (X, Y, Z)
            data.qpos[0:3] = root_pos[i]
            #next 4: Root orientation quaternion (W, X, Y, Z) - ts was the oreintation
            data.qpos[3:7] = root_rot[i]
            # Last 29: Actuated joint angles (body joints)
            data.qpos[7:36] = q_ref[i] #reference q posiitons
        elif model.nq == 29:
            # Fixed base model
            data.qpos[:] = q_ref[i]
        else:
            # Slicing fallback (e.g. if qpos contains extra passive joints)
            data.qpos[-29:] = q_ref[i]

        #NOTE: this is set to 0 because in pure playback, the poses are just "teleported" and there's no actual velocity
        #play. In th actual physics based simulation, this velocity would come into play.
        data.qvel[:] = 0.0

        #Compute spatial coordinates from qpos 
        
        # mj_forward updates data.xpos, geom transforms, etc.
        mujoco.mj_forward(model, data)

        #Render frame to pixel buffer 
        renderer.update_scene(data, scene_option=scene_option)
        pixel_array = renderer.render()  # Returns (720, 1280, 3) uint8 numpy array
        frames.append(pixel_array)

        if (i + 1) % 50 == 0 or i == num_frames - 1:
            print(f"Rendered frame {i + 1}/{num_frames}")

    #Save frames to an MP4 file
    print(f"Writing video file to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Video export complete!")

def physics_simulation(model, data, motion_data, output_path="demo/simulator/g1_mujoco_physics.mp4", fps=30, kp=100.0, kd=5.0):
    '''
    Unlike kinematic_playback (teleporting qpos), this drives the robot with a per-joint
    PD tracking controller and lets mj_step integrate real dynamics (gravity, contact,
    actuator torque limits) each physics tick.
    '''
    q_ref = motion_data["dof_pos"]
    num_frames = q_ref.shape[0]

    has_root = ("root_pos" in motion_data) and ("root_rot" in motion_data)
    if has_root:
        root_pos = motion_data["root_pos"]
        root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # GMR [x,y,z,w] -> mujoco [w,x,y,z]

    # GMR only gives us positions, so approximate reference joint velocities by finite difference
    dt_ref = 1.0 / fps #dt is time
    
    '''
    below is essentially dx/dt, where x is position.
    '''
    qdot_ref = np.gradient(q_ref, dt_ref, axis=0) #ts is the reference joint velocities

    joint_ids = model.actuator_trnid[:, 0] #"actuator transmission ID"
    #which joints does the actuator/motor drive? -> generally one motor/actuator per joint
    
    max_torque = model.jnt_actfrcrange[joint_ids, 1] #max torque for that joint

    mujoco.mj_resetData(model, data)
    if has_root:
        data.qpos[0:3] = root_pos[0]
        data.qpos[3:7] = root_rot[0]
        data.qpos[7:36] = q_ref[0]
    else:
        data.qpos[:] = q_ref[0]
        
    #initial condition
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    physics_dt = model.opt.timestep  # 0.002s here, vs. 0.0333s per mocap frame
    substeps = max(1, round(dt_ref / physics_dt))
    
    #for one frame of reference motion, how many physics steps to take??

    renderer = mujoco.Renderer(model, height=720, width=1280)
    scene_option = mujoco.MjvOption()
    frames = []

    print(f"Simulating {num_frames} frames ({substeps} physics steps/frame, dt={physics_dt}) -> {output_path}")

    for i in range(num_frames):
        for _ in range(substeps):
            # qpos root is 7-wide (3 pos + 4 quat), qvel root is 6-wide (3 lin + 3 ang vel) -
            # quaternions have no direct derivative, so nv = nq - 1 whenever there's a free joint.
            if has_root:
                q_now = data.qpos[7:36]
                qd_now = data.qvel[6:35]
            else:
                q_now = data.qpos[:]
                qd_now = data.qvel[:]

            tau = kp * (q_ref[i] - q_now) + kd * (qdot_ref[i] - qd_now)
            data.ctrl[:] = np.clip(tau / max_torque, -1.0, 1.0)

            mujoco.mj_step(model, data)

        renderer.update_scene(data, scene_option=scene_option)
        frames.append(renderer.render())

        if (i + 1) % 50 == 0 or i == num_frames - 1:
            print(f"Simulated frame {i + 1}/{num_frames}")

    print(f"Writing video file to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Physics-based video export complete!")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    #checking
    try:
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print("MuJoCo installed and verified successfully.")
    except Exception as e:
        print(f"Installation check failed: {e}")
        
    model = mujoco_model("demo/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
    data = mujoco.MjData(model)
    #inspect the model/data in debug console..
    
    g1_retargeted = joblib.load("demo/GMR/g1_retargeted_expert.pkl")
    # q_ref = g1_retargeted["dof_pos"]
    # num_frames = q_ref.shape[0]
    
    '''
    1)
    retargeting playback in the simulator (no added physicsi)
    '''
    kinematic_playback(
        model=model, 
        data=data, 
        motion_data=g1_retargeted, 
        # output_path="g1_playback.mp4", 
        fps=30
    )
    
    '''
    2)
    running actual dynamic physics based simulation
    '''
    physics_simulation(
        model=model,
        data=data,
        motion_data=g1_retargeted,
        fps=30,
    )

    