'''
retargeted unitree g1 motion is at: "demo/GMR/g1_retargeted_expert.pkl"

bringing that into the mujoco simulator..
'''
import mujoco
import numpy as np


def mujoco_model():
    xml = """
    
    """
    
    model = mujoco.MjModel.from_xml_string(xml)
    return model

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    #checking
    try:
        mujoco.MjModel.from_xml_string('<mujoco/>')
        print("MuJoCo installed and verified successfully.")
    except Exception as e:
        print(f"Installation check failed: {e}")