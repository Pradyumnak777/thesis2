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

'''
below is fix for some pip error
'''

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


if __name__ == "__main__":
    