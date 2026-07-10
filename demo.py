#so i have the pose data, expert commentary, etc of a learner video and an expert video.
#goal- use retargeting (like SOMA does for meshes) to warp the expert's motion onto the
#learner's own bone lengths, so the "visual feedback" shows the correct move on your own body
#just doing retargeting here, nothing else yet

import json, glob, os
import numpy as np
import debugpy
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()


#coco-17 joints, matches egoexo4d ego_pose annotation format
JOINTS = ["nose","left-eye","right-eye","left-ear","right-ear",
          "left-shoulder","right-shoulder","left-elbow","right-elbow",
          "left-wrist","right-wrist","left-hip","right-hip",
          "left-knee","right-knee","left-ankle","right-ankle"]
JIDX = {n:i for i,n in enumerate(JOINTS)}
NJ = len(JOINTS)

#which joints connect to which, just for drawing
EDGES = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),
         (5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16)]

#parent of each joint in the skeleton tree, need this to walk outward from the root
#and rescale bone by bone. -1 = root (mid hip), -2 = mid shoulder
PARENT = {
    JIDX["left-hip"]:-1, JIDX["right-hip"]:-1,
    JIDX["left-knee"]:JIDX["left-hip"], JIDX["right-knee"]:JIDX["right-hip"],
    JIDX["left-ankle"]:JIDX["left-knee"], JIDX["right-ankle"]:JIDX["right-knee"],
    JIDX["left-shoulder"]:-1, JIDX["right-shoulder"]:-1,
    JIDX["left-elbow"]:JIDX["left-shoulder"], JIDX["right-elbow"]:JIDX["right-shoulder"],
    JIDX["left-wrist"]:JIDX["left-elbow"], JIDX["right-wrist"]:JIDX["right-elbow"],
    JIDX["nose"]:-2,
    JIDX["left-eye"]:JIDX["nose"], JIDX["right-eye"]:JIDX["nose"],
    JIDX["left-ear"]:JIDX["nose"], JIDX["right-ear"]:JIDX["nose"],
}
#parents always get placed before children
ORDER = [JIDX[n] for n in ["left-hip","right-hip","left-knee","right-knee","left-ankle","right-ankle",
          "left-shoulder","right-shoulder","left-elbow","right-elbow","left-wrist","right-wrist",
          "nose","left-eye","right-eye","left-ear","right-ear"]]

ROOT = "egoexo_survey"
DRIVE = "expertaf_drive/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json"


#grab the 3d body keypoints for one take, returns frame numbers + poses array
def load_pose(uid):
    hits = glob.glob(f"{ROOT}/annotations/ego_pose/*/body/annotation/{uid}.json")
    if not hits:
        return None, None
    raw = json.load(open(hits[0]))
    frames = sorted(int(k) for k in raw if str(k).lstrip("-").isdigit())
    poses = []
    for k in frames:
        ann = raw[str(k)]
        ann = ann[0] if isinstance(ann, list) else ann
        a3d = ann.get("annotation3D") or {}
        p = np.full((NJ,3), np.nan)
        for jname, jv in a3d.items():
            if jname in JIDX:
                p[JIDX[jname]] = [jv["x"], jv["y"], jv["z"]]
        poses.append(p)
    return np.array(frames), np.stack(poses)


#some joints are missing per frame, just interpolate over time to fill them in
def fill_gaps(poses):
    out = poses.copy()
    t = np.arange(len(poses))
    for j in range(NJ):
        for c in range(3):
            v = out[:,j,c]
            ok = ~np.isnan(v)
            if ok.sum() == 0:
                continue
            out[:,j,c] = np.interp(t, t[ok], v[ok])
    return out


#center pose at mid hip, rotate so hips/shoulders face a consistent way, scale by torso size
#need this so two different people's bone lengths are actually comparable
def normalize(poses):
    lhip, rhip = poses[:,JIDX["left-hip"]], poses[:,JIDX["right-hip"]]
    lsho, rsho = poses[:,JIDX["left-shoulder"]], poses[:,JIDX["right-shoulder"]]
    midhip, midsho = (lhip+rhip)/2, (lsho+rsho)/2
    scale = np.nanmedian(np.linalg.norm(midsho-midhip, axis=1))
    out = np.empty_like(poses)
    for t in range(len(poses)):
        up = midsho[t]-midhip[t]; up = up/(np.linalg.norm(up)+1e-9)
        x = rhip[t]-lhip[t]; x = x-up*(x@up); x = x/(np.linalg.norm(x)+1e-9)
        z = np.cross(x, up)
        R = np.stack([x, up, z])
        out[t] = (poses[t]-midhip[t]) @ R.T / scale
    return out


#measure each bone length off the learner's own (normalized) skeleton
def get_bone_lengths(norm_poses):
    lens = {}
    for j in ORDER:
        par = PARENT[j]
        if par == -1:
            d = np.linalg.norm(norm_poses[:,j], axis=1)
        elif par == -2:
            midsho = (norm_poses[:,JIDX["left-shoulder"]]+norm_poses[:,JIDX["right-shoulder"]])/2
            d = np.linalg.norm(norm_poses[:,j]-midsho, axis=1)
        else:
            d = np.linalg.norm(norm_poses[:,j]-norm_poses[:,par], axis=1)
        lens[j] = np.nanmedian(d)
    return lens


#the actual retargeting step. keep the expert's joint directions (the "how they move")
#but rebuild every bone using the learner's bone lengths instead of the expert's.
#so this is the expert's motion, but performed by a body shaped like the learner's
def retarget(expert_norm, learner_lens):
    out = np.full_like(expert_norm, np.nan)
    for t in range(len(expert_norm)):
        src = expert_norm[t]
        src_midsho = (src[JIDX["left-shoulder"]]+src[JIDX["right-shoulder"]])/2
        placed = {}
        for j in ORDER:
            par = PARENT[j]
            if par == -1:
                p_new, p_src = np.zeros(3), np.zeros(3)
            elif par == -2:
                p_new = (placed[JIDX["left-shoulder"]]+placed[JIDX["right-shoulder"]])/2
                p_src = src_midsho
            else:
                p_new, p_src = placed[par], src[par]
            d = src[j]-p_src
            d = d/(np.linalg.norm(d)+1e-9)
            placed[j] = p_new + d*learner_lens[j]
        for j,v in placed.items():
            out[t,j] = v
    return out


#draw one skeleton frame
def draw(ax, pose, color):
    for a,b in EDGES:
        if np.isnan(pose[a]).any() or np.isnan(pose[b]).any():
            continue
        ax.plot([pose[a,0],pose[b,0]], [pose[a,2],pose[b,2]], [pose[a,1],pose[b,1]], c=color, lw=2)


#side by side animation - learner's own motion vs expert's motion retargeted onto the learner's body
def render(learner, expert_rt, out_path, fps=12):
    T = min(len(learner), len(expert_rt))
    allp = np.concatenate([learner, expert_rt]).reshape(-1,3)
    allp = allp[~np.isnan(allp).any(1)]
    lo, hi = allp.min(0), allp.max(0)
    mid, rng = (lo+hi)/2, (hi-lo).max()*0.55

    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(1,2,1, projection="3d")
    ax2 = fig.add_subplot(1,2,2, projection="3d")

    def update(t):
        ax1.cla(); ax2.cla()
        for ax, title in [(ax1,"learner"), (ax2,"expert (retargeted to learner)")]:
            ax.set_title(title)
            ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[2]-rng, mid[2]+rng); ax.set_zlim(mid[1]-rng, mid[1]+rng)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=12, azim=-70)
        draw(ax1, learner[t], "tab:blue")
        draw(ax2, expert_rt[t], "tab:green")

    anim = FuncAnimation(fig, update, frames=T)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("saved", out_path)


#quick fake pair so the script actually runs even without the real dataset around
def synthetic_pair():
    def body(T, arm_amp, height):
        ts = np.linspace(0, 2*np.pi, T)
        P = np.zeros((T,NJ,3))
        for t in range(T):
            y = height
            P[t,JIDX["left-hip"]] = [-0.12*height, y, 0]
            P[t,JIDX["right-hip"]] = [0.12*height, y, 0]
            sy = y+0.55*height
            P[t,JIDX["left-shoulder"]] = [-0.20*height, sy, 0]
            P[t,JIDX["right-shoulder"]] = [0.20*height, sy, 0]
            P[t,JIDX["nose"]] = [0, sy+0.25*height, 0.05]
            for s,e in [("left-eye",-0.03),("right-eye",0.03)]:
                P[t,JIDX[s]] = P[t,JIDX["nose"]]+[e,0.03,0]
            for s,e in [("left-ear",-0.07),("right-ear",0.07)]:
                P[t,JIDX[s]] = P[t,JIDX["nose"]]+[e,0,-0.03]
            ang = arm_amp*(0.5-0.5*np.cos(ts[t]))
            for side,sgn in [("left",-1),("right",1)]:
                sh = P[t,JIDX[f"{side}-shoulder"]]
                d = np.array([sgn*np.sin(0.3), -np.cos(ang), np.sin(ang)*0.6]); d = d/np.linalg.norm(d)
                el = sh+d*0.30*height
                P[t,JIDX[f"{side}-elbow"]] = el
                P[t,JIDX[f"{side}-wrist"]] = el+d*0.28*height
            for side in ["left","right"]:
                hip = P[t,JIDX[f"{side}-hip"]]
                P[t,JIDX[f"{side}-knee"]] = [hip[0], hip[1]-0.5*height, 0]
                P[t,JIDX[f"{side}-ankle"]] = [hip[0], hip[1]-1.0*height, 0]
        return P
    #learner has a short arm swing and a smaller body, expert swings way higher and is bigger
    return body(60, arm_amp=0.5, height=0.9), body(60, arm_amp=2.4, height=1.3)


if __name__ == "__main__":
    if os.path.isdir(ROOT):
        #real data path - grab the first usable pair from the drive json
        takes = json.load(open(f"{ROOT}/takes.json"))
        name2uid = {t["take_name"]:t["take_uid"] for t in takes}
        drive = json.load(open(DRIVE))

        learner_raw = expert_raw = None
        for row in drive:
            l_uid = name2uid.get(row["video1"]) #learner
            e_uid = name2uid.get(row["video2"]) #expert
            if not (l_uid and e_uid):
                continue
            lf, lp = load_pose(l_uid)
            ef, ep = load_pose(e_uid)
            if lf is None or ef is None:
                continue
            
            #above code searches for usable pair- many pairs strangely dont have pose data in my setup..
            
            lr = json.loads(row["video1_range"]); er = json.loads(row["video2_range"])
            lmask = (lf>=lr[0])&(lf<=lr[1])
            emask = (ef>=er[0])&(ef<=er[1])
            if lmask.sum()<15 or emask.sum()<15:
                continue
            learner_raw, expert_raw = lp[lmask], ep[emask]
            print("using pair:", row["video1"], "vs", row["video2"])
            print("commentary:", row["text"])
            break
        if learner_raw is None:
            raise SystemExit("no usable pair found in the drive json")
    else:
        print("no egoexo_survey dir found, using synthetic pair to test the pipeline")
        learner_raw, expert_raw = synthetic_pair()

    learner = normalize(fill_gaps(learner_raw))
    expert = normalize(fill_gaps(expert_raw))

    #this is the retargeting - rebuild expert's motion using learner's bone lengths
    expert_retargeted = retarget(expert, get_bone_lengths(learner))

    render(learner, expert_retargeted, "retarget_demo.gif")