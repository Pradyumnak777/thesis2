#so i have the pose data, expert commentary, etc of a learner video and an expert video.
#want to see the poses moving, score how similar they are with PA-MPJPE like the paper,
#and line them up in time properly with DTW so the side-by-side shows the SAME moment of the move

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import debugpy
import sys

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


#subtract the hip-center every frame so the body stays put and only the joints move.
#this only kills the walking-around drift, not rotation or size
def center_root(poses):
    lhip, rhip = poses[:,JIDX["left-hip"]], poses[:,JIDX["right-hip"]]
    midhip = (lhip+rhip)/2
    return poses - midhip[:,None,:]


#the two people are different sizes (or their 3d reconstructions are at different scales),
#so for a fair side-by-side we divide each one by its own torso length. now both show up
#at the same body size and "green smaller than blue" goes away. display-only, doesnt touch the score
def scale_to_torso(poses):
    lhip, rhip = poses[:,JIDX["left-hip"]], poses[:,JIDX["right-hip"]]
    lsho, rsho = poses[:,JIDX["left-shoulder"]], poses[:,JIDX["right-shoulder"]]
    midhip, midsho = (lhip+rhip)/2, (lsho+rsho)/2
    torso = np.median(np.linalg.norm(midsho-midhip, axis=1))
    return poses/torso if torso > 1e-9 else poses


#procrustes: find the best rotation + scale + translation that maps source onto target,
#return source moved to that spot. just numpy svd, no extra installs. this is what makes
#the comparison translation/rotation/scale invariant (the paper's "body shape-invariant measure")
def procrustes_align(source, target):
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s0 = source-mu_s
    t0 = target-mu_t
    var_s = (s0**2).sum()
    K = s0.T @ t0
    U, sing, Vt = np.linalg.svd(K)
    V = Vt.T
    Z = np.eye(3)
    Z[-1,-1] = np.sign(np.linalg.det(V @ U.T))  #stops it flipping the pose inside out
    R = V @ Z @ U.T
    scale = np.trace(R @ K) / var_s
    t = mu_t - scale*(R @ mu_s)
    return scale*(source @ R.T) + t


#distance between two single poses AFTER procrustes aligning them - this is the per-frame
#pose distance, meaningful even when the two clips are in different coordinate frames
def pose_distance(a, b):
    a_aligned = procrustes_align(a, b)
    return np.linalg.norm(a_aligned - b, axis=1).mean()


def dtw_match(learner, expert):
    Ta, Tb = len(learner), len(expert)
    #cost[i,j] = how different is learner frame i from expert frame j (procrustes-aligned)
    cost = np.zeros((Ta, Tb))
    for i in range(Ta):
        for j in range(Tb):
            cost[i,j] = pose_distance(expert[j], learner[i])
    #dynamic programming: cheapest path from (0,0) to (Ta,Tb) stepping down/right/diagonal
    D = np.full((Ta+1, Tb+1), np.inf); D[0,0] = 0
    for i in range(1, Ta+1):
        for j in range(1, Tb+1):
            D[i,j] = cost[i-1,j-1] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    #walk the path back out
    i, j = Ta, Tb
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        k = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]])
        i, j = (i-1,j-1) if k==0 else ((i-1,j) if k==1 else (i,j-1))
    path = path[::-1]
    #for each learner frame, take the (first) expert frame it matched to
    match = np.zeros(Ta, dtype=int)
    seen = set()
    for li, ej in path:
        if li not in seen:
            match[li] = ej
            seen.add(li)
    return match


#PA-MPJPE from the paper - dtw-match the frames, then for each matched pair procrustes-align
#and measure leftover per-joint distance. returns the per-frame error curve + one overall score
def pa_mpjpe(learner, expert):
    match = dtw_match(learner, expert)
    errs = np.array([pose_distance(expert[match[i]], learner[i]) for i in range(len(learner))])
    return errs, errs.mean()


#draw one skeleton frame
def draw(ax, pose, color):
    for a,b in EDGES:
        if np.isnan(pose[a]).any() or np.isnan(pose[b]).any():
            continue
        ax.plot([pose[a,0],pose[b,0]], [pose[a,2],pose[b,2]], [pose[a,1],pose[b,1]], c=color, lw=2)


#save one person's motion as its own video
def render_single(poses, title, color, out_path, fps=12):
    flat = poses.reshape(-1,3); flat = flat[~np.isnan(flat).any(1)] #collapsing 17 * num of frames
    lo, hi = flat.min(0), flat.max(0)
    mid, rng = (lo+hi)/2, (hi-lo).max()*0.55

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1, projection="3d")

    def update(t):
        ax.cla()
        ax.set_title(f"{title}  (frame {t+1}/{len(poses)})")
        ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[2]-rng, mid[2]+rng); ax.set_zlim(mid[1]-rng, mid[1]+rng)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=12, azim=-70)
        draw(ax, poses[t], color)

    anim = FuncAnimation(fig, update, frames=len(poses))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("saved", out_path)


#side by side - both sequences are already dtw-aligned to the same length here, so frame t
#on the left is the SAME phase of the move as frame t on the right. shared box so sizes match
def render_side_by_side(poses1, title1, color1, poses2, title2, color2, out_path, fps=12):
    T = min(len(poses1), len(poses2))
    both = np.concatenate([poses1, poses2]).reshape(-1,3); both = both[~np.isnan(both).any(1)]
    lo, hi = both.min(0), both.max(0)
    mid, rng = (lo+hi)/2, (hi-lo).max()*0.55

    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(1,2,1, projection="3d")
    ax2 = fig.add_subplot(1,2,2, projection="3d")

    def update(t):
        for ax, title, poses, color in [(ax1,title1,poses1,color1), (ax2,title2,poses2,color2)]:
            ax.cla()
            ax.set_title(f"{title}  (frame {t+1}/{T})")
            ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[2]-rng, mid[2]+rng); ax.set_zlim(mid[1]-rng, mid[1]+rng)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=12, azim=-70)
            draw(ax, poses[t], color)

    anim = FuncAnimation(fig, update, frames=T)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    if os.path.isdir(ROOT):
        #real data path - grab the first usable pair from the drive json
        takes = json.load(open(f"{ROOT}/takes.json"))
        name2uid = {t["take_name"]:t["take_uid"] for t in takes}
        drive = json.load(open(DRIVE))

        learner_raw = expert_raw = None
        for row in drive:
            if row["video1"] == row["video2"]:
                continue
            l_uid = name2uid.get(row["video1"])
            e_uid = name2uid.get(row["video2"])
            if not (l_uid and e_uid):
                continue
            lf, lp = load_pose(l_uid)
            ef, ep = load_pose(e_uid)
            if lf is None or ef is None:
                continue
            lr = json.loads(row["video1_range"]); er = json.loads(row["video2_range"])
            lmask = (lf>=lr[0])&(lf<=lr[1])
            emask = (ef>=er[0])&(ef<=er[1])
            if lmask.sum()<15 or emask.sum()<15:
                continue
            learner_raw, expert_raw = lp[lmask], ep[emask]
            print("using pair:", row["video1"], "vs", row["video2"])
            print("commentary:", row["text"])
            print(f"learner frames: {len(learner_raw)}, expert frames: {len(expert_raw)}")
            break
        if learner_raw is None:
            raise SystemExit("no usable pair found in the drive json")
    else:
        print("ERROR!")
        sys.exit(1)
        
    learner = fill_gaps(learner_raw)
    expert = fill_gaps(expert_raw)

    #the actual similarity score - dtw lines up the frames, procrustes handles the coordinate mismatch
    errs, score = pa_mpjpe(learner, expert)
    print(f"\nPA-MPJPE similarity score: {score:.4f}  (lower = more alike)")
    print(f"per-frame range: {errs.min():.4f} to {errs.max():.4f}")

    #for display: center + same-size, and dtw-match the expert onto the learner's timeline
    learner_disp = scale_to_torso(center_root(learner)) #(frames, 17 joints, 3 axes)
    expert_disp  = scale_to_torso(center_root(expert))
    match = dtw_match(learner, expert)
    expert_aligned = expert_disp[match]   #now same length as learner, phase-matched frame by frame

    render_single(learner_disp, "learner", "tab:blue", "learner_original.gif")
    render_single(expert_disp, "expert", "tab:green", "expert_original.gif")
    render_side_by_side(learner_disp, "learner", "tab:blue",
                        expert_aligned, "expert (dtw-aligned)", "tab:green",
                        "side_by_side.gif")