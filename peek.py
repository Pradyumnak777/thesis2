import json, glob

ROOT="egoexo_survey"; POSE=f"{ROOT}/annotations/ego_pose"
DRIVE="expertaf_drive/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json"
takes=json.load(open(f"{ROOT}/takes.json"))
name2uid={t["take_name"]:t["take_uid"] for t in takes}
drive=json.load(open(DRIVE))

def body_file(uid):
    h=glob.glob(f"{POSE}/*/body/annotation/{uid}.json")
    return h[0] if h else None

def frame_keys(f):
    d=json.load(open(f))
    return set(int(k) for k in d if str(k).lstrip('-').isdigit())

cache={}
def keys_for(uid):
    if uid not in cache:
        f=body_file(uid)
        cache[uid]=frame_keys(f) if f else None
    return cache[uid]

usable=[]
for row in drive:
    l_uid=name2uid.get(row["video1"]); e_uid=name2uid.get(row["video2"])
    if not(l_uid and e_uid): continue
    lk=keys_for(l_uid); ek=keys_for(e_uid)
    if lk is None or ek is None: continue

    lr=json.loads(row["video1_range"]); er=json.loads(row["video2_range"])
    l_hits=[k for k in lk if lr[0]<=k<=lr[1]]
    e_hits=[k for k in ek if er[0]<=k<=er[1]]

    if len(l_hits)>=15 and len(e_hits)>=15:
        usable.append({
            "video1": row["video1"], "video1_range": lr, "learner_frames": len(l_hits),
            "video2": row["video2"], "video2_range": er, "expert_frames": len(e_hits),
            "body_part": row["corrected_body_part"], "text": row["text"][:80],
        })

print(f"Scanned {len(drive)} rows. Usable pairs: {len(usable)}")
for u in usable[:10]:
    print(u)