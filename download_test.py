import json
from collections import Counter

# --- adjust these two paths if needed ---
DRIVE = "expertaf_drive"
TAKES = "egoexo_survey/takes.json"

FILES = {
    "train": f"{DRIVE}/pose_dataset_concise_NE_DTW_with_soccer_train_N5.json",
    "val":   f"{DRIVE}/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json",
}

# --- robust loader: normal parse, else salvage complete objects before truncation ---
def load_array(path):
    s = open(path).read()
    try:
        return json.loads(s), "clean"
    except json.JSONDecodeError as e:
        dec = json.JSONDecoder()
        i = s.find("[") + 1
        out, n = [], len(s)
        while i < n:
            while i < n and s[i] in " \t\r\n,":
                i += 1
            if i >= n or s[i] == "]":
                break
            try:
                obj, end = dec.raw_decode(s, i)
            except json.JSONDecodeError:
                break            # hit the truncation point
            out.append(obj)
            i = end
        return out, f"salvaged {len(out)} (truncated near char {e.pos})"

# --- name -> uid map ---
takes = json.load(open(TAKES))
name2uid = {t["take_name"]: t["take_uid"] for t in takes}

take_names = set()
scenario_rows = Counter()      # rough category tally by name prefix
for split, path in FILES.items():
    data, status = load_array(path)
    for row in data:
        take_names.add(row["video1"])
        take_names.add(row["video2"])
    print(f"[{split}] rows={len(data):>6}  ({status})")

# --- map to UIDs ---
found, missing = [], []
for name in sorted(take_names):
    (found if name in name2uid else missing).append(name)

uids = sorted({name2uid[n] for n in found})

# --- category breakdown (by take-name keyword) ---
cat = Counter()
for n in take_names:
    low = n.lower()
    if "basketball" in low:      cat["basketball"] += 1
    elif "bouldering" in low or "climb" in low: cat["rock climbing"] += 1
    elif "soccer" in low:        cat["soccer"] += 1
    else:                        cat["other"] += 1

print("\nUnique take names referenced:", len(take_names))
print("Category breakdown:", dict(cat))
print("Mapped to UID:", len(found))
print("UNMAPPED (not in takes.json):", len(missing))
if missing:
    print("  examples:", missing[:8])

open("download_uids.txt", "w").write("\n".join(uids))
print(f"\nWrote download_uids.txt with {len(uids)} unique takes")