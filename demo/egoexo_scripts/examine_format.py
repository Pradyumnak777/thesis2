"""Inspect EgoExo4D metadata + proficiency annotations to discover the real schema.
Run after Phase 1. Prints structure and actual label values"""
import json, collections
from pathlib import Path

ROOT = Path("demo/egoexo4d_v2")

def peek(obj, name, depth=0, max_depth=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{pad}{name}: dict({len(keys)} keys)")
        for k in keys[:8]:
            peek(obj[k], k, depth + 1, max_depth) if depth < max_depth else print(f"{pad}  {k}: ...")
        if len(keys) > 8:
            print(f"{pad}  ... +{len(keys)-8} more keys")
    elif isinstance(obj, list):
        print(f"{pad}{name}: list[{len(obj)}]")
        if obj and depth < max_depth:
            peek(obj[0], f"{name}[0]", depth + 1, max_depth)
    else:
        v = repr(obj)
        print(f"{pad}{name}: {type(obj).__name__} = {v[:90]}")

# ---- takes.json: what identifies a basketball take? ----
takes = json.load(open(ROOT / "takes.json"))
print(f"=== takes.json: {len(takes)} takes ===")
peek(takes[0], "takes[0]")

print("\n=== distinct task_name values containing 'basket' ===")
tasks = collections.Counter(t.get("task_name", "?") for t in takes)
for k, v in tasks.most_common():
    if "basket" in str(k).lower():
        print(f"  {v:5d}  {k}")

# ---- proficiency annotations ----
for split in ("train", "val"):
    p = ROOT / "annotations" / f"proficiency_demonstrator_{split}.json"
    if not p.exists():
        print(f"\n!! missing {p.name}")
        continue
    d = json.load(open(p))
    print(f"\n=== proficiency_demonstrator_{split}.json ===")
    peek(d, "root")

# ---- narrations ----
for split in ("train", "val"):
    p = ROOT / "annotations" / f"atomic_action_descriptions_{split}.json"
    if not p.exists():
        print(f"\n!! missing {p.name}")
        continue
    d = json.load(open(p))
    print(f"\n=== atomic_action_descriptions_{split}.json ===")
    peek(d, "root")

print("\n=== annotation files actually present ===")
for f in sorted((ROOT / "annotations").glob("*.json")):
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")