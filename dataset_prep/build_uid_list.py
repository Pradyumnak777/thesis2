#!/usr/bin/env python3
"""Collect the EgoExo4D take UIDs referenced by the ExpertAF data files."""
import argparse
import json
import sys


def names_from_pairs(path):
    with open(path) as f:
        data = json.load(f)
    names = set()
    for entry in data:
        for key in ("take", "video1", "video2"):
            if entry.get(key):
                names.add(entry[key])
    return names


def names_from_commentary(path):
    with open(path) as f:
        data = json.load(f)
    return {k.split("-----")[0] for k in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="ExpertAF train/test json files")
    ap.add_argument("--commentary", default=None,
                    help="optional parsed-commentary json; only needed if you "
                         "want to re-mine pairs yourself (superset of takes)")
    ap.add_argument("--takes-json", required=True,
                    help="<EGOEXO_ROOT>/takes.json (from --parts metadata)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = set()
    for p in args.pairs:
        names |= names_from_pairs(p)
    if args.commentary:
        names |= names_from_commentary(args.commentary)

    with open(args.takes_json) as f:
        takes = json.load(f)
    name2uid = {t["take_name"]: t["take_uid"] for t in takes}

    missing = sorted(n for n in names if n not in name2uid)
    uids = sorted(name2uid[n] for n in names if n in name2uid)

    with open(args.out, "w") as f:
        f.write("\n".join(uids) + "\n")

    print(f"{len(names)} unique take names -> {len(uids)} uids -> {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} take names not in takes.json:", file=sys.stderr)
        for n in missing:
            print(f"  {n}", file=sys.stderr)


if __name__ == "__main__":
    main()