#!/usr/bin/env python3
"""Filter the manifest to samples with usable pose coverage on both sides."""
import argparse
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="e.g. dataset_out/val_manifest.jsonl")
    ap.add_argument("--out", required=True, help="filtered manifest path")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="min fraction of frames with pose (0.0 = just non-empty, "
                         "0.9 = near-full coverage)")
    ap.add_argument("--require", choices=["both", "either"], default="both",
                    help="both sides must clear the threshold, or just one")
    args = ap.parse_args()

    kept, dropped = [], 0
    with open(args.manifest) as f:
        lines = [json.loads(l) for l in f]

    for r in lines:
        cov_l = r["learner"]["pose_coverage"]
        cov_e = r["expert"]["pose_coverage"]
        ok_l = cov_l > args.min_coverage
        ok_e = cov_e > args.min_coverage
        keep = (ok_l and ok_e) if args.require == "both" else (ok_l or ok_e)
        if keep:
            kept.append(r)
        else:
            dropped += 1

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"{len(lines)} total | {len(kept)} kept | {dropped} dropped -> {args.out}")

if __name__ == "__main__":
    main()