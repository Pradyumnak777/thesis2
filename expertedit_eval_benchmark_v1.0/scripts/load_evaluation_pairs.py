#!/usr/bin/env python3
"""Load ExpertEdit evaluation pair metadata."""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path


def load_pairs(path: str | Path = "data/evaluation_pairs.csv") -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_dtw_ids(root: str | Path = ".") -> set[str]:
    root = Path(root)
    ids: set[str] = set()
    for path in (root / "data" / "dtw_alignments").glob("*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["dtw_alignment_id"])
    return ids


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    pairs = load_pairs(root / "data" / "evaluation_pairs.csv")
    pair_ids = [row["pair_id"] for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise SystemExit("pair_id values are not unique")
    print("Counts by dataset:")
    for key, value in sorted(Counter(row["dataset"] for row in pairs).items()):
        print(f"  {key}: {value}")
    print("Counts by technique:")
    for key, value in sorted(Counter(row["technique_slug"] for row in pairs).items()):
        print(f"  {key}: {value}")
    dtw_ids = load_dtw_ids(root)
    missing = [
        row["pair_id"]
        for row in pairs
        if row["dtw_used"].lower() == "true" and row["dtw_alignment_id"] not in dtw_ids
    ]
    if missing:
        raise SystemExit(f"{len(missing)} DTW alignment IDs are missing")
    print(f"Loaded {len(dtw_ids)} DTW alignments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
