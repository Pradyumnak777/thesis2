#!/usr/bin/env python3
"""Validate the ExpertEdit benchmark metadata release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {
    "reverse_layup": 499,
    "penalty_kick": 173,
    "midrange_jumpshot": 499,
    "mikan_layup": 436,
    "reverse_punch": 248,
    "spinning_back_kick": 224,
    "front_kick": 231,
    "roundhouse_kick": 239,
}
REQUIRED_FILES = [
    "README.md",
    "VERSION.txt",
    "LICENSE.md",
    "DATASET_LICENSE_NOTICE.md",
    "CITATION.cff",
    "manifest.json",
    "checksums.sha256",
    "data/evaluation_pairs.csv",
    "data/evaluation_pairs.jsonl",
    "data/clips.csv",
    "data/technique_metadata.json",
    "data/kinematic_signals.json",
    "data/evaluation_counts.json",
    "data/dtw_alignments/README.md",
    "data/dtw_alignments/egoexo4d_alignments.jsonl.gz",
    "scripts/README.md",
    "scripts/load_evaluation_pairs.py",
    "scripts/validate_release.py",
    "docs/field_dictionary.md",
    "docs/release_notes.md",
]
RAW_MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".npy", ".npz", ".pt", ".pth", ".pkl"}
ABS_PATH_RE = re.compile(r"(/vision/|/home/|/mnt/|/scratch/|/data/)")


def fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_checksums(root: Path, errors: list[str]) -> None:
    checksum_path = root / "checksums.sha256"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, path = line.split(None, 1)
        expected[path.strip().lstrip("*")] = digest
    for path_str, digest in expected.items():
        path = root / path_str.replace("./", "", 1)
        if not path.exists():
            fail(errors, f"checksum target missing: {path_str}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            fail(errors, f"checksum mismatch: {path_str}")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            fail(errors, f"required file missing: {rel}")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in RAW_MEDIA_EXTS:
            fail(errors, f"raw media/array/checkpoint-like file included: {path.relative_to(root)}")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".md", ".cff", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ABS_PATH_RE.search(text):
                fail(errors, f"absolute/internal path found in {path.relative_to(root)}")
    eval_rows = rows(root / "data/evaluation_pairs.csv")
    ids = [r["pair_id"] for r in eval_rows]
    if len(ids) != len(set(ids)):
        fail(errors, "pair_id values are not unique")
    counts = Counter(r["technique_slug"] for r in eval_rows)
    if dict(counts) != EXPECTED_COUNTS:
        fail(errors, f"evaluation counts mismatch: {dict(counts)}")
    counts_json = json.loads((root / "data/evaluation_counts.json").read_text(encoding="utf-8"))
    if counts_json != EXPECTED_COUNTS:
        fail(errors, "data/evaluation_counts.json does not match expected counts")
    clip_rows = rows(root / "data/clips.csv")
    clip_ids = {r["clip_uid"] for r in clip_rows}
    for r in eval_rows:
        if r["novice_clip_uid"] not in clip_ids:
            fail(errors, f"novice clip missing from clips.csv: {r['novice_clip_uid']}")
        if r["expert_clip_uid"] not in clip_ids:
            fail(errors, f"expert clip missing from clips.csv: {r['expert_clip_uid']}")
    dtw_ids: set[str] = set()
    with gzip.open(root / "data/dtw_alignments/egoexo4d_alignments.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dtw_ids.add(json.loads(line)["dtw_alignment_id"])
    for r in eval_rows:
        if r["dtw_used"].lower() == "true" and r["dtw_alignment_id"] not in dtw_ids:
            fail(errors, f"missing dtw alignment: {r['pair_id']}")
    validate_checksums(root, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
