# Release Contents

Release name: expertedit_eval_benchmark_v1.0
Build date: 2026-07-01T02:42:58Z
Git commit of codebase used: e96efce4a7d551d5ce0d3f692830b5507568620f

## Files

- `data/evaluation_pairs.csv`: one row per released novice--expert benchmark pair.
- `data/evaluation_pairs.jsonl`: JSON Lines copy of the same pair records.
- `data/clips.csv`: de-duplicated clip/action-instance metadata.
- `data/dtw_alignments/egoexo4d_alignments.jsonl.gz`: DTW frame correspondences for Ego-Exo4D pairs.
- `data/technique_metadata.json`: technique-level metadata.
- `data/kinematic_signals.json`: kinematic signal names and defaults.
- `data/evaluation_counts.json`: row counts by technique.
- `scripts/`: lightweight loader and release validator.
- `docs/field_dictionary.md`: field definitions.

## Row Counts

Evaluation pair rows: 2549
DTW alignment rows: 1607
Technique row counts:
- reverse_layup: 499
- penalty_kick: 173
- midrange_jumpshot: 499
- mikan_layup: 436
- reverse_punch: 248
- spinning_back_kick: 224
- front_kick: 231
- roundhouse_kick: 239

Validation status: passed by `scripts/validate_release.py` during release build.

## Notes

- URLs, DOI, and final metadata/script license require author confirmation.
