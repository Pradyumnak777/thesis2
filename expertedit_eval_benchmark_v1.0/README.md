# ExpertEdit Evaluation Benchmark Metadata

This release contains metadata, evaluation pseudo-pairs, DTW alignments, technique metadata, and kinematic signal definitions for the ExpertEdit evaluation benchmark described in the ECCV 2026 camera-ready paper.

## Project links

- Project page: (https://vision.cs.utexas.edu/projects/expert_edit/)
- Paper: (https://arxiv.org/abs/2604.10466)

## What is included

- Evaluation pseudo-pair metadata for Ego-Exo4D basketball/soccer and Kyokushin Karate.
- DTW frame correspondences for Ego-Exo4D basketball/soccer evaluation pairs.
- Technique metadata and kinematic signal definitions.
- Lightweight loader and validation scripts.

## What is not included

- No raw Ego-Exo4D videos.
- No restricted source dataset media.
- No raw pose or mocap arrays.
- Users must obtain underlying datasets from official providers and comply with their terms.

## Data format

The core file is `data/evaluation_pairs.csv`, with one row per released novice--expert reference pair. `data/evaluation_pairs.jsonl` contains the same records for line-oriented loading. Clip metadata is de-duplicated in `data/clips.csv`. DTW mappings are stored in `data/dtw_alignments/egoexo4d_alignments.jsonl.gz`.

See `docs/field_dictionary.md` for field definitions and missing-value conventions. CSV scalar missing values use `NA`; JSON missing values use `null`.

## How to use

Run the loader from the release root:

```bash
python3 scripts/load_evaluation_pairs.py
```

Example Python usage:

```python
from scripts.load_evaluation_pairs import load_pairs

pairs = load_pairs("data/evaluation_pairs.csv")
for pair in pairs:
    print(pair["pair_id"], pair["novice_pose_ref"], pair["expert_pose_ref"])
```

Join rows to local dataset copies using the source dataset IDs, video IDs, clip IDs, start/end frames, and relative pose references. This release provides identifiers and mappings only; it does not grant access to the underlying source datasets.

## Versioning

`v1.0` corresponds to the ECCV 2026 camera-ready version.

## License and source dataset restrictions

The intended metadata license is CC BY 4.0, pending author confirmation. The scripts in `scripts/` are intended to be MIT licensed. Underlying dataset licenses and terms still apply.

## Citation

Please cite the ExpertEdit paper:

- ExpertEdit (ECCV 2026)

## Contact

Contact: ars7452@my.utexas.edu
