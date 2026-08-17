# DTW Alignments

`egoexo4d_alignments.jsonl.gz` contains one JSON object per Ego-Exo4D evaluation pair with `dtw_used=true`.

Frame-index arrays are global frame indices in the referenced local pose sequence. `expert_frame_indices_original` gives the one-expert-frame-per-novice-frame mapping used for release loading and sanity checks.

Karate pairs are fixed-length 64-frame motion clips and use `dtw_used=false`.
