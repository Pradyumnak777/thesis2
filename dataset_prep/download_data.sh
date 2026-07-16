#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

TRAIN_JSON="expertaf_drive/pose_dataset_concise_NE_DTW_with_soccer_train_N5.json"
TEST_JSON="expertaf_drive/pose_dataset_concise_NE_DTW_with_soccer_val_N5.json"
EGOEXO_ROOT="egoexo_new"
UID_FILE="expertaf_take_uids.txt"

mkdir -p "$EGOEXO_ROOT"

egoexo -o "$EGOEXO_ROOT" --parts metadata -y

python3 build_uid_list.py \
    --pairs "$TRAIN_JSON" "$TEST_JSON" \
    --takes-json "$EGOEXO_ROOT/takes.json" \
    --out "$UID_FILE"

UIDS=$(tr '\n' ' ' < "$UID_FILE")

egoexo -o "$EGOEXO_ROOT" --parts annotations --benchmarks egopose --uids $UIDS -y
egoexo -o "$EGOEXO_ROOT" --parts ego_pose_pseudo_gt --uids $UIDS -y
egoexo -o "$EGOEXO_ROOT" --parts downscaled_takes/448 --uids $UIDS -y

echo "Done. Data in $EGOEXO_ROOT"