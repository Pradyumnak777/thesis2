import csv
from pathlib import Path

CLIPS_CSV = Path("data/clips.csv")
OUT_DIR = Path(".") 

def take_name(row):
    return row["video_uid"] if row["video_uid"] != "NA" else row["source_sequence_id"]

def timeframe(row):
    if row["start_sec"] != "NA" and row["end_sec"] != "NA":
        return f'{row["start_sec"]}-{row["end_sec"]}'
    # karate clips have no seconds, fall back to frame range
    return f'{row["start_frame"]}-{row["end_frame"]} frames'

novice_rows = []
expert_rows = []

with CLIPS_CSV.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        entry = [row["technique"], take_name(row), timeframe(row)]
        if row["role"] == "novice":
            novice_rows.append(entry)
        elif row["role"] == "expert":
            expert_rows.append(entry)

for name, rows in [("novice_clips.csv", novice_rows), ("expert_clips.csv", expert_rows)]:
    with (OUT_DIR / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["action", "take", "timeframe"])
        writer.writerows(rows)

print(f"novice: {len(novice_rows)} rows")
print(f"expert: {len(expert_rows)} rows")