#!/usr/bin/env python3
"""
Batch-extract “force-ramp” regions from all JSON logs in a folder.

For each file:
  • Ignore everything up to (and incl.) the first gpio.C1 == True row
  • Start 10 rows BEFORE the first sample whose |force| > 0.1 kN [set suitable threshold]
  • End at the first sample whose |force| ≥ 5 kN
  • Export time_s, position, velocity, load_cell.force (all abs), gpio.C1-3
    as <test_id>.csv in OUTPUT_DIR
"""

import json, os, glob, pandas as pd

# ─────────── USER SETTINGS ────────────────────────────────────────────
INPUT_DIR       = "raw_force_data/"          # folder of *.json logs
OUTPUT_DIR      = "./force_data_extracted/"         # where CSVs will go

FORCE_LOW_N    = 0.1
FORCE_HIGH_N   = 5.0
ROWS_BEFORE     = 10                     # context rows before LOW_KN crossing

COLS = ["time_s", "position", "velocity",
        "load_cell.force", "gpio.C1", "gpio.C2", "gpio.C3"]
# ──────────────────────────────────────────────────────────────────────


def process_file(path: str):
    """Return a trimmed DataFrame according to the rules, or None on error."""
    with open(path) as f:
        raw = json.load(f)

    # drop ros heartbeat rows
    raw = [r for r in raw if "position" in r]
    if not raw:
        print(f"⚠ {os.path.basename(path)}: no position rows, skipped")
        return None

    df = pd.json_normalize(raw)
    df["time_s"] = (df["ts"] - df["ts"].iloc[0]) / 1e9
    df.drop(columns="ts", inplace=True)

    # after first gpio.C1 == True
    c1_idx = df.index[df["gpio.C1"] == True]
    if c1_idx.empty:
        print(f"⚠ {os.path.basename(path)}: no gpio.C1==True, skipped")
        return None
    df2 = df.iloc[df.index.get_loc(c1_idx[0]) + 1 :]

    # start (|force| > low threshold)
    abs_force = df2["load_cell.force"].abs()
    low_hits = abs_force[abs_force > FORCE_LOW_N].index
    if low_hits.empty:
        print(f"⚠ {os.path.basename(path)}: force never > {FORCE_LOW_N} kN, skipped")
        return None
    start_pos = max(df2.index.get_loc(low_hits[0]) - ROWS_BEFORE, 0)

    # end (|force| ≥ high threshold)
    high_hits = abs_force[abs_force >= FORCE_HIGH_N].index
    end_pos = (
        df2.index.get_loc(high_hits[0]) - 1 if high_hits.any()
        else len(df2) - 1
    )

    region = df2.iloc[start_pos : end_pos + 1].copy()
    region["position"]        = region["position"].abs()
    region["load_cell.force"] = region["load_cell.force"].abs()
    return region


# ─────────── RUN BATCH ────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
if not files:
    raise RuntimeError(f"No .json files found in {INPUT_DIR}")

# process each test file, extract the region, and write to CSV
for fp in files:
    df_trim = process_file(fp)
    if df_trim is None:
        continue                      # skip files that failed validation
    test_id = os.path.splitext(os.path.basename(fp))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"{test_id}.csv")
    df_trim.loc[:, COLS].to_csv(csv_path, index=False)
    print(f"✓ wrote {csv_path}  ({len(df_trim)} rows)")

print("\nAll done.")

