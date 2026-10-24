import io
import os
import time
from collections import deque

import numpy as np
import pandas as pd

RT_WINDOW_SECONDS  = 15 # widened from 10, helped with misses
RT_WARMUP_SECONDS  = 20
SENSOR_HZ          = 104
LANDING_BUFFER_S   = 0.5 # has to match landing buffer in jump_detection.py
DATA_FOLDER        = "data_live"
CHEST_FILE         = os.path.join(DATA_FOLDER, "chest.csv")
WRIST_FILE         = os.path.join(DATA_FOLDER, "wrist.csv")

# csv columns from recorder.py
_RAW_COLS = [
    "timestamp_utc_ms",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "magn_x", "magn_y", "magn_z",
]

#open filepath and seek to read only bytes appended since last call
def _read_new_rows(filepath: str, byte_offset: int, col_names: list) -> tuple[pd.DataFrame, int]:
    empty = pd.DataFrame(columns=col_names)

    if not os.path.exists(filepath):
        return empty, byte_offset

    try:
        with open(filepath, "rb") as f:
            if byte_offset == 0:
                f.readline()       
                byte_offset = f.tell()

            f.seek(byte_offset)
            new_bytes = f.read()
            new_offset = f.tell()

        if not new_bytes.strip():
            return empty, byte_offset # nothing new yet

        new_df = pd.read_csv(
            io.BytesIO(new_bytes),
            names=col_names,
            on_bad_lines="skip",
        )
        return new_df, new_offset

    except Exception:
        return empty, byte_offset


def realtime_worker(shared: dict, params: dict):
    from processing.filters        import process_imu_data
    from processing.jump_detection import detect_jumps
    from processing.hit_detection  import detect_hits
    from processing.hit_timing     import evaluate_hit_timing

    #Buffer holds: madgwick 20s; 15s processing; and landing buffer
    MAX_BUFFER_ROWS = int((RT_WARMUP_SECONDS + RT_WINDOW_SECONDS + LANDING_BUFFER_S) * SENSOR_HZ)
    synced_buffer: deque = deque(maxlen=MAX_BUFFER_ROWS)

    #track how many bytes already consumed
    chest_offset: int = 0
    wrist_offset: int = 0

    
    _OVERLAP_ROWS = int(0.5 * SENSOR_HZ)
    prev_chest_tail: pd.DataFrame = pd.DataFrame(columns=_RAW_COLS)
    prev_wrist_tail: pd.DataFrame = pd.DataFrame(columns=_RAW_COLS)

  # init
    rt_seen_takeoffs: list = []
    rt_seen_landings: list = []
    rt_seen_flights:  list = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    while shared.get("running", False):
        time.sleep(1.0)  # reduced from 1.5 s, worked better

        try:
            #only new rows in csv (used to read all)
            new_chest, chest_offset = _read_new_rows(CHEST_FILE, chest_offset, _RAW_COLS)
            new_wrist, wrist_offset = _read_new_rows(WRIST_FILE, wrist_offset, _RAW_COLS)

            if new_chest.empty or new_wrist.empty:
                continue

            # fix column name
            for df in (new_chest, new_wrist):
                if "timestamp_utc_ms" in df.columns:
                    df.rename(columns={"timestamp_utc_ms": "timestamp"}, inplace=True)

            # numeric timestamps or else thres a warning
            for df in (new_chest, new_wrist):
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                df.dropna(subset=["timestamp"], inplace=True)

            if new_chest.empty or new_wrist.empty:
                continue

            #sync new rows and skip concat on the first second to avoid pandas FutureWarning
            if prev_chest_tail.empty:
                chest_for_sync = new_chest
                wrist_for_sync = new_wrist
            else:
                chest_for_sync = pd.concat([prev_chest_tail, new_chest], ignore_index=True)
                wrist_for_sync = pd.concat([prev_wrist_tail, new_wrist], ignore_index=True)

            chest_for_sync.sort_values("timestamp", inplace=True)
            wrist_for_sync.sort_values("timestamp", inplace=True)

            new_synced = pd.merge_asof(
                chest_for_sync,
                wrist_for_sync,
                on="timestamp",
                suffixes=("_chest", "_wrist"),
                direction="nearest",
                tolerance=50,
            )
            new_synced.dropna(subset=["acc_x_wrist"], inplace=True)

            # Drop overlap rows from last second
            if not prev_chest_tail.empty:
                overlap_cutoff = prev_chest_tail["timestamp"].iloc[-1]
                new_synced = new_synced[new_synced["timestamp"] > overlap_cutoff]

            prev_chest_tail = new_chest.tail(_OVERLAP_ROWS).copy()
            prev_wrist_tail = new_wrist.tail(_OVERLAP_ROWS).copy()

            if new_synced.empty:
                continue

            #Append new synced rows
            for row in new_synced.itertuples(index=False):
                synced_buffer.append(row._asdict())

            # Need at least half the warmup window before doing anything useful
            if len(synced_buffer) < int(RT_WARMUP_SECONDS * SENSOR_HZ * 0.5):
                continue

            #data frame build
            working_df = pd.DataFrame(list(synced_buffer))
            working_df.sort_values("timestamp", inplace=True)
            working_df.reset_index(drop=True, inplace=True)

            #madgwick filter
            processed_full = process_imu_data(
                working_df,
                enable_butterworth=True,
                cutoff_freq=10.0,
                sensors="chest_only",
            )

            #remove last 0.5s before passing to detect_jumps so there is enough data after a landing for 'landing force' check. Any jumps in this zone will go into next one second refresh
            detection_rows = int(RT_WINDOW_SECONDS * SENSOR_HZ)
            buffer_rows    = int(LANDING_BUFFER_S * SENSOR_HZ)
            trimmed        = processed_full.tail(detection_rows + buffer_rows)
            if len(trimmed) > buffer_rows:
                processed = trimmed.iloc[:-buffer_rows].copy()
            else:
                processed = trimmed.copy()

            # now detection processing pipeline
            jumps = detect_jumps(processed, **params["jump"])

            if jumps:
                jumps = detect_hits(processed, jumps, **params["hit"])
                jumps = evaluate_hit_timing(jumps, **params["timing"])

            shared["latest_jumps"] = jumps if jumps else []
            shared["last_update"]  = time.time()
            shared["error"]        = None

            # update RT jumps if a 'better' (by jump height) one comes instead of dropping duplicates because then it keeps the wrong shorter jump
            # replaces in stored memory and csv
            rt_jumps_file = params.get("rt_jumps_file")
            if rt_jumps_file and jumps:
                inserts: list = []   # actual new jumps
                updates: list = []   # (match_idx, jump_dict) pairs to overwrite

                for j in jumps:
                    t_off  = j.get("takeoff_t", 0)
                    t_land = j.get("landing_t", t_off)
                    flight = j.get("flight_time_s", 0)

                    # check for takeoff within 0.5 s that's already been detected
                    match_idx = next(
                        (i for i, seen_t in enumerate(rt_seen_takeoffs)
                         if abs(t_off - seen_t) < 0.5),
                        None,
                    )
                    # check if takeoff is within 0.6 s of previous landing because then its not actually a takeoff
                    ghost_after_landing = any(
                        0 <= t_off - seen_land < 0.6
                        for seen_land in rt_seen_landings
                    )

                    if match_idx is not None:
                        #fixes cuttoff jumps at end of window
                        if flight > rt_seen_flights[match_idx] + 0.05:
                            rt_seen_takeoffs[match_idx] = t_off
                            rt_seen_landings[match_idx] = t_land
                            rt_seen_flights[match_idx]  = flight
                            updates.append((match_idx, j))
                        
                    elif not ghost_after_landing:
                        rt_seen_takeoffs.append(t_off)
                        rt_seen_landings.append(t_land)
                        rt_seen_flights.append(flight)
                        inserts.append(j)

                    else:
                        rt_seen_landings.append(t_land)

                # Cap list incase of a long session
                rt_seen_takeoffs = rt_seen_takeoffs[-50:]
                rt_seen_landings = rt_seen_landings[-50:]
                rt_seen_flights  = rt_seen_flights[-50:]

                if inserts or updates:
                    if os.path.exists(rt_jumps_file):
                        existing = pd.read_csv(rt_jumps_file)
                        existing = existing.dropna(axis=1, how="all")
                        if "takeoff_t" in existing.columns:
                            existing["takeoff_t"] = existing["takeoff_t"].astype(float)
                        if "landing_t" in existing.columns:
                            existing["landing_t"] = existing["landing_t"].astype(float)
                        if "hit_time" in existing.columns:
                            existing["hit_time"] = existing["hit_time"].astype(float)    

                        for _, j in updates:
                            t_off = j.get("takeoff_t", 0)
                            mask = (existing["takeoff_t"] - t_off).abs() < 0.5
                            if mask.any():
                                for col, val in j.items():
                                    if col in existing.columns:
                                        existing.loc[mask, col] = val

                        # Append new jumps
                        if inserts:
                            existing = pd.concat(
                                [existing, pd.DataFrame(inserts)],
                                ignore_index=True,
                            ).dropna(axis=1, how="all") #4-15 otherwise get a futurewarning: behavior of DataFrame concatenation with empty or all-NA entries is deprecated

                        existing = existing.sort_values("takeoff_t").reset_index(drop=True)
                        existing.to_csv(rt_jumps_file, index=False)

                    else:
                        if inserts:
                            pd.DataFrame(inserts).to_csv(rt_jumps_file, index=False)

        except Exception as e:
            shared["error"] = str(e)

    shared["latest_jumps"] = []
