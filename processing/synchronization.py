import pandas as pd


def align_sensors_zeroed(reference_df, target_df):
    ref = reference_df.copy()
    tgt = target_df.copy()

    #either column name
    for df in (ref, tgt):
        if 'timestamp_utc_ms' in df.columns and 'timestamp' not in df.columns:
            df.rename(columns={'timestamp_utc_ms': 'timestamp'}, inplace=True)

    ref.sort_values('timestamp', inplace=True)
    tgt.sort_values('timestamp', inplace=True)

    # 50 ms tolerance just bc, this won't fix clock drift though
    merged = pd.merge_asof(
        ref, tgt,
        on='timestamp',
        suffixes=('_chest', '_wrist'),
        direction='nearest',
        tolerance=50
    )

    merged.dropna(subset=['acc_x_wrist'], inplace=True)

    return merged