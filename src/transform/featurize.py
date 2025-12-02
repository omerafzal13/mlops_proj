# src/transform/featurize.py
import pandas as pd
import numpy as np
import os

def run_featurization(raw_path, out_path, horizon=4):
    df = pd.read_parquet(raw_path)
    if df.empty:
        raise ValueError("raw dataset empty")
    df = df.sort_values('dt').reset_index(drop=True)
    # target: temp at t + horizon (for forecast horizon in hours)
    df['temp_t+{}'.format(horizon)] = df['temp'].shift(-horizon)
    # lag features
    for lag in [1,2,3,6,12,24]:
        df[f'temp_lag_{lag}'] = df['temp'].shift(lag)
    # rolling aggregates
    df['temp_roll_3'] = df['temp'].rolling(window=3, min_periods=1).mean()
    df['temp_roll_6'] = df['temp'].rolling(window=6, min_periods=1).mean()
    # time features
    df['hour'] = df['dt'].dt.hour
    df['dow'] = df['dt'].dt.dayofweek
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    # drop rows without target
    df = df.dropna(subset=[f'temp_t+{horizon}']).reset_index(drop=True)
    # Optionally, keep only columns needed for model
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    # Save simple stats for drift detection
    stats = {
        'columns': {},
    }
    for c in ['temp','humidity','wind_speed','pressure','clouds']:
        if c in df.columns:
            stats['columns'][c] = {'min': float(df[c].min()), 'max': float(df[c].max())}
    import json
    stats_path = os.path.splitext(out_path)[0] + '_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    return out_path, stats_path

if __name__ == "__main__":
    import sys
    # simple CLI
    raw = sys.argv[1]
    out = sys.argv[2]
    print(run_featurization(raw, out))
