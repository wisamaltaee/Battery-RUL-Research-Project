"""
features_nasa.py — Parse raw NASA PCoE .mat files (B0005/6/7/18) into
per-discharge-cycle features, using the SAME threshold-based RUL definition
and (as far as physically comparable) the same feature schema as
features_calce.py, so both chemistries can be pooled into one LOBO study.

NASA rated capacity is 2 Ah; PCoE's own convention (used across nearly all
papers built on this dataset) defines EOL at 30% fade, i.e. 1.4 Ah -- NOT
80%/80% like CALCE. We keep each dataset's own literature-standard
threshold (NASA: 30% fade -> 70% SOH remaining considered failed; CALCE:
20% fade -> 80% SOH). Using the "wrong" published threshold for either
would make our numbers incomparable to prior work, which defeats the
point of benchmarking against it.
"""

import numpy as np
import pandas as pd
import scipy.io as sio

NASA_EOL_FRACTION = 0.70  # NASA PCoE convention: EOL at 30% capacity fade


def load_discharge_cycles(mat_path: str, battery_id: str) -> pd.DataFrame:
    m = sio.loadmat(mat_path, simplify_cells=True)
    cycles = m[battery_id]["cycle"]
    rows = []
    cyc_num = 0
    for c in cycles:
        if c["type"] != "discharge":
            continue
        cyc_num += 1
        d = c["data"]
        v = np.asarray(d["Voltage_measured"], dtype=float)
        i = np.asarray(d["Current_measured"], dtype=float)
        t = np.asarray(d["Temperature_measured"], dtype=float)
        time = np.asarray(d["Time"], dtype=float)
        capacity = float(d["Capacity"]) if np.isscalar(d["Capacity"]) or np.ndim(d["Capacity"]) == 0 else float(np.ravel(d["Capacity"])[0])

        rows.append({
            "global_cycle": cyc_num,
            "discharge_capacity_Ah": capacity,
            "avg_voltage": np.nanmean(v),
            "voltage_drop": np.nanmax(v) - np.nanmin(v),
            "avg_current": np.nanmean(i),
            "discharge_time_s": float(time[-1] - time[0]) if len(time) > 1 else np.nan,
            "avg_temperature": np.nanmean(t),
            "temp_variance": np.nanvar(t),
            "ambient_temperature": c["ambient_temperature"],
        })
    return pd.DataFrame(rows)


def add_soh_rul(feat: pd.DataFrame, eol_fraction: float = NASA_EOL_FRACTION) -> pd.DataFrame:
    feat = feat.copy()
    ref_window = feat[(feat["global_cycle"] >= 2) & (feat["global_cycle"] <= 6)]
    ref_capacity = ref_window["discharge_capacity_Ah"].mean()
    feat["soh_pct"] = 100 * feat["discharge_capacity_Ah"] / ref_capacity

    eol_threshold = 100 * eol_fraction
    below = feat[feat["soh_pct"] < eol_threshold]
    if len(below):
        eol_cycle = below["global_cycle"].iloc[0]
        censored = False
    else:
        eol_cycle = feat["global_cycle"].max()
        censored = True

    feat["rul"] = eol_cycle - feat["global_cycle"]
    feat = feat[feat["global_cycle"] <= eol_cycle].reset_index(drop=True)
    feat.attrs["eol_cycle"] = eol_cycle
    feat.attrs["ref_capacity_Ah"] = ref_capacity
    feat.attrs["censored"] = censored
    return feat


def build(mat_path: str, battery_id: str) -> pd.DataFrame:
    feat = load_discharge_cycles(mat_path, battery_id)
    feat = add_soh_rul(feat)
    feat["battery_id"] = battery_id
    print(f"{battery_id}: ref_capacity={feat.attrs['ref_capacity_Ah']:.3f} Ah, "
          f"EOL_cycle={feat.attrs['eol_cycle']}, censored={feat.attrs['censored']}, "
          f"n_cycles_kept={len(feat)}")
    return feat


if __name__ == "__main__":
    for b in ["B0005", "B0006", "B0007", "B0018"]:
        build(f"data/raw/nasa/fy08q4/{b}.mat", b)
