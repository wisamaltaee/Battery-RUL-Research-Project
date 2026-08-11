"""
features_calce.py — Build per-cycle tabular features + threshold-based RUL
labels from parsed raw CALCE Arbin logs.

Key correction vs. a naive read of the data: Discharge_Capacity(Ah) and
Charge_Capacity(Ah) are CUMULATIVE WITHIN EACH SOURCE FILE (they reset to 0
at the start of every new dated .xlsx, not at the start of every cycle).
True per-cycle discharge capacity = diff of consecutive cycle-end cumulative
values, with the first cycle of each file diffed against 0.

RUL is defined properly here (unlike the NASA prototype's "cycles-to-end-
of-test" placeholder): End-of-life = first cycle where capacity drops below
80% of the cell's early-life reference capacity (mean of cycles 2-6, to
avoid first-cycle formation-cycle noise). RUL[c] = EOL_cycle - c. Cycles
after EOL are dropped (censored) rather than clipped to 0, since we want
the model predicting genuine "distance to failure", not distance to an
already-passed threshold.
"""

import numpy as np
import pandas as pd
from src.parse_calce import parse_battery

EOL_FRACTION = 0.80


def cycle_features(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (file_order, cyc), g in raw.groupby(["__file_order", "Cycle_Index"], sort=True):
        global_cycle = g["global_cycle"].iloc[0]
        dis_cap_cum_max = g["Discharge_Capacity(Ah)"].max()
        chg_cap_cum_max = g["Charge_Capacity(Ah)"].max()
        volt = g["Voltage(V)"]
        curr = g["Current(A)"]
        ir = g["Internal_Resistance(Ohm)"]
        ir_nonzero = ir[ir > 0]

        # discharge step = negative current in this schedule convention
        discharge_rows = g[g["Current(A)"] < -1e-6]
        discharge_time_s = (
            discharge_rows["Test_Time(s)"].max() - discharge_rows["Test_Time(s)"].min()
            if len(discharge_rows) > 1 else 0.0
        )

        rows.append({
            "file_order": file_order,
            "local_cycle": cyc,
            "global_cycle": global_cycle,
            "cum_discharge_Ah": dis_cap_cum_max,
            "cum_charge_Ah": chg_cap_cum_max,
            "avg_voltage": volt.mean(),
            "voltage_drop": volt.max() - volt.min(),
            "avg_current": curr.mean(),
            "discharge_time_s": discharge_time_s,
            "internal_resistance": ir_nonzero.mean() if len(ir_nonzero) else np.nan,
            "n_points": len(g),
        })

    feat = pd.DataFrame(rows).sort_values(["file_order", "local_cycle"]).reset_index(drop=True)

    # true per-cycle discharge capacity: diff within each file, first cycle
    # of each file diffed against 0
    feat["discharge_capacity_Ah"] = feat.groupby("file_order")["cum_discharge_Ah"].diff()
    first_in_file = feat.groupby("file_order")["cum_discharge_Ah"].transform("first")
    is_first_row_of_file = feat["discharge_capacity_Ah"].isna()
    feat.loc[is_first_row_of_file, "discharge_capacity_Ah"] = feat.loc[is_first_row_of_file, "cum_discharge_Ah"]

    feat = feat.sort_values("global_cycle").reset_index(drop=True)
    feat["internal_resistance"] = feat["internal_resistance"].ffill()

    # The Arbin log for each battery is split into ~25 dated .xlsx files
    # (weekly data pulls), and the LAST cycle logged in every file except
    # the final one is mid-discharge when the log cuts off -- it reads as
    # a near-zero-capacity cycle that is NOT real degradation. Drop those
    # truncated boundary rows; the genuinely short cycles that occur near
    # real end-of-life (fewer points because a degraded cell hits cutoff
    # voltage faster) are untouched since they aren't file-boundary rows.
    max_file_order = feat["file_order"].max()
    is_last_in_file = feat.groupby("file_order")["local_cycle"].transform("max") == feat["local_cycle"]
    truncated = is_last_in_file & (feat["file_order"] != max_file_order)
    feat = feat[~truncated].reset_index(drop=True)
    return feat


def drop_anomalous_short_cycles(feat: pd.DataFrame, window: int = 11, rel_thresh: float = 0.90) -> pd.DataFrame:
    """
    Beyond the file-boundary truncation already handled, the CALCE test
    schedule periodically runs a shorter/interrupted cycle mid-file (every
    ~10-11 cycles) whose discharge_time_s and discharge_capacity_Ah are
    well below neighboring cycles, then immediately recovers next cycle --
    a signature of a non-full-rate check cycle, not real fade. Flag cycles
    whose discharge_time_s falls sharply below a local rolling median and
    drop them from RUL/SOH modeling.
    """
    feat = feat.sort_values("global_cycle").reset_index(drop=True)
    roll_med = feat["discharge_time_s"].rolling(window, center=True, min_periods=3).median()
    is_anomalous = feat["discharge_time_s"] < (rel_thresh * roll_med)
    return feat[~is_anomalous].reset_index(drop=True)


def add_soh_rul(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()
    # reference capacity: mean of cycles 2-6 (skip cycle 1 formation noise)
    ref_window = feat[(feat["global_cycle"] >= 2) & (feat["global_cycle"] <= 6)]
    ref_capacity = ref_window["discharge_capacity_Ah"].mean()
    feat["soh_pct"] = 100 * feat["discharge_capacity_Ah"] / ref_capacity

    eol_threshold = 100 * EOL_FRACTION
    below = feat[feat["soh_pct"] < eol_threshold]
    if len(below):
        eol_cycle = below["global_cycle"].iloc[0]
        censored = False
    else:
        eol_cycle = feat["global_cycle"].max()
        censored = True  # never actually crossed 80% in the observed data

    feat["rul"] = eol_cycle - feat["global_cycle"]
    feat = feat[feat["global_cycle"] <= eol_cycle].reset_index(drop=True)
    feat.attrs["eol_cycle"] = eol_cycle
    feat.attrs["ref_capacity_Ah"] = ref_capacity
    feat.attrs["censored"] = censored
    return feat


def build(folder: str, battery_id: str) -> pd.DataFrame:
    raw = parse_battery(folder)
    feat = cycle_features(raw)
    feat = drop_anomalous_short_cycles(feat)
    feat = add_soh_rul(feat)
    feat["battery_id"] = battery_id
    print(f"{battery_id}: ref_capacity={feat.attrs['ref_capacity_Ah']:.3f} Ah, "
          f"EOL_cycle={feat.attrs['eol_cycle']}, censored={feat.attrs['censored']}, "
          f"n_cycles_kept={len(feat)}")
    return feat


if __name__ == "__main__":
    for batt in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        build(f"data/raw/calce/{batt}", batt)
