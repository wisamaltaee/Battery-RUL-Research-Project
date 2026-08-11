"""
build_dataset.py — Combine cleaned NASA (raw .mat) and CALCE (raw Arbin)
per-cycle features into one dataset with a COMMON feature subset, so we can
test generalization not just within a chemistry/cell-batch but across
completely different datasets (different lab, tester hardware, cell
manufacturer, form factor, and rated capacity).

Honesty note kept from the data itself: NASA's PCoE cells are commonly
described only as "2 Ah 18650 Li-ion" without a confirmed public cathode
chemistry; CALCE's CS2 cells are documented as LiCoO2 prismatic. Both may
plausibly be cobalt-oxide-based. Rather than assert an unverified "cross-
chemistry" claim, we describe this as cross-dataset / cross-manufacturer /
cross-form-factor generalization -- which is still the deployment-relevant
stress test (a BMS trained on one product line seeing a cell from a
different one), just accurately labeled.
"""

import pandas as pd
from src.features_calce import build as build_calce
from src.features_nasa import build as build_nasa

COMMON_FEATURES = ["global_cycle", "discharge_time_s", "avg_voltage", "voltage_drop", "avg_current"]
TARGET = "rul"


def load_all():
    calce_batts = {
        "CS2_35": "data/raw/calce/CS2_35",
        "CS2_36": "data/raw/calce/CS2_36",
        "CS2_37": "data/raw/calce/CS2_37",
        "CS2_38": "data/raw/calce/CS2_38",
    }
    nasa_batts = {
        "B0005": "data/raw/nasa/fy08q4/B0005.mat",
        "B0006": "data/raw/nasa/fy08q4/B0006.mat",
        "B0007": "data/raw/nasa/fy08q4/B0007.mat",
        "B0018": "data/raw/nasa/fy08q4/B0018.mat",
    }

    frames = []
    for bid, path in calce_batts.items():
        f = build_calce(path, bid)
        f["dataset"] = "CALCE"
        frames.append(f)
    for bid, path in nasa_batts.items():
        f = build_nasa(path, bid)
        f["dataset"] = "NASA"
        frames.append(f)

    keep = COMMON_FEATURES + [TARGET, "battery_id", "dataset", "soh_pct"]
    combined = pd.concat([f[keep] for f in frames], ignore_index=True)
    return combined


if __name__ == "__main__":
    df = load_all()
    print(df.shape)
    print(df.groupby(["dataset", "battery_id"]).agg(
        n_cycles=("global_cycle", "count"),
        max_rul=(TARGET, "max"),
    ))
    df.to_csv("results/combined_dataset.csv", index=False)
    print("saved results/combined_dataset.csv")
