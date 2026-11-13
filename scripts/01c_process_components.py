"""Split each country's process CO2 into cement, flaring and other industry (2022).

The panel stores only the total (data/processed/panel.npz, process_co2). The
technology-based process bound (plan/scenario.technology_process_floor) needs the split,
because cement and flaring follow different roadmaps. This reads the OWID CO2 data
(scripts/download_data.sh) and writes each country's 2022 component SHARES, which are
applied to the panel's own 2022 totals.

The OWID file is revised over time: the vintage downloaded on 2026-11-13 sums to 2260 Mt
for the 60 countries against the panel's 2244 Mt, with 15 countries differing by more than
5%. Only shares are used, so the panel totals stay as built. Missing other-industry
values (26 countries) are treated as zero.

Usage:  bash scripts/download_data.sh && python scripts/01c_process_components.py
Writes data/processed/process_components.json.
"""
import datetime
import json

import numpy as np
import pandas as pd

COLS = ("cement_co2", "flaring_co2", "other_industry_co2")
YEAR = 2022


def main() -> None:
    meta = json.load(open("data/processed/panel_meta.json"))
    iso = meta["iso_codes"]
    df = pd.read_csv("data/raw/owid_co2/owid-co2-data.csv", usecols=["iso_code", "year", *COLS])
    df = df[df.iso_code.notna() & (df.iso_code.str.len() == 3) & (df.year == YEAR)]
    d = df.set_index("iso_code").reindex(iso)[list(COLS)].fillna(0.0)
    tot = d.sum(axis=1).to_numpy()
    shares = d.to_numpy() / np.where(tot > 0, tot, 1.0)[:, None]
    shares[tot <= 0] = [1.0, 0.0, 0.0]   # no recorded process CO2: treat as cement
    out = {"year": YEAR, "source": "Our World in Data CO2 dataset (github.com/owid/co2-data), "
                                   "columns cement_co2, flaring_co2, other_industry_co2",
           "downloaded": datetime.date.today().isoformat(), "components": list(COLS),
           "iso_codes": iso, "countries": meta["countries"],
           "mt": d.round(3).to_numpy().tolist(), "shares": shares.round(6).tolist()}
    with open("data/processed/process_components.json", "w") as fh:
        json.dump(out, fh, indent=1)
    s = d.sum() / d.sum().sum()
    print(f"[ok] {len(iso)} countries; global shares " +
          ", ".join(f"{c.split('_co2')[0]} {100*v:.1f}%" for c, v in s.items()))


if __name__ == "__main__":
    main()
