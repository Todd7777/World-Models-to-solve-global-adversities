"""Build annual SSP1-5 GDP/population series (both vintages) for the panel countries.

Inputs (fetched from JGCRI's open-source GCAM data system):
  data/raw/ssp_v9/SSP_database_v9.csv            (JGCRI/gcamdata, SSP db v9, Feb 2014)
  data/raw/ssp_2025/SSP_database_2025.csv.gz     (JGCRI/gcam-core, SSP db v3.2, Jun 2025)
  data/raw/ssp_2025/iso_SSP_regID.csv            (country name -> ISO3 map)
Output (committed, small): data/processed/ssp_series.npz + ssp_series_meta.json

Usage:  python scripts/01b_build_ssp.py
"""
import numpy as np

from carbon_world_model.config import default_config
from carbon_world_model.data.panel import Panel
from carbon_world_model.data.ssp import SSPS, build_series, save_processed


def main() -> None:
    cfg = default_config()
    panel = Panel.load(cfg)
    assert panel.iso_codes, "panel has no ISO codes; rebuild it with scripts/01_build_dataset.py"
    out = {}
    for v in ("v9", "v3.2"):
        s = build_series(panel.iso_codes, v)
        out[v] = s
        missing = [(SSPS[si], panel.countries[ci]) for si in range(len(SSPS))
                   for ci in range(len(panel.countries)) if not s.avail[si, ci]]
        fallbacks = sorted({panel.countries[ci] for si in range(len(SSPS))
                            for ci in range(len(panel.countries)) if "fallback" in s.source[si][ci]})
        print(f"[{v}] available {int(s.avail.sum())}/{s.avail.size} (scenario,country) pairs; "
              f"missing={missing[:6]}{'...' if len(missing) > 6 else ''}; fallbacks={fallbacks}")
    save_processed(out, panel.iso_codes)
    print("[ok] saved data/processed/ssp_series.npz")


if __name__ == "__main__":
    main()
