"""Build the processed multi-country panel from raw OWID CSVs.

Usage:  python scripts/01_build_dataset.py
"""
from carbon_world_model.config import default_config
from carbon_world_model.data.panel import build_panel


def main() -> None:
    cfg = default_config()
    panel = build_panel(cfg, verbose=True)
    panel.save(cfg)
    print(f"[ok] saved -> {cfg.data.processed_npz}")
    print(f"[ok] meta  -> {cfg.data.meta_json}")
    print("countries:", ", ".join(panel.countries))


if __name__ == "__main__":
    main()
