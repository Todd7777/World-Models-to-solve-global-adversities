#!/usr/bin/env bash
# Download the public Our World in Data (OWID) datasets used by the Carbon World Model.
# Data are released by OWID under CC-BY. Sources:
#   CO2 & GHG:  https://github.com/owid/co2-data
#   Energy:     https://github.com/owid/energy-data
#
# Usage:  bash scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW/owid_co2" "$RAW/owid_energy"

echo "Downloading OWID CO2 data..."
curl -fSL --retry 4 --retry-delay 2 \
  -o "$RAW/owid_co2/owid-co2-data.csv" \
  "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

echo "Downloading OWID energy data..."
curl -fSL --retry 4 --retry-delay 2 \
  -o "$RAW/owid_energy/owid-energy-data.csv" \
  "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

echo "Done. Files in $RAW"
ls -la "$RAW"/owid_co2 "$RAW"/owid_energy
