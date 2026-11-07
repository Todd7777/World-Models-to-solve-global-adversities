"""Ground sites that receive beamed power (and host the terrestrial counterfactual).

``ghi`` is an approximate long-term annual global horizontal irradiation [kWh/m^2/yr]
(rounded values typical of public solar atlases; not fetched from an atlas in this build)
and ``clear_los`` an approximate cloud-free line-of-sight probability, used only for
laser links. Replace both with measured site data for real decisions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float
    ghi: float          # kWh/m^2/yr
    clear_los: float    # cloud-free line-of-sight probability


SITES = {s.name: s for s in [
    Site("Las Vegas", 36.17, -115.14, 2150, 0.80),
    Site("Madrid", 40.42, -3.70, 1750, 0.65),
    Site("London", 51.51, -0.13, 1000, 0.35),
    Site("Tokyo", 35.68, 139.69, 1350, 0.45),
    Site("Singapore", 1.35, 103.82, 1650, 0.35),
    Site("Delhi", 28.61, 77.21, 1950, 0.65),
    Site("Riyadh", 24.71, 46.68, 2250, 0.85),
    Site("New York", 40.71, -74.01, 1450, 0.50),
    Site("Oslo", 59.91, 10.75, 900, 0.35),
    Site("Tromso", 69.65, 18.96, 650, 0.25),
    Site("Sydney", -33.87, 151.21, 1750, 0.60),
    Site("Johannesburg", -26.20, 28.05, 2050, 0.70),
    Site("Sao Paulo", -23.55, -46.63, 1600, 0.45),
    Site("Cairo", 30.04, 31.24, 2200, 0.85),
    Site("Lagos", 6.52, 3.38, 1700, 0.35),
    Site("Shanghai", 31.23, 121.47, 1300, 0.40),
    Site("Mexico City", 19.43, -99.13, 2000, 0.55),
    Site("Santiago", -33.45, -70.67, 1900, 0.70),
    Site("Perth", -31.95, 115.86, 2100, 0.70),
    Site("Mumbai", 19.08, 72.88, 1900, 0.50),
    Site("Beijing", 39.90, 116.40, 1500, 0.55),
    Site("Moscow", 55.76, 37.62, 950, 0.35),
    Site("Lima", -12.05, -77.04, 1800, 0.40),
    Site("Honolulu", 21.31, -157.86, 2000, 0.55),
]}

# Global receiving network for shared (LEO/MEO) constellations: the target site first,
# then the others in this order (spread in longitude and latitude).
NETWORK_ORDER = ["Las Vegas", "Madrid", "Tokyo", "Delhi", "Sao Paulo", "Sydney", "Riyadh",
                 "New York", "Shanghai", "Johannesburg", "Mexico City", "Cairo", "Perth",
                 "Lagos", "Mumbai", "Santiago", "Beijing", "Honolulu", "Lima", "London",
                 "Singapore", "Moscow", "Oslo", "Tromso"]


def network(target: str, n: int) -> list[Site]:
    order = [target] + [s for s in NETWORK_ORDER if s != target]
    return [SITES[s] for s in order[:n]]
