"""The space-solar world: placements x sites x links, evaluated by the structural model.

``SpaceSolarWorld`` is the object an agent plans in. It owns the placement catalog, the
(cached) link-coverage statistics for every placement/site pair and the scenario priors,
and evaluates any *action* -- where to put the arrays, what to do with the power, how to
get there and how big to build -- in one world or in thousands of sampled worlds.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, replace

import numpy as np

from . import exogenous as EX
from .economics.system import placement_vector, system
from .physics.beaming import LINKS
from .physics.coverage import (K_SAMPLES, SiteLink, geo_link, heo_link, l1_link,
                               shared_constellation_link)
from .placements import Placement, standard_catalog
from .scenario import SCENARIOS, Scenario
from .sites import SITES, network

DEFAULT_CACHE = os.path.join("results", "space_solar", "cache")
DEDICATED = {"geo": 1, "tundra": 3, "molniya": 3, "l1": 1}
BEAM_FAMILIES = ("leo", "sso", "meo", "geo", "heo", "lagrange")


@dataclass(frozen=True)
class Action:
    """A complete decision: placement, product, link technology, transfer and design."""

    placement: str
    use: str = "beam"                   # beam | space_firm | space_asavail
    tech: str | None = "mw5.8"          # mw2.45 | mw5.8 | laser (beam only)
    transfer: str = "sep_tug"
    site: str | None = "Las Vegas"      # target site (beam only)
    n_sites: int = 1                    # receiving sites (shared constellations)
    design: tuple = ()                  # ((name, value), ...)

    def design_dict(self) -> dict:
        return dict(self.design)

    def with_design(self, **kw) -> "Action":
        d = self.design_dict()
        d.update(kw)
        return replace(self, design=tuple(sorted(d.items())))

    @property
    def mode(self) -> str:
        return "dedicated" if self.placement in DEDICATED else "shared"

    def label(self) -> str:
        if self.use != "beam":
            return f"{self.placement} [{self.use}]"
        extra = f", {self.n_sites} sites" if self.mode == "shared" else ""
        return f"{self.placement} -> {self.site} [{self.tech}{extra}]"


class SpaceSolarWorld:
    def __init__(self, scenario: str | Scenario = "starship", catalog: dict | None = None,
                 cache_dir: str | None = DEFAULT_CACHE, el_min_shared: float = 20.0,
                 el_min_geo: float = 10.0):
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.catalog: dict[str, Placement] = catalog if catalog is not None else standard_catalog()
        self.cache_dir = cache_dir
        self.el_min_shared = el_min_shared
        self.el_min_geo = el_min_geo
        self._links: dict = {}

    # ------------------------------------------------------------------ physics
    def placement(self, key: str) -> Placement:
        return self.catalog[key]

    def with_scenario(self, scenario: str | Scenario) -> "SpaceSolarWorld":
        w = SpaceSolarWorld(scenario, self.catalog, self.cache_dir, self.el_min_shared, self.el_min_geo)
        w._links = self._links
        return w

    def site_link(self, key: str, site_name: str) -> SiteLink:
        ck = (key, site_name)
        if ck in self._links:
            return self._links[ck]
        path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            tag = hashlib.md5(f"{key}|{site_name}|{self.el_min_shared}|{self.el_min_geo}|v1".encode()).hexdigest()[:10]
            path = os.path.join(self.cache_dir, f"link_{key}_{site_name.replace(' ', '_')}_{tag}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    self._links[ck] = pickle.load(fh)
                return self._links[ck]
        site = SITES[site_name]
        p = self.catalog[key]
        if key == "geo":
            sl = geo_link(site, el_min_deg=self.el_min_geo)
        elif key in ("tundra", "molniya"):
            sl = heo_link(key, site, n_sats=DEDICATED[key], el_min_deg=self.el_min_shared)
        elif key == "l1":
            sl = l1_link(site, el_min_deg=self.el_min_shared)
        elif p.family in ("leo", "sso", "meo"):
            ls = shared_constellation_link(p.alt_km, [site], inc_deg=None if p.ltan_hours else p.inc_deg,
                                           ltan_hours=p.ltan_hours, el_min_deg=self.el_min_shared,
                                           placement=key)
            sl = ls.sites[0]
        else:
            raise ValueError(f"{key} cannot beam to Earth in this model")
        self._links[ck] = sl
        if path:
            with open(path, "wb") as fh:
                pickle.dump(sl, fh)
        return sl

    def link_arrays(self, action: Action) -> dict:
        tech = LINKS[action.tech]
        if action.mode == "dedicated":
            sites = [SITES[action.site]]
        else:
            sites = network(action.site, action.n_sites)
        sls = [self.site_link(action.placement, s.name) for s in sites]
        S, K = len(sls), K_SAMPLES
        D = np.ones((S, K)); el = np.full((S, K), np.pi / 2); w = np.zeros((S, K))
        for i, sl in enumerate(sls):
            k = len(sl.D)
            D[i, :k], el[i, :k], w[i, :k] = sl.D, sl.el, sl.w
        clear = np.array([s.site.clear_los if tech.weather_blocked else 1.0 for s in sls])
        out = dict(mode=action.mode, D=D, el=el, w=w, clear=clear, site_mask=np.ones(S))
        if action.mode == "dedicated":
            out.update(qbar=np.array([sl.availability() for sl in sls]),
                       n_sats=float(DEDICATED[action.placement]))
        else:
            out.update(q=np.stack([sl.q for sl in sls]), n_sats=0.0)
        return out

    # -------------------------------------------------------------- evaluation
    def central_u(self) -> dict:
        return EX.central(self.scenario)

    def sample_u(self, n: int, seed: int = 0, fixed: dict | None = None) -> dict:
        return EX.sample(self.scenario, n, seed, fixed)

    def outcome(self, action: Action, u: dict | None = None, do: dict | None = None,
                link: dict | None = None) -> dict:
        """Structural evaluation of ``action`` in world(s) ``u`` (default: central world)."""
        u = self.central_u() if u is None else u
        pv = placement_vector(self.catalog[action.placement], action.transfer)
        if action.use == "beam":
            link = self.link_arrays(action) if link is None else link
            tech = LINKS[action.tech]
        else:
            link, tech = None, None
        design = action.design_dict() or dict(P_sat=1e6)
        out = system(pv, u, design, link, tech, action.use, do)
        return {k: np.asarray(v) for k, v in out.items()}
