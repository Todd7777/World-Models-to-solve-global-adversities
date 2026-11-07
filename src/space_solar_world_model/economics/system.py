"""Structural equations of the space-solar world: physics + design + world state -> outcomes.

Everything is written in ``jax.numpy`` so one call evaluates thousands of Monte-Carlo
worlds at once and is differentiable with respect to the design (array power,
transmitter and receiver size, constellation size), which the planner exploits.

Products ("use")
----------------
* ``beam``          electricity delivered to the grid through rectennas/receivers;
* ``space_firm``    constant electrical power for an in-orbit load (batteries ride through
                    the worst eclipse);
* ``space_asavail`` in-orbit power whenever the Sun shines (no storage).

Every intermediate quantity is a named node; ``do={"node": value}`` overrides it, which
is how the causal model performs interventions on mechanisms (e.g. "no Van Allen belts":
``do={"fluence_rate": 0.0}``). Nodes: M_dry, cost_per_kg_orbit, sk_dv_yr, fluence_rate,
transfer_fluence, debris_loss, site_availability, link_efficiency, delivered_power_bol,
capex -- plus any placement parameter (irr, worst_ecl, max_ecl_h, drag, deb10, ...).

Approximation: the beam-centre power cap is evaluated at beginning of life, so a capped
design is not credited with recovering efficiency as its array degrades (conservative).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ..constants import G0, HOURS_PER_YEAR, TSI_1AU
from ..physics.beaming import LinkTech
from ..physics.radiation import CELL_C, CELL_PHI_X
from ..placements import REF_ACCEL, Placement

MAX_YEARS = 40
YEARS = np.arange(MAX_YEARS) + 0.5
DEBRIS_1CM_DAMAGE_M2 = 0.1       # array area destroyed per >1 cm impact
LAND_REF_M2 = 1.0e8              # rectenna land cost per m^2 doubles at 100 km^2 (scarcity)


def shared_availability_series(T, q, xp=np):
    """Per-instant served probability [..., S, t] under the one-site-per-satellite limit."""
    T = xp.asarray(T)
    A = 1.0 - xp.exp(-T[..., None, None] * q)
    demand = xp.sum(A, axis=-2)
    supply = T[..., None] * xp.minimum(1.0, xp.sum(q, axis=-2))
    scale = xp.minimum(1.0, supply / xp.maximum(demand, 1e-12))
    return A * scale[..., None, :]


def shared_availability(T, q, xp=np):
    """Time-mean probability that each site is being served by a shared constellation.

    q[S, t] is the per-satellite probability of being sunlit and above site s's mask.
    Without conflicts site s is served with probability A = 1 - exp(-T q). But a
    satellite beams to one site at a time, so at every instant the number of served
    sites cannot exceed the number of satellites that see any site,
    T * min(1, sum_s q). Where that binds, all sites are scaled down together.
    """
    T = xp.asarray(T)
    A = 1.0 - xp.exp(-T[..., None, None] * q)                                 # [..., S, t]
    demand = xp.sum(A, axis=-2)                                               # [..., t]
    supply = T[..., None] * xp.minimum(1.0, xp.sum(q, axis=-2))
    scale = xp.minimum(1.0, supply / xp.maximum(demand, 1e-12))
    return xp.mean(A * scale[..., None, :], axis=-1)                          # [..., S]


def placement_vector(p: Placement, transfer: str) -> dict:
    """Flatten a placement + chosen transfer option into the scalars the model consumes."""
    t = p.transport[transfer]
    return dict(
        irr=p.illum.irradiance_factor, irr_min=p.irr_min,
        worst_ecl=p.illum.worst_eclipse_fraction, max_ecl_h=p.illum.max_eclipse_min / 60.0,
        ecl_per_yr=p.illum.eclipses_per_year,
        flu_p=p.fluence["p"], flu_e=p.fluence["e"], flu_sep=p.fluence["sep"],
        drag=p.drag_coeff, srp=p.srp_coeff, sk=p.sk_dv, disposal=p.disposal_dv,
        deb10=p.debris_10cm, deb1=p.debris_1cm,
        ep=bool(t["ep"]), dv=t["dv"], incl_mult=t["incl_mult"], chem_mult=t["mass_mult_chem"],
        days_ref=t["days_ref"], phi_p=t["phi_ref"]["p"], phi_e=t["phi_ref"]["e"],
        phi_sep=t["phi_ref"]["sep"])


def _do(do, name, value):
    return jnp.asarray(do[name], dtype=jnp.float32) if name in do else value


def system(pv: dict, u: dict, design: dict, link: dict | None = None,
           tech: LinkTech | None = None, use: str = "beam", do: dict | None = None) -> dict:
    """Evaluate the structural equations. ``u`` values may be scalars or [N] arrays."""
    do = do or {}
    pv = {**pv, **{k: v for k, v in do.items() if k in pv}}          # do() on placement physics
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    U = {k: f32(v) for k, v in u.items()}
    P = f32(design["P_sat"])                                   # BOL array power at 1 AU [W]
    years = jnp.asarray(YEARS, dtype=jnp.float32)

    # ---------------------------------------------------------------- mass model
    A_pv = P / (U["pv_eff"] * TSI_1AU)
    M_pv = P / U["pv_w_kg"]
    M_pmad = P * U["pmad_kg_kw"] * 1e-3
    M_tx = jnp.zeros_like(M_pv)
    M_ap = jnp.zeros_like(M_pv)
    if use == "beam":
        A_t = math.pi / 4.0 * f32(design["d_t"]) ** 2
        if tech.kind == "microwave":
            M_tx = P * U["tx_kg_kw"] * 1e-3
            M_ap = U["ant_kg_m2"] * A_t
            eta_tx, eta_rx = U["eta_dcrf"], U["eta_rfdc"]
        else:
            M_tx = P * U["laser_kg_kw"] * 1e-3
            M_ap = 30.0 * A_t                                  # beam-director optics
            eta_tx, eta_rx = U["eta_laser"], U["eta_laser_pv"]
    M_core = (M_pv + M_pmad + M_tx + M_ap) * (1.0 + U["struct_frac"])

    life = jnp.maximum(U["life_yr"], 3.0)
    M_batt = jnp.zeros_like(M_core)
    firm = U["irr_min_override"] if "irr_min_override" in U else f32(pv["irr_min"])
    if use == "space_firm":
        fw = f32(pv["worst_ecl"])
        firm = firm * (1.0 - fw) / ((1.0 - fw) + fw / U["eta_batt_rt"])
        n_cyc = jnp.maximum(f32(pv["ecl_per_yr"]) * life, 1.0)
        dod = jnp.minimum(0.8, 0.8 * jnp.sqrt(3000.0 / n_cyc))
        E_b = P * U["eta_pmad"] * firm * f32(pv["max_ecl_h"]) / (dod * jnp.sqrt(U["eta_batt_rt"]))
        M_batt = E_b / U["battery_wh_kg"] * (1.0 + U["struct_frac"])
    M_dry = _do(do, "M_dry", M_core + M_batt)

    # ------------------------------------------------------------- transport
    if pv["ep"]:
        mass_mult = jnp.exp(f32(pv["dv"]) / (U["isp_ep"] * G0)) * (1.0 + U["ep_tug_overhead"])
        t_days = f32(pv["days_ref"]) * REF_ACCEL / U["ep_accel"]
        acc_scale = REF_ACCEL / U["ep_accel"]
    else:
        mass_mult = f32(pv["chem_mult"])
        t_days = f32(pv["days_ref"])
        acc_scale = 0.0
    c_kg = _do(do, "cost_per_kg_orbit", U["launch_usd_kg"] * f32(pv["incl_mult"]) * mass_mult)
    M_disp = M_dry * (1.0 - jnp.exp(-f32(pv["disposal"]) / (U["isp_ep"] * G0)))

    # ------------------------------------------------------- station-keeping
    am = A_pv / M_dry
    dv_yr = _do(do, "sk_dv_yr", f32(pv["drag"]) * U["atm_mult"] * am + f32(pv["srp"]) * am
                + f32(pv["sk"]))
    m_prop_yr = M_dry * (1.0 - jnp.exp(-dv_yr / (U["isp_ep"] * G0)))

    # ----------------------------------------------------------- degradation
    phi_rate = _do(do, "fluence_rate", f32(pv["flu_p"]) * U["rad_mult_p"]
                   + f32(pv["flu_e"]) * U["rad_mult_e"] + f32(pv["flu_sep"]) * U["rad_mult_sep"])
    phi_tr = _do(do, "transfer_fluence", acc_scale * (f32(pv["phi_p"]) * U["rad_mult_p"]
                 + f32(pv["phi_e"]) * U["rad_mult_e"] + f32(pv["phi_sep"]) * U["rad_mult_sep"]))
    deb = _do(do, "debris_loss", U["debris_mult"] * (
        f32(pv["deb10"]) * jnp.minimum(A_pv, U["module_m2"]) + f32(pv["deb1"]) * DEBRIS_1CM_DAMAGE_M2))
    phi = phi_tr[..., None] + phi_rate[..., None] * years
    R_rad = jnp.clip(1.0 - CELL_C * jnp.log10(1.0 + phi / CELL_PHI_X), 0.05, 1.0)
    R = R_rad * jnp.exp(-(U["base_degr"] + deb)[..., None] * years)          # [..., Y]

    # ------------------------------------------------------ capex per satellite
    C_hw = M_dry * U["hw_usd_kg"]
    C_tr = (M_dry + M_disp) * c_kg
    C_sat = C_hw + C_tr

    out = {}
    # ------------------------------------------------------ delivered power
    if use == "beam":
        lam = tech.wavelength_m
        b2 = lambda x: x[..., None, None]                                    # batch -> [..., 1, 1]
        D, el, w = f32(link["D"]), f32(link["el"]), f32(link["w"])            # [S, K]
        sin_el = jnp.maximum(jnp.sin(el), 1e-3)
        A_r = math.pi / 4.0 * f32(design["d_r"]) ** 2 * jnp.ones_like(P)
        P_rf = P * U["eta_pmad"] * eta_tx                                    # [...]
        A_t = A_t * jnp.ones_like(P)
        lamD2 = (lam * D) ** 2
        if tech.kind == "microwave":
            cap = tech.max_peak_intensity * lamD2 / b2(A_t)                  # [..., S, K]
            if tech.max_tx_flux is not None:                                 # transmitter heat
                cap = jnp.minimum(cap, tech.max_tx_flux * b2(A_t))
            P_tx = jnp.minimum(b2(P_rf), cap)
            I0 = P_tx * b2(A_t) / lamD2
            A_min = jnp.log(jnp.maximum(I0 / tech.max_edge_intensity, 1.0)) * lamD2 / (b2(A_t) * sin_el)
            A_min = jnp.max(jnp.where(w > 0, A_min, 0.0), axis=-1)            # [..., S]
            A_r_eff = jnp.maximum(A_r[..., None], A_min)
            tau2 = b2(A_t) * A_r_eff[..., None] * sin_el / lamD2
            eta_cap = 1.0 - jnp.exp(-tau2)
        else:
            P_tx = jnp.minimum(b2(P_rf) * jnp.ones_like(D), tech.max_tx_flux * b2(A_t))
            A_rx_min = jnp.max(jnp.where(w > 0, P_tx / (tech.max_rx_flux * sin_el), 0.0), axis=-1)
            A_r_eff = jnp.maximum(A_r[..., None], A_rx_min)
            w0 = jnp.sqrt(A_t / (2.0 * math.pi))
            theta = jnp.sqrt((lam / (math.pi * w0)) ** 2 + tech.divergence_floor_rad ** 2)
            eta_cap = 1.0 - jnp.exp(-2.0 * (A_r_eff[..., None] * sin_el / math.pi) / (D * b2(theta)) ** 2)
        elc = jnp.maximum(el, 0.0)
        eta_atm = jnp.exp(math.log(1.0 - tech.zenith_loss) / (
            jnp.sin(elc) + 0.50572 * (elc * 57.29578 + 6.07995) ** -1.6364))
        P_rx = P_tx * eta_cap * eta_atm * b2(eta_rx * U["eta_grid"] * jnp.ones_like(P))
        frac = jnp.sum(w * P_rx, axis=-1) / P_rf[..., None]                  # [..., S] BOL transfer
        if link["mode"] == "shared":
            T = f32(design["T"]) * jnp.ones_like(P)
            if "avail" in link:                                              # precomputed for this T
                avail = f32(link["avail"]) * jnp.ones_like(P)[..., None]
            else:
                avail = shared_availability(T, f32(link["q"]), jnp)
            n_sats = T
        else:
            avail = f32(link["qbar"]) * jnp.ones_like(P)[..., None]
            n_sats = f32(link["n_sats"]) * jnp.ones_like(P)
        avail = _do(do, "site_availability", avail) * f32(link["clear"]) * f32(link["site_mask"])
        per_site = frac * avail * U["availability"][..., None]               # [..., S]
        eff = _do(do, "link_efficiency", jnp.sum(per_site, axis=-1))
        P_del = _do(do, "delivered_power_bol", P_rf * eff)                   # network, BOL [W]
        P_peak = jnp.max(jnp.where(w > 0, P_rx, 0.0), axis=-1)               # [..., S]
        if tech.kind == "microwave":
            land = 1.0 + A_r_eff / LAND_REF_M2                               # land scarcity
            C_site = (A_r_eff * land * U["rect_usd_m2"][..., None]
                      + P_peak * 1e-3 * U["grid_usd_kw"][..., None] + U["site_fixed_usd"][..., None])
        else:
            C_site = (P_peak * 1e-3 * (U["laser_rx_usd_kw"] + U["grid_usd_kw"])[..., None]
                      + U["site_fixed_usd"][..., None])
        C_ground = jnp.sum(C_site * f32(link["site_mask"]), axis=-1)
        out.update(A_r_ground=A_r_eff, link_frac_bol=frac, site_avail=avail, link_efficiency=eff,
                   P_rx_peak=P_peak, A_t=A_t)
    else:
        n_sats = f32(1.0)
        C_ground = jnp.zeros_like(C_sat)
        if use == "space_firm":
            P_del = P * U["eta_pmad"] * firm * U["availability"]
        else:
            P_del = P * U["eta_pmad"] * f32(pv["irr"]) * U["availability"]
    P_year = P_del[..., None] * R                                            # [..., Y]

    # ---------------------------------------------------------------- finance
    capex = _do(do, "capex", n_sats * C_sat + C_ground)
    r = U["discount"]
    carry = (1.0 + r) ** (t_days / 365.25)
    mask = (years < life[..., None]).astype(jnp.float32)
    df = (1.0 + r[..., None]) ** (-years) * mask
    opex = U["om_frac"][..., None] * capex[..., None] + (n_sats * m_prop_yr * c_kg)[..., None]
    energy_mwh = P_year * HOURS_PER_YEAR * 1e-6
    pv_cost = capex * carry + jnp.sum(df * opex, axis=-1)
    pv_energy = jnp.sum(df * energy_mwh, axis=-1)
    lcoe = pv_cost / jnp.maximum(pv_energy, 1e-9)

    out.update(
        lcoe=lcoe, capex=capex, C_sat=C_sat, C_hw=C_hw, C_transport=C_tr, C_ground=C_ground,
        M_dry=M_dry, M_batt=M_batt, A_pv=A_pv, n_sats=n_sats * jnp.ones_like(P),
        cost_per_kg_orbit=c_kg, transfer_days=t_days * jnp.ones_like(P),
        sk_dv_yr=dv_yr, prop_kg_yr=m_prop_yr, fluence_rate=phi_rate, transfer_fluence=phi_tr,
        debris_loss=deb, R_year1=R[..., 0], R_traj=R,
        R_eol=jnp.sum(R * mask, axis=-1) / jnp.sum(mask, axis=-1),
        P_delivered_bol=P_del, energy_mwh_yr1=energy_mwh[..., 0],
        lifetime_energy_mwh=jnp.sum(energy_mwh * mask, axis=-1),
        pv_cost=pv_cost, pv_energy=pv_energy,
        capex_per_kw=capex / jnp.maximum(P_del * 1e-3, 1e-9),
        kg_per_kw=n_sats * M_dry / jnp.maximum(P_del * 1e-3, 1e-9),
        catastrophic_impacts_yr=n_sats * U["debris_mult"] * f32(pv["deb10"]) * A_pv,
    )
    return out
