"""Fit the trapped-radiation surrogate constants to the published design anchors.

Least squares in log-fluence, each residual scaled by its anchor's uncertainty band,
with weak Gaussian priors that keep the belt shapes physical (inner proton belt peak
near L~1.5, outer electron belt near L~4.5). Prints a ``RAD_PARAMS`` dict to paste into
``space_solar_world_model/physics/radiation.py``.

    python scripts/space_solar/fit_radiation.py
"""
import numpy as np
from scipy.optimize import least_squares

from space_solar_world_model.physics import radiation as R

NAMES = ["lgA_p", "L_p", "s_p", "k_p", "h_c", "w_h", "lgA_e", "L_e", "s_e", "k_e"]
PRIOR = dict(lgA_p=(16.5, 1.5), L_p=(1.55, 0.15), s_p=(0.17, 0.06), k_p=(1.0, 0.6),
             h_c=(900.0, 500.0), w_h=(180.0, 100.0), lgA_e=(15.0, 1.0), L_e=(4.5, 0.3),
             s_e=(0.15, 0.05), k_e=(1.0, 0.6))
LO = [13, 1.2, 0.08, 0.0, 200, 60, 13, 3.8, 0.08, 0.0]
HI = [19, 2.2, 0.35, 3.0, 2500, 500, 17, 5.2, 0.30, 3.0]


def to_params(x):
    d = dict(zip(NAMES, x))
    q = dict(R.RAD_PARAMS)
    q.update(A_p=10 ** d["lgA_p"], L_p=d["L_p"], s_p=d["s_p"], k_p=d["k_p"], h_c=d["h_c"],
             w_h=d["w_h"], A_e=10 ** d["lgA_e"], L_e=d["L_e"], s_e=d["s_e"], k_e=d["k_e"])
    return q


def residuals(x):
    q = to_params(x)
    res = []
    for _, alt, inc, target, band, _ in R.ANCHORS:
        f = R.orbit_fluence_rate(alt, inc, params=q, n_anom=64, n_rot=32)["total"]
        res.append(np.log(f / target) / np.log(band))
    for k, v in zip(NAMES, x):
        mu, sd = PRIOR[k]
        res.append(0.3 * (v - mu) / sd)
    return np.array(res)


if __name__ == "__main__":
    x0 = np.array([PRIOR[k][0] for k in NAMES])
    sol = least_squares(residuals, x0, bounds=(LO, HI), x_scale="jac")
    q = to_params(sol.x)
    print("RAD_PARAMS = dict(")
    print(f"    A_p={q['A_p']:.3e}, L_p={q['L_p']:.3f}, s_p={q['s_p']:.3f}, k_p={q['k_p']:.3f},")
    print(f"    A_e={q['A_e']:.3e}, L_e={q['L_e']:.3f}, s_e={q['s_e']:.3f}, k_e={q['k_e']:.3f},")
    print(f"    h_c={q['h_c']:.1f}, w_h={q['w_h']:.1f},")
    print(f"    A_s={q['A_s']:.3e}, L_c={q['L_c']:.2f},")
    print(")")
    for row in R.anchor_table(q):
        print(f"  {row['label']:18s} target {row['target']:.2e}  model {row['model']:.2e}  "
              f"ratio {row['ratio']:.2f} (band x{row['band']})")
