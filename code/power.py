"""Sample-size / power numbers for the experimental designs proposed in the report.

All formulas are standard normal approximations; they are used to size designs, not
to run tests.
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
Z = 1.959963985  # two-sided alpha=0.05
ZB = 0.8416212336  # power 0.80


def ci_half_width(p, n, z=Z):
    return z * math.sqrt(p * (1 - p) / n)


def mcnemar_detectable(n, p_discordant_null=0.10, alpha=Z, power=ZB):
    """Smallest *excess* discordance proportion detectable with a paired test."""
    # power: (p10-p01)*sqrt(n)/sqrt(p10+p01) > z_{1-a/2} + z_beta  (normal approx)
    pi = p_discordant_null
    delta = (alpha + power) * math.sqrt(pi / n)
    return delta


def kappa_n(kappa, n_cat, half_width=0.05, z=Z):
    """Approximate n for a two-rater kappa with a target CI half-width."""
    pe = 1.0 / n_cat
    p0 = pe + kappa * (1 - pe)
    # Var(kappa) ~ p0(1-p0) / (n (1-pe)^2)
    n = (z / half_width) ** 2 * (p0 * (1 - p0)) / (1 - pe) ** 2
    return math.ceil(n)


def n_for_r(r, alpha=Z, power=ZB):
    z = 0.5 * math.log((1 + r) / (1 - r))
    return math.ceil(((alpha + power) / z) ** 2 + 3)


def main():
    rep = {
        "accuracy_ci_half_width_pp": {
            str(n): round(100 * ci_half_width(0.6, n), 2)
            for n in (300, 716, 1000, 2290, 5000, 19316, 39590)
        },
        "mcnemar_min_detectable_pp": {
            str(n): round(100 * mcnemar_detectable(n), 2)
            for n in (716, 1000, 2290, 5000, 19316)
        },
        "kappa_n_for_halfwidth_0.05": {
            str(k): kappa_n(k, 12) for k in (0.4, 0.5, 0.6, 0.7, 0.8)
        },
        "kappa_n_for_halfwidth_0.10": {
            str(k): kappa_n(k, 12, half_width=0.10) for k in (0.4, 0.5, 0.6, 0.7, 0.8)
        },
        "n_for_correlation": {str(r): n_for_r(r) for r in (0.2, 0.3, 0.4, 0.5)},
    }
    # how many 5-s segments a 3.5-minute song yields
    rep["segments_per_3_5min_song"] = 42
    # CI half width on a per-song accuracy with 716 songs
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "power.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
