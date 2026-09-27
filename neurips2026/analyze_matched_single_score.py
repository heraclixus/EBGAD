"""Analyze matched single-score results.

Protocol (fixed before reading numbers): under one shared EB fit per dataset,
  equilibrium side = NullKS pick from {J*, R};
  finite side      = NullKS pick from all single finite bank members
                     (C/CR at finite (Gamma, lambda_c), including the
                     hard-endpoint lambda_c=inf column), no fusion, no
                     aggregation, criterion identical on both sides.
Oracles reported for completeness. Delta = finite selected - equilibrium
selected.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import argparse
import json


def fam_scores(rec, key):
    return dict(rec.get(key, {}).get("per_score", {}))


def pick(pool):
    if not pool:
        return None
    name = max(pool, key=lambda n: pool[n]["nullks"])
    return {"name": name, **pool[name]}


def best(pool):
    if not pool:
        return None
    name = max(pool, key=lambda n: pool[n]["auc"])
    return {"name": name, **pool[name]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="results/horizon_study/matched_single_score.json")
    args = parser.parse_args()
    data = json.load(open(args.json))

    hdr = ("%-14s %-6s %-22s %-24s %-24s %6s | %-22s %6s"
           % ("dataset", "fitOK", "eq NullKS pick", "finite NullKS pick",
              "finite oracle", "dSel", "routed-family view", "dRout"))
    print(hdr)
    print("-" * len(hdr))
    for ds, rec in data.items():
        if "error" in rec:
            print("%-14s ERROR %s" % (ds, rec["error"][:80]))
            continue
        fit = rec["fit"]
        fit_ok = (abs(fit["rho"] - fit["table6_rho"]) <= 0.03
                  and (fit["kappa"] == fit["table6_kappa"]
                       or abs(fit["kappa"] - fit["table6_kappa"]) <= 0.01))

        energy = fam_scores(rec, "energy")
        ratio = fam_scores(rec, "ratio")
        energy_lam = fam_scores(rec, "energy_lam")
        ratio_lam = fam_scores(rec, "ratio_lam")

        eq_pool = {}
        if "J@inf" in energy:
            eq_pool["J*"] = energy["J@inf"]
        if "R@inf" in ratio:
            eq_pool["R"] = ratio["R@inf"]
        eq = pick(eq_pool)

        finite = {}
        for src in (energy, ratio):
            finite.update({n: r for n, r in src.items() if not n.endswith("@inf")})
        for src in (energy_lam, ratio_lam):
            finite.update({n: r for n, r in src.items() if not n.endswith("@inf")})
        fin_sel = pick(finite)
        fin_ora = best(finite)

        d_sel = 100 * (fin_sel["auc"] - eq["auc"]) if (fin_sel and eq) else float("nan")

        # Routed-family view: the family Table 6 selects for this dataset,
        # finite members only, NullKS pick vs that family's own limit.
        routed = {
            "weibo": ("energy", "J@inf"), "reddit": ("energy", "J@inf"),
            "amazon": ("energy_lam", "J@inf"), "yelpchi": ("energy_lam", "J@inf"),
            "blogcatalog": ("energy_lam", "J@inf"), "facebook": ("ratio_lam", "R@inf"),
            "acm": ("ratio_lam", "R@inf"), "acm_aff": ("ratio_lam", "R@inf"),
            "t_finance": ("ratio_lam", "R@inf"),
            "elliptic": ("ratio", "R@inf"), "elliptic_plus_plus": ("ratio", "R@inf"),
        }
        rkey, rlimit = routed.get(ds, ("energy", "J@inf"))
        rpool_all = fam_scores(rec, rkey)
        rlim = rpool_all.get(rlimit)
        rfin = {n: r for n, r in rpool_all.items() if not n.endswith("@inf")}
        rsel = pick(rfin)
        d_rout = 100 * (rsel["auc"] - rlim["auc"]) if (rsel and rlim) else float("nan")

        print("%-14s %-6s %-22s %-24s %-24s %+6.2f | %-22s %+6.2f" % (
            ds, "yes" if fit_ok else "NO",
            "%s=%.2f" % (eq["name"], 100 * eq["auc"]) if eq else "-",
            "%s=%.2f" % (fin_sel["name"], 100 * fin_sel["auc"]) if fin_sel else "-",
            "%s=%.2f" % (fin_ora["name"], 100 * fin_ora["auc"]) if fin_ora else "-",
            d_sel,
            "%s=%.2f vs %.2f" % (rsel["name"], 100 * rsel["auc"], 100 * rlim["auc"])
            if (rsel and rlim) else "-",
            d_rout))


if __name__ == "__main__":
    main()
