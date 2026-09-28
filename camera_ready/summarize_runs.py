"""Compact summary of the regenerated runs (one line per bank)."""
import glob, json, os, sys

BANKS = ["stationary", "path_dissipation", "robust_path", "two_groups_path", "latent_mixture",
         "dynamic_profile", "dynamic_combined", "regime_gated", "anchored_override"]


def rec(bank, name):
    for key in ("top_auc", "top_ks"):
        for r in bank.get(key) or []:
            if isinstance(r, dict) and r.get("name") == name:
                return r
    return None


def main(pattern):
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        ds = list(r)[0]
        r = r[ds]
        if "stationary_fit" not in r:
            continue
        g, fit, am = r["graph"], r["stationary_fit"], r["anchored_override_meta"]
        print("== %s | n=%d D=%d h=%.3f d_e=%.2f pca=%s | fit %s rho=%.3f kappa=%g" % (
            os.path.basename(f), g["nodes"], g["dim"], g["homophily_cos"], g["density"], g["pca_dim"],
            fit["entry_key"], fit["rho"], fit["kappa"]))
        st = r["stationary"]
        for nm in ("J@inf", "R@inf"):
            x = rec(st, nm)
            if x:
                print("   eq %-6s auc=%.2f ks=%.3f" % (nm, 100 * x["auc"], x["ks"]))
        rg = r["regime_gated_meta"]
        print("   h8: static=%s (%s, ks=%.3f) sparse bj=%.1f ks=%.3f clean=%s shape clean=%s -> v2=%s" % (
            rg["static"]["pick"], rg["static"]["reason"], rg["static"]["ks_value"], rg["sparse"]["bj"],
            rg["sparse"]["ks"], rg["sparse"]["clean"], rg["shape"]["clean"], rg["regime_v2"]))
        print("   h9: regime=%s entropy clean=%s pi=%.3f | h10: regime=%s anchor_ks=%.3f" % (
            am["selected_regime"], am["entropy"]["clean"], am["entropy"]["pi"],
            am["h10"]["selected_regime"], am["h10"]["anchor_ks"]))
        for b in BANKS:
            bank = r[b]
            sel = bank.get("selectors", {})
            picks = "  ".join("%s=%s:%.1f" % (k, v["score"], 100 * v["auc"]) for k, v in sel.items()
                              if k in ("graph_tail", "tail_stable", "corr_stable") and v.get("score"))
            if bank.get("score_ks") is None:
                print("   %-18s (empty)" % b)
                continue
            print("   %-18s KS=%s:%.1f  %s  oracle=%s:%.1f" % (
                b, bank["score_ks"], 100 * bank["auc_ks_selected"], picks, bank["oracle_score"], 100 * bank["auc_oracle"]))
        for nm in ("h8_regime_v2", "h9_anchor", "h9_override", "h10_override"):
            for b in ("regime_gated", "anchored_override"):
                x = rec(r[b], nm)
                if x:
                    print("   %-14s auc=%.2f ks=%.3f" % (nm, 100 * x["auc"], x["ks"]))


if __name__ == "__main__":
    main(sys.argv[1])
