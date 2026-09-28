"""Appendix figures from the regenerated runs.

likelihood   residual-space likelihood over (rho, kappa) at the EB-fitted configuration of the run
             (likelihood-selected bandwidth), with the profile path, the selected optimum and the
             Newton iterates in rho
densities    density of the selected score for normal and anomalous nodes
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neurips2026"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "serif",
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
DISPLAY = {"weibo": "Weibo", "facebook": "Facebook", "t_finance": "T-Finance", "elliptic": "Elliptic",
           "yelpchi": "YelpChi", "reddit": "Reddit", "amazon": "Amazon"}
KAPPAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]


def likelihood(root, out, datasets):
    from data_utils import load_data
    from soc.prior_optimizer import Q_rho_eigenvalues, marginal_log_likelihood_stationary
    from test_dynamic_score_bank import prepare_model_entries
    cfgs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitted_configs.json")))
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.3 * len(datasets), 4.6),
                             gridspec_kw={"height_ratios": [2.2, 1.0]}, squeeze=False)
    record = {}
    for c, ds in enumerate(datasets):
        cfg = cfgs[ds]
        data = load_data({"yelpchi": "YelpChi", "facebook": "Facebook"}.get(ds, ds))
        e = prepare_model_entries(data, graph_types=["original"], template_gammas=[cfg["gamma"]], pca_dim=cfg["pca"],
                                  truncated_k=cfg["tk"], modeled_subspace=cfg["sub"],
                                  template_type=cfg["tmpl"], device="cpu")[0]
        S, lam, d, n = np.asarray(e["S"], np.float64), np.asarray(e["lam_L"], np.float64), int(e["d"]), len(e["lam_L"])
        rhos = np.linspace(0.0, 1.0, 101)
        kaps = np.geomspace(0.01, 40.0, 121)
        Z = np.array([[marginal_log_likelihood_stationary(S, Q_rho_eigenvalues(lam, rho=r, kappa=k, nu=1.0), n, d)
                       for k in kaps] for r in rhos])

        def newton(kappa, rho0=0.5, iters=20):
            a = (kappa ** 2 + lam) - 1.0
            rho, path = rho0, [rho0]
            for _ in range(iters):
                q = np.maximum(rho * a + 1.0, 1e-12)
                g = np.sum(a * (-S / 2.0 + d / (2.0 * q)))
                h = -np.sum(d * a ** 2 / (2.0 * q ** 2))
                if abs(h) < 1e-15:
                    break
                new = float(np.clip(rho - g / h, 0.0, 1.0))
                path.append(new)
                if abs(new - rho) < 1e-12:
                    break
                rho = new
            return path

        prof = [newton(k)[-1] for k in kaps]
        best = max(((marginal_log_likelihood_stationary(S, Q_rho_eigenvalues(lam, rho=newton(k)[-1], kappa=k, nu=1.0), n, d),
                     newton(k)[-1], k) for k in KAPPAS), key=lambda t: t[0])
        ax = axes[0, c]
        lo = np.percentile(Z, 35)
        cs = ax.contourf(kaps, rhos, np.clip(Z, lo, None), levels=30, cmap="RdYlBu_r")
        ax.plot(kaps, prof, "k--", lw=1.2, label=r"profile path $\rho^\star(\kappa)$")
        ax.plot([max(best[2], kaps[0])], [best[1]], marker="*", ms=11, mfc="white", mec="k", ls="none", label="selected")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title("%s ($\\gamma=%g$)" % (DISPLAY.get(ds, ds), cfg["gamma"]))
        ax.legend(loc="lower left", framealpha=0.9)
        path = newton(best[2])
        err = np.abs(np.array(path) - path[-1])
        ax = axes[1, c]
        ax.semilogy(range(len(err)), np.maximum(err, 1e-16), "o-", color="#c0392b", ms=3, lw=1)
        ax.set_xlabel("Newton iteration")
        ax.set_ylabel(r"$|\rho_k-\rho^\star|$")
        record[ds] = {"rho": best[1], "kappa": best[2], "iterations": len(path) - 1, "config": cfg}
    fig.tight_layout()
    fig.savefig(os.path.join(out, "likelihood_landscape.pdf"))
    json.dump(record, open(os.path.join(root, "studies", "fig_likelihood.json"), "w"), indent=1)
    print(record)


def densities(root, out, datasets):
    import select_table1 as S
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.6 * len(datasets), 2.2), squeeze=False)
    record = {}
    for c, ds in enumerate(datasets):
        r = S.run_one(root, ds, "orig", want_scores=True)
        y, mask = None, None
        df = os.path.join(root, "dumps", "hyp_%s_orig" % ds, "hyp_%s_orig_scores.npz" % ds)
        y, mask, _, _ = S.load_dump(df)
        v = np.nan_to_num(r["_vec"])[mask]
        yy = y[mask]
        from scipy.stats import rankdata
        u = (rankdata(v, method="average") - 0.5) / len(v)
        xs = np.linspace(0, 1, 200)
        ax = axes[0, c]
        for lab, col, name in ((0, "#2e86c1", "normal"), (1, "#c0392b", "anomalous")):
            k = gaussian_kde(u[yy == lab], bw_method=0.15)
            ax.fill_between(xs, k(xs), alpha=0.45, color=col, label=name)
        ax.set_title("%s (AUROC %.1f)" % (DISPLAY.get(ds, ds), r["selected"]["auroc"]))
        ax.set_xlabel("rank of the selected score")
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("density")
            ax.legend(loc="upper left")
        record[ds] = {"auroc": r["selected"]["auroc"], "score": r["score"]}
    fig.tight_layout()
    fig.savefig(os.path.join(out, "score_densities.pdf"))
    json.dump(record, open(os.path.join(root, "studies", "fig_densities.json"), "w"), indent=1)
    print(record)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["likelihood", "densities"])
    ap.add_argument("--root", default="results/camera_ready")
    ap.add_argument("--out", default="results/camera_ready/figures")
    ap.add_argument("--datasets", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what == "likelihood":
        likelihood(a.root, a.out, (a.datasets or "weibo,facebook").split(","))
    else:
        densities(a.root, a.out, (a.datasets or "t_finance,elliptic,facebook,weibo").split(","))
