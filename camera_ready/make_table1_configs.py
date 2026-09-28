"""Configurations for the studies and figures that run at a configuration of the pipeline.

table1_configs.json   the configuration behind each cell of Table 1 (matched single-score comparison)
fitted_configs.json   the EB-fitted configuration of each run, at the likelihood-selected bandwidth
                      (likelihood figure)
Written from results/camera_ready/table1_ebgad.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
t = json.load(open(os.path.join(HERE, "..", "results", "camera_ready", "table1_ebgad.json")))
cells, fitted = {}, {}
for ds, v in t.items():
    if ds == "dgraph":  # spectrum of the labeled subgraph, computed by a separate script
        continue
    f = v["first"]
    c, st = f["config"], f["stats"]
    pca = None if c["pca"] in (None, "None") else int(c["pca"])
    cells[ds] = {"tmpl": "affinity" if c["template"] == "affinity" else "low", "gamma": c["gamma"], "tk": st["k"],
                 "pca": pca, "sub": "legacy", "rho": c["rho"], "kappa": c["kappa"]}
    fitted[ds] = {"tmpl": "low", "gamma": f["fit"]["gamma"], "tk": st["k"], "pca": st["pca"], "sub": "legacy",
                  "rho": f["fit"]["rho"], "kappa": f["fit"]["kappa"]}
for name, obj in (("table1_configs.json", cells), ("fitted_configs.json", fitted)):
    json.dump(obj, open(os.path.join(HERE, name), "w"), indent=1)
    print("saved", name)
