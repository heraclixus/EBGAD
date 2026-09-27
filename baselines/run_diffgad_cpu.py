"""Run DiffGAD (the authors' code, cloned into third_party/DiffGAD or DIFFGAD_DIR: DiffGAD.py, auto_encoder.py, diffusion_model.py) on CPU with the
published configuration configs/<dataset>.yaml and the constructor arguments of main.py, 20 trials as in
DiffGAD.forward (the authors' protocol; mean, std and max over trials are parsed from the script's final line).
The data are pygod's, identical to data_utils.load_data (checked: same nodes, edges and labels for Weibo); AUROC is
over all nodes, as in the authors' evaluation. .cuda() calls are made no-ops. Usage: python run_diffgad_cpu.py weibo
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

import sys, os, re, io, time, json, argparse, yaml, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFGAD_DIR = os.environ.get("DIFFGAD_DIR", os.path.join(REPO, "third_party", "DiffGAD"))   # clone of github.com/fortunato-all/DiffGAD
ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="+"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=os.path.join(REPO, "results/baselines/pygod_rerun")); ap.add_argument("--workdir", default=os.path.join(REPO, "cache/diffgad_work"))
ap.add_argument("--ae-ckpt", type=int, default=None, help="reuse the autoencoder checkpoint the authors' code selected in an earlier run (skips its 20-trial AE stage)")
ap.add_argument("--trials", type=int, default=20, help="diffusion-stage trials in this process (the authors' protocol is 20; split over processes and pooled by diffgad_pool.py)")
ap.add_argument("--tag", default="", help="suffix of the output file for a split run")
ap.add_argument("--device", default="cpu", help="cpu (the .cuda() calls become no-ops) or a CUDA device index")
args = ap.parse_args(); torch.set_num_threads(args.threads)
if args.device == "cpu":
    torch.Tensor.cuda = lambda self, *a, **k: self; torch.nn.Module.cuda = lambda self, *a, **k: self
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
sys.path.insert(0, REPO); sys.path.insert(0, DIFFGAD_DIR); os.makedirs(args.workdir, exist_ok=True); os.chdir(args.workdir)   # the authors' code writes models/ to the CWD
# the authors' DiffGAD.py with two literal edits: the diffusion-stage trial count, and one print per trial (pooled later)
src = open(os.path.join(DIFFGAD_DIR, "DiffGAD.py")).read()
src = src.replace("        num_trial = 20\n        dm_auc, dm_ap, dm_rec, dm_auprc = [], [], [], []", "        num_trial = %d\n        dm_auc, dm_ap, dm_rec, dm_auprc = [], [], [], []" % args.trials, 1)
src = src.replace("            dm_auc.append(auc_this_trial)\n", "            dm_auc.append(auc_this_trial); print('TRIAL', float(auc_this_trial), float(ap_this_trial), float(auprc_this_trial), flush=True)\n", 1)
assert src.count("num_trial = %d" % args.trials) >= 1 and "print('TRIAL'" in src
import types; mod = types.ModuleType("DiffGAD_patched"); mod.__file__ = os.path.join(DIFFGAD_DIR, "DiffGAD.py"); exec(compile(src, mod.__file__, "exec"), mod.__dict__); DiffGAD = mod.DiffGAD
from data_utils import load_data as _our_load          # our loader (identical data for pygod graphs; adds the .mat graphs and BOND's binarized labels)
mod.load_data = _our_load


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s); sys.__stdout__.flush(); return super().write(s)


for ds in args.datasets:
    cfg_path = os.path.join(DIFFGAD_DIR, "configs", ds + ".yaml"); t0 = time.time()
    if os.path.exists(cfg_path):      # the authors' published configuration, with main.py's arguments
        cfg = yaml.load(open(cfg_path), Loader=yaml.Loader)
        model = DiffGAD(lr=0.004, ae_alpha=cfg["ae_alpha"], ae_lr=cfg["ae_lr"], ae_dropout=cfg["ae_dropout"], proto_alpha=cfg["proto_alpha"], hid_dim=cfg["hid_dim"], weight=cfg["weight"])
    else:                             # nothing published for this graph: the class defaults of DiffGAD.py
        cfg = {"note": "class defaults of DiffGAD.py (no published configuration)"}; model = DiffGAD()
    if args.ae_ckpt is not None:
        model.ae_ckpt = args.ae_ckpt
    buf = Tee(); old = sys.stdout; sys.stdout = buf
    try:
        model(ds)
    finally:
        sys.stdout = old
    m = re.search(r"Final AUC: ([\d.]+)\S([\d.]+) \(([\d.]+)\).*?Final AUPRC: ([\d.]+)\S([\d.]+) \(([\d.]+)\)", buf.getvalue(), re.S)
    trials = [tuple(map(float, l.split()[1:])) for l in buf.getvalue().splitlines() if l.startswith("TRIAL ")]
    rec = dict(dataset=ds, method="DiffGAD", auroc=100 * float(m.group(1)), auroc_std=100 * float(m.group(2)), auroc_max=100 * float(m.group(3)),
               auprc=100 * float(m.group(4)), trials=[[100 * v for v in t] for t in trials], ae_ckpt=model.ae_ckpt, config={k: v for k, v in cfg.items()},
               config_provenance="%s; %d diffusion trials in this process (authors' protocol: 20); device %s" % (("published configs/%s.yaml with main.py arguments" % ds) if os.path.exists(cfg_path) else cfg["note"], args.trials, args.device), seconds=time.time() - t0)
    os.makedirs(args.out, exist_ok=True); json.dump(rec, open(os.path.join(args.out, f"diffgad_{ds}{args.tag}.json"), "w"), indent=1)
    print("%-12s DiffGAD AUROC %.1f +- %.1f (max %.1f) AUPRC %.1f | %.0f s" % (ds, rec["auroc"], rec["auroc_std"], rec["auroc_max"], rec["auprc"], rec["seconds"]), flush=True)
