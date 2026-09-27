"""Incremental result saving for optimizer scripts.

Saves JSON after every improvement, so progress is never lost on timeout.
On restart, loads existing best and continues from there.
"""
import json
import os


class IncrementalSaver:
    """Saves best result incrementally to a JSON file.

    Usage:
        saver = IncrementalSaver("results/eb_gad_v8_configs_acm.json")
        saver.load()  # loads existing best if file exists

        # In the optimization loop:
        if auc > saver.best_ce:
            saver.update_ce(auc, config)
        if auc > saver.best_cr:
            saver.update_cr(auc, config)
    """

    def __init__(self, path, dataset=""):
        self.path = path
        self.dataset = dataset
        self.best_ce = 0.0
        self.best_cr = 0.0
        self.best_ce_cfg = {}
        self.best_cr_cfg = {}
        self.configs_evaluated = 0

    def load(self):
        """Load existing results if file exists. Returns True if loaded."""
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    r = json.load(f)
                self.best_ce = r.get("best_ce", 0.0)
                self.best_cr = r.get("best_cr", 0.0)
                self.best_ce_cfg = r.get("best_ce_config", r.get("best_config", {}))
                self.best_cr_cfg = r.get("best_cr_config", {})
                self.configs_evaluated = r.get("configs_evaluated", 0)
                print("  Loaded existing: CE=%.1f%% CR=%.1f%% (%d configs)" % (
                    self.best_ce * 100, self.best_cr * 100, self.configs_evaluated))
                return True
            except:
                pass
        return False

    def update_ce(self, auc, config):
        """Update best CE score and save."""
        self.best_ce = auc
        self.best_ce_cfg = config
        self._save()

    def update_cr(self, auc, config):
        """Update best CR score and save."""
        self.best_cr = auc
        self.best_cr_cfg = config
        self._save()

    def tick(self):
        """Increment config counter."""
        self.configs_evaluated += 1

    def _save(self):
        """Write current best to disk."""
        best_auc = max(self.best_ce, self.best_cr)
        best_cfg = self.best_ce_cfg if self.best_ce >= self.best_cr else self.best_cr_cfg

        result = {
            "dataset": self.dataset,
            "best_auc": best_auc,
            "best_ce": self.best_ce,
            "best_cr": self.best_cr,
            "best_config": best_cfg,
            "best_ce_config": self.best_ce_cfg,
            "best_cr_config": self.best_cr_cfg,
            "configs_evaluated": self.configs_evaluated,
        }

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    @property
    def best_auc(self):
        return max(self.best_ce, self.best_cr)
