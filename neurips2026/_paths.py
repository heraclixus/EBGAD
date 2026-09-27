"""Put the repository root (data_utils, soc, bridge, ebgad) and the script folders on sys.path, so that every
script here runs from any working directory. Relative output paths (results/, cache/, figures/) are resolved
against the current directory: run from the repository root."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "baselines"), os.path.join(ROOT, "neurips2026"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
