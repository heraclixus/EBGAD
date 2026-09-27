"""Regenerate Figure 1 by updating formulas in the original Keynote figure.

The paper uses the original three-panel Keynote layout. The only generated
layer is a TeX overlay that redraws the green formula boxes with the current
finite-time transport notation.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
FIGURE_DIRS = [ROOT.parent / "figures"]


def build_overlay(figure_dir: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure1_overlay.tex"],
        cwd=figure_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    shutil.copyfile(figure_dir / "figure1_overlay.pdf", figure_dir / "figure1.pdf")
    print(f"Updated {figure_dir / 'figure1.pdf'}")


def main() -> None:
    for figure_dir in FIGURE_DIRS:
        build_overlay(figure_dir)


if __name__ == "__main__":
    main()
