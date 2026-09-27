"""Generate standalone green formula boxes for Figure 1 assembly.

These PDFs are meant to be imported into Keynote/PowerPoint and positioned on
top of the original Figure 1 artwork. They do not regenerate the full figure.
"""

import _paths  # noqa: F401  (puts the repository root on sys.path)

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
OUT_DIRS = [
    ROOT / "tex" / "figures" / "fig1_formula_boxes",
    ROOT.parent / "figures" / "fig1_formula_boxes",
]

BOXES = {
    "gou_sde_box": {
        "fontsize": 26,
        "formula": r"""
\(\displaystyle
dX_t=-\alpha(t)\bar Q_\rho(X_t-\mathbf m)\,dt+\sqrt{2\alpha(t)}\,dW_t,
\; X_0=0 .
\)
""",
    },
    "finite_time_mean_mu_gamma_box": {
        "fontsize": 22,
        "formula": r"""
\(\displaystyle
\begin{aligned}
X_\Gamma\mid X_0=0,\mathbf m&\sim \mathcal N(\mu_\Gamma,\Sigma_\Gamma)\\[4pt]
\mu_\Gamma&=(I-e^{-\Gamma\bar Q_\rho})\mathbf m
\end{aligned}
\)
""",
    },
    "finite_time_mean_mu_t_box": {
        "fontsize": 22,
        "formula": r"""
\(\displaystyle
\begin{aligned}
X_t\mid X_0=0,\mathbf m&\sim \mathcal N(\mu_t,\Sigma_t)\\[4pt]
\mu_t&=(I-e^{-\Gamma_t\bar Q_\rho})\mathbf m,\quad
\Gamma_t=\int_0^t\alpha(s)\,ds
\end{aligned}
\)
""",
    },
    "eb_likelihood_box": {
        "fontsize": 24,
        "formula": r"""
\(\displaystyle
\ell_{\mathrm{EB}}(\rho,\kappa;\gamma)
=-\frac12\sum_j q_jS_j+\frac d2\sum_j\log q_j+\mathrm{const}
\)
""",
    },
    "score_energy_compact_box": {
        "fontsize": 24,
        "formula": r"""
\(\displaystyle
C_i(\Gamma,\lambda_c)
=\frac12\left\|[W_{\Gamma,\lambda_c}^{1/2}\Delta_\Gamma]_{i,:}\right\|^2
\)
""",
    },
    "score_energy_expanded_box": {
        "fontsize": 20,
        "formula": r"""
\(\displaystyle
\begin{aligned}
C_i(\Gamma,\lambda_c)
&=\frac12\left\|
\sum_j\sqrt{c_j(\Gamma,\lambda_c)}[v_j]_i
\bigl(v_j^\top\Delta_\Gamma\bigr)
\right\|^2
\end{aligned}
\)
""",
    },
    "score_ratio_box": {
        "fontsize": 25,
        "formula": r"""
\(\displaystyle
CR_i(\Gamma,\lambda_c)
=\frac{C_i(\Gamma,\lambda_c)}{\|\Delta_{\Gamma,i}\|^2+\epsilon}
\)
""",
    },
    "score_pair_box": {
        "fontsize": 21,
        "formula": r"""
\(\displaystyle
\begin{aligned}
C_i&=\frac12\left\|[W_{\Gamma,\lambda_c}^{1/2}\Delta_\Gamma]_{i,:}\right\|^2,\\[4pt]
CR_i&=\frac{C_i}{\|\Delta_{\Gamma,i}\|^2+\epsilon}
\end{aligned}
\)
""",
    },
}

TEMPLATE = r"""\documentclass[tikz,border=0pt]{{standalone}}
\usepackage{{amsmath,amssymb}}
\usepackage{{xcolor}}
\definecolor{{boxgreen}}{{RGB}}{{31,142,43}}
\definecolor{{boxfill}}{{RGB}}{{238,248,239}}
\pagestyle{{empty}}
\begin{{document}}
\begin{{tikzpicture}}
\node[
  draw=boxgreen,
  fill=boxfill,
  line width=2.2pt,
  inner xsep=8pt,
  inner ysep=5pt,
  align=center
] {{
\fontsize{{{fontsize}}}{{{lineheight}}}\selectfont
{formula}
}};
\end{{tikzpicture}}
\end{{document}}
"""


def build_box(out_dir: Path, name: str, spec: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{name}.tex"
    lineheight = int(spec["fontsize"] * 1.2)
    tex_path.write_text(
        TEMPLATE.format(
            fontsize=spec["fontsize"],
            lineheight=lineheight,
            formula=spec["formula"].strip(),
        )
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=out_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for suffix in [".aux", ".log"]:
        path = out_dir / f"{name}{suffix}"
        if path.exists():
            path.unlink()


def main() -> None:
    for out_dir in OUT_DIRS:
        for name, spec in BOXES.items():
            build_box(out_dir, name, spec)
        print(f"Wrote {len(BOXES)} boxes to {out_dir}")


if __name__ == "__main__":
    main()
