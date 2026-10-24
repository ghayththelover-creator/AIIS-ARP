#!/usr/bin/env python3
"""
11_cd_diagram.py - Friedman test, Nemenyi critical difference and CD diagram (thesis 5.7.2,
Figure 5.14) from a matrix of fold-level F1 scores.

Input  : CSV with one row per fold and one column per configuration, e.g. E1_fold_f1_matrix.csv
         written by 4_revision_experiments.py (E1), or the fold-level F1 table exported from the
         original 16-configuration Kaggle notebook (first column = fold index).
Output : fig_cd_diagram.png and cd_summary.json next to the input file (or in OUT_DIR).
Usage  : python 11_cd_diagram.py E1_fold_f1_matrix.csv
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def cd_diagram(piv, path_png, alpha=0.05):
    chi, pf = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    k, n = piv.shape[1], piv.shape[0]
    q = stats.studentized_range.ppf(1 - alpha, k, 1e6) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * n))
    rs, names, half = ranks.values, ranks.index.tolist(), (k + 1) // 2
    fig, ax = plt.subplots(figsize=(10, 0.32 * half + 2.2))
    ax.set_xlim(-2.5, k + 3.5); ax.set_ylim(-(half + 1) * 0.5, 1.6); ax.axis("off")
    ax.hlines(0, 1, k, color="k", lw=1)
    for i in range(1, k + 1):
        ax.vlines(i, -0.06, 0.06, color="k"); ax.text(i, 0.15, str(i), ha="center", fontsize=8)
    for i, (nm, r) in enumerate(zip(names, rs)):
        if i < half:
            yy = -(i + 1) * 0.45; ax.plot([r, r, 0.6], [0, yy, yy], color="0.35", lw=0.8)
            ax.text(0.5, yy, f"{nm} ({r:.1f})", ha="right", va="center", fontsize=7.5)
        else:
            yy = -(k - i) * 0.45; ax.plot([r, r, k + 0.4], [0, yy, yy], color="0.35", lw=0.8)
            ax.text(k + 0.5, yy, f"({r:.1f}) {nm}", ha="left", va="center", fontsize=7.5)
    ax.plot([1, 1 + cd], [0.9, 0.9], color="#c0392b", lw=2.5)
    ax.text(1 + cd / 2, 1.05, f"CD = {cd:.2f}", ha="center", fontsize=8, color="#c0392b")
    ns = [r for r in rs if r - rs[0] <= cd]
    ax.plot([rs[0], max(ns)], [-0.15, -0.15], color="k", lw=3)
    ax.set_title(f"Critical-difference diagram (Nemenyi, alpha = {alpha}; lower mean rank = better)", fontsize=9)
    fig.tight_layout(); fig.savefig(path_png, dpi=300); plt.close(fig)
    return dict(friedman_chi2=float(chi), friedman_p=float(pf), n_folds=n, k=k, critical_difference=cd,
                mean_ranks=ranks.to_dict(), best=names[0],
                within_cd_of_best=[c for c in names if ranks[c] - rs[0] <= cd])


if __name__ == "__main__":
    src = sys.argv[1]
    piv = pd.read_csv(src, index_col=0)
    out = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(src)))
    os.makedirs(out, exist_ok=True)
    res = cd_diagram(piv, os.path.join(out, "fig_cd_diagram.png"))
    json.dump(res, open(os.path.join(out, "cd_summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
