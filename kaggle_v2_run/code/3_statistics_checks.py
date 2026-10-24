#!/usr/bin/env python3
"""Statistics reported in Chapter 5: MCC, Wilson 95% CIs, Friedman p-value, McNemar tests with
Bonferroni adjustment (m = 15) and the exact Wilcoxon minimum p-value (11 non-tied batches)."""
import math
from scipy.stats import chi2

tp, fn, fp = 20655, 2669, 505
tn = 83058 - fp
mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
print("MCC", round(mcc, 3))                                           # 0.912


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


print("TPR CI", wilson(tp, tp + fn), "FPR CI", wilson(fp, fp + tn))
print("Friedman p", chi2.sf(74.47, 15))                              # 7.1e-10
for name, a, b in [("Fusion T=20", 561, 498), ("Fusion T=10", 585, 480), ("XGB T=5", 451, 243),
                   ("RF T=1", 600, 354), ("XGB T=20", 612, 248), ("DT T=1", 2871, 971),
                   ("Hybrid T=20", 5164, 586), ("LSTM T=20", 5267, 521), ("LR T=1", 7026, 531)]:
    s = (abs(a - b) - 1) ** 2 / (a + b); p = chi2.sf(s, 1)
    print(f"McNemar {name}: p = {p:.3g}, Bonferroni p = {min(1, 15 * p):.3g}")
print("exact Wilcoxon, 11 pairs, all positive:", 2 / 2 ** 11)            # 0.00098
