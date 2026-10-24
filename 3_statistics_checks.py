#!/usr/bin/env python3
"""3_statistics_checks.py - statistical checks of Chapter 5 (as listed in Appendix B)."""
import math
from scipy.stats import chi2

tp, fn, fp = 20655, 2669, 505
tn = 83058 - fp
mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))  # 0.912
print("MCC", round(mcc, 3))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


print("TPR 95% CI", wilson(tp, tp + fn), "FPR 95% CI", wilson(fp, fp + tn))
print("Friedman p", chi2.sf(74.47, 15))                       # 7.1e-11
for a, b in [(561, 498), (585, 480), (451, 243)]:              # McNemar, Table 5.2
    s = (abs(a - b) - 1) ** 2 / (a + b)
    p = chi2.sf(s, 1)
    print("McNemar", a, b, "p =", p, "Bonferroni p =", min(1, 15 * p))   # m = 15
print("exact Wilcoxon p (11 pairs)", 2 / 2 ** 11)
