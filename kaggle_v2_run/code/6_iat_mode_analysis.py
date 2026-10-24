#!/usr/bin/env python3
"""
AIIS-ARP - IAT artefact analysis (Section 5.7.4, Table 5.8, Figures 5.16 and 5.17).

Input : arp_15parts.csv.gz  (Benign + MITM-ArpSpoofing records of the first 15 CICIoT2023
        partitions, with a `partition` column; produced by the Kaggle notebook)
        revision_out/revision_results.json (optional, for Figure 5.17)
Output: iat_mode_analysis.json, fig_iat_analysis_v2.png, fig_distant_stream_v2.png

Usage : python 6_iat_mode_analysis.py arp_15parts.csv.gz [revision_out/revision_results.json]
"""
import sys, json, datetime
import numpy as np, pandas as pd, xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

DATA = sys.argv[1] if len(sys.argv) > 1 else "arp_15parts.csv.gz"
RES = sys.argv[2] if len(sys.argv) > 2 else None
B, A = "BenignTraffic", "MITM-ArpSpoofing"
LARGE = 1e6                                   # boundary between the two IAT value ranges

d = pd.read_csv(DATA)
y = (d.label == A).astype(int).values
g = d.partition.values
F = [c for c in d.columns if c not in ("label", "partition")]
target = d.groupby("partition").cumcount().values >= 19          # same 106,382 targets as the thesis
big = (d.IAT > LARGE).values
out = {"n_records": int(len(d)), "n_targets": int(target.sum())}

# ---------------------------------------------------------------- descriptive statistics
out["share_large_mode"] = {c: float((d[d.label == c].IAT > LARGE).mean()) for c in (B, A)}
out["large_mode_range"] = {c: [float(d[(d.label == c) & big].IAT.min()), float(d[(d.label == c) & big].IAT.max())] for c in (B, A)}
out["large_mode_as_unix_time_x10"] = {c: [str(datetime.datetime.fromtimestamp(v * 10, datetime.UTC)) for v in out["large_mode_range"][c]] for c in (B, A)}
out["small_mode_median"] = {c: float(d[(d.label == c) & ~big].IAT.median()) for c in (B, A)}
out["share_large_mode_by_partition"] = d.assign(big=big).groupby(["partition", "label"]).big.mean().unstack().round(4).to_dict()


# ---------------------------------------------------------------- detector per IAT range
def evaluate(X):
    P = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, g):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1)
        m.fit(sc.transform(imp.transform(X[tr])), y[tr])
        tt = te[target[te]]
        P[tt] = m.predict_proba(sc.transform(imp.transform(X[tt])))[:, 1]
    res = {}
    for name, msk in [("all", target), ("small_mode", target & ~big), ("large_mode", target & big)]:
        yy, pp = y[msk], P[msk]
        pr = (pp >= .5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, pr, labels=[0, 1]).ravel()
        res[name] = dict(n=int(msk.sum()), f1=f1_score(yy, pr), tpr=tp / (tp + fn), fpr=fp / (fp + tn),
                         auc=roc_auc_score(yy, pp), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))
    return res


X = d[F].values.astype(float)
out["all_features"] = evaluate(X)
Xm = X.copy(); Xm[big, F.index("IAT")] = np.nan                  # artefact masked as missing
out["large_iat_masked"] = evaluate(Xm)
out["without_iat"] = evaluate(np.delete(X, F.index("IAT"), axis=1))
json.dump(out, open("iat_mode_analysis.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("all_features", "large_iat_masked", "without_iat")}, indent=1, default=float))

# ---------------------------------------------------------------- Figure 5.16
cb, ca = "#2c6fbb", "#c0392b"
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[0.8, 1.4, 1.4]))
sh = [out["share_large_mode"][c] * 100 for c in (B, A)]
ax[0].bar([0, 1], sh, color=[cb, ca], width=.6)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Benign", "ARP spoofing"], fontsize=8)
for i, v in enumerate(sh):
    ax[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)
ax[0].set_ylim(0, 65); ax[0].set_ylabel("Records with IAT > 10$^6$ (%)", fontsize=8)
ax[0].set_title("(a) Share in the large-value mode", fontsize=9)
L = d[big]; bins = np.linspace(1.66510e8, 1.66618e8, 120)
for c, col, lab in [(B, cb, "Benign"), (A, ca, "ARP spoofing")]:
    ax[1].hist(L[L.label == c].IAT, bins, color=col, alpha=.8, label=lab)
ax[1].xaxis.get_offset_text().set_visible(False)
ticks = [1.6652e8, 1.6654e8, 1.6656e8, 1.6658e8, 1.6660e8]
ax[1].set_xticks(ticks); ax[1].set_xticklabels([f"{t / 1e8:.4f}" for t in ticks], fontsize=7)
ax[1].set_xlabel("IAT (large-value mode, ×10$^8$)", fontsize=8); ax[1].set_ylabel("Records", fontsize=8)
ax[1].legend(fontsize=7); ax[1].set_title("(b) Large-value mode: disjoint class ranges", fontsize=9)
S = d[~big]; lb = np.linspace(-6, 1.5, 80)
for c, col, lab in [(B, cb, "Benign"), (A, ca, "ARP spoofing")]:
    v = S[S.label == c].IAT; v = v[v > 0]
    ax[2].hist(np.log10(v), lb, color=col, alpha=.6, density=True, label=lab)
ax[2].set_xlabel("log$_{10}$ IAT (small-value mode)", fontsize=8); ax[2].set_ylabel("Density", fontsize=8)
ax[2].legend(fontsize=7); ax[2].set_title("(c) Small-value mode: overlapping classes", fontsize=9)
for a in ax:
    a.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig("fig_iat_analysis_v2.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Figure 5.17 (from the Kaggle results)
if RES:
    e = json.load(open(RES))["E4_external"]["distant_stream"]
    fig, ax = plt.subplots(figsize=(8.5, 3.4)); x = np.arange(len(e["partitions"]))
    ax.plot(x, e["static_f1"], "o-", color="#7f8c8d", label="Static (trained on partitions 1–15)", ms=4)
    ax.plot(x, e["adaptive_f1"], "s-", color=cb, label="Adaptive (cumulative retraining)", ms=4)
    ax.set_xticks(x); ax.set_xticklabels(e["partitions"], fontsize=7)
    ax.set_xlabel("Unseen CICIoT2023 partition (stream order)", fontsize=8); ax.set_ylabel("F1-score", fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig("fig_distant_stream_v2.png", dpi=300); plt.close(fig)
print("done")
