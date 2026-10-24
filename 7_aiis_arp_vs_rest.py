#!/usr/bin/env python3
"""
AIIS-ARP - ARP-vs-rest experiment (specificity fix for reviewer comment on binary scope).

Compares, on the same 15 CICIoT2023 partitions and the same partition-disjoint
GroupKFold(5) folds:
  M0  binary detector      : trained on Benign vs MITM-ArpSpoofing (as in the paper)
  M1  ARP-vs-rest detector : trained on MITM-ArpSpoofing vs {Benign + other 32 attack classes}
  M2  ARP-vs-rest, no IAT  : M1 without the IAT feature (robustness to the dominant feature)

Evaluation (test partitions only, threshold 0.5):
  E1  benign+ARP targets (same 106,382 targets as Table 3): F1, TPR, FPR on benign
  E2  other attacks: share flagged as ARP (overall and per class)
  E3  mixed test set (benign + ARP + other attacks): precision, recall, F1 for ARP
  E4  partition-level cluster bootstrap of F1 differences on the mixed set
  E5  top-10 global SHAP features of M1 (fold 1)
Outputs: /kaggle/working/supp2/arp_vs_rest_report.md + CSV files.
"""
import glob, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
DATA_GLOB = os.environ.get("DATA_GLOB", os.environ.get("CICIOT_GLOB", "/kaggle/input/**/part-*.csv"))
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working/supp2")
N_PARTS, MIN_PRED, SEED, N_BOOT = 15, 19, 42, 2000
OTHER_PER_CLASS = int(os.environ.get("OTHER_PER_CLASS", 1000))   # per class per partition
BENIGN, ARP, LABEL = "BenignTraffic", "MITM-ArpSpoofing", "label"
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
rep = ["# AIIS-ARP: ARP-vs-rest experiment\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n"]


def log(s=""):
    print(s, flush=True)
    rep.append(s)


def xgb():
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                         random_state=SEED, n_jobs=-1)


# ------------------------------------------------------------------ load
files = sorted(glob.glob(DATA_GLOB, recursive=True))[:N_PARTS]
assert len(files) == N_PARTS, f"found {len(files)} files"
bin_parts, oth_parts = [], []
for k, f in enumerate(files):
    df = pd.read_csv(f).replace([np.inf, -np.inf], np.nan)
    df["partition"] = k
    b = df[df[LABEL].isin([BENIGN, ARP])].copy()
    b["pos"] = np.arange(len(b))
    bin_parts.append(b)
    o = df[~df[LABEL].isin([BENIGN, ARP])]
    o = pd.concat([g.sample(min(len(g), OTHER_PER_CLASS), random_state=SEED + k)
                   for _, g in o.groupby(LABEL)]) if len(o) else o
    oth_parts.append(o)
    print(f"loaded {os.path.basename(f)}", flush=True)
bin_df = pd.concat(bin_parts, ignore_index=True)
oth = pd.concat(oth_parts, ignore_index=True)
FEATS = [c for c in bin_df.columns if c not in (LABEL, "partition", "pos")]
tgt = bin_df[bin_df["pos"] >= MIN_PRED].reset_index(drop=True)
tgt["y"] = (tgt[LABEL] == ARP).astype(int)
oth["y"] = 0
log(f"Targets (benign+ARP): {len(tgt):,}; other-attack records: {len(oth):,} "
    f"({oth[LABEL].nunique()} classes, up to {OTHER_PER_CLASS} per class per partition); features: {len(FEATS)}")

# same folds as the paper (GroupKFold on the benign+ARP targets)
folds = list(GroupKFold(n_splits=5).split(tgt, tgt["y"], groups=tgt["partition"]))
fold_parts = [sorted(tgt.loc[te, "partition"].unique()) for _, te in folds]
log(f"Test partitions per fold (0-based): {[list(map(int, p)) for p in fold_parts]}")

MODELS = {"M0 binary (paper)": dict(others=False, feats=FEATS),
          "M1 ARP-vs-rest": dict(others=True, feats=FEATS),
          "M2 ARP-vs-rest, no IAT": dict(others=True, feats=[c for c in FEATS if c != "IAT"])}

# out-of-fold scores
p_tgt = {m: np.full(len(tgt), np.nan) for m in MODELS}
p_oth = {m: np.full(len(oth), np.nan) for m in MODELS}
shap_m1 = None
for k, (tr, te) in enumerate(folds):
    tr_parts = set(tgt.loc[tr, "partition"])
    o_tr = oth["partition"].isin(tr_parts).values
    o_te = oth["partition"].isin(fold_parts[k]).values
    for m, cfg in MODELS.items():
        F = cfg["feats"]
        train = pd.concat([tgt.loc[tr, F + ["y"]], oth.loc[o_tr, F + ["y"]]]) if cfg["others"] else tgt.loc[tr, F + ["y"]]
        imp = SimpleImputer(strategy="median").fit(train[F])
        sc = StandardScaler().fit(imp.transform(train[F]))
        T = lambda d: sc.transform(imp.transform(d[F]))
        t0 = time.time()
        clf = xgb().fit(T(train), train["y"].values)
        p_tgt[m][te] = clf.predict_proba(T(tgt.loc[te]))[:, 1]
        p_oth[m][o_te] = clf.predict_proba(T(oth.loc[o_te]))[:, 1]
        print(f"fold {k+1} {m}: trained on {len(train):,} rows in {time.time()-t0:.0f}s", flush=True)
        if k == 0 and m == "M1 ARP-vs-rest":
            Xs = T(tgt.loc[te].sample(min(2000, len(te)), random_state=SEED))
            sv = shap.TreeExplainer(clf).shap_values(Xs)
            shap_m1 = pd.Series(np.abs(sv).mean(0), index=F).sort_values(ascending=False)

# ------------------------------------------------------------------ E1-E3
rows = []
y_t = tgt["y"].values
mixed_y = np.concatenate([y_t, np.zeros(len(oth), int)])
mixed_part = np.concatenate([tgt["partition"].values, oth["partition"].values])
for m in MODELS:
    yhat_t = (p_tgt[m] >= 0.5).astype(int)
    yhat_o = (p_oth[m] >= 0.5).astype(int)
    ben = y_t == 0
    mp = np.concatenate([p_tgt[m], p_oth[m]])
    mh = (mp >= 0.5).astype(int)
    rows.append({"model": m,
                 "F1 (benign+ARP)": f1_score(y_t, yhat_t), "TPR": recall_score(y_t, yhat_t),
                 "FPR benign (%)": 100 * yhat_t[ben].mean(),
                 "ROC-AUC (benign+ARP)": roc_auc_score(y_t, p_tgt[m]),
                 "Other attacks flagged (%)": 100 * yhat_o.mean(),
                 "Other attacks flagged, class mean (%)": 100 * pd.Series(yhat_o).groupby(oth[LABEL].values).mean().mean(),
                 "Precision (mixed)": precision_score(mixed_y, mh), "F1 (mixed)": f1_score(mixed_y, mh),
                 "PR-AUC (mixed)": average_precision_score(mixed_y, mp)})
e1 = pd.DataFrame(rows)
e1.to_csv(f"{OUT_DIR}/E1_E3_summary.csv", index=False)
log("\n## E1-E3 Summary (out-of-fold, threshold 0.5)\n")
log(e1.to_markdown(index=False, floatfmt=".4f"))
log("Mixed set = benign+ARP targets plus the sampled other-attack records of the test partitions; "
    "its class proportions depend on the sampling and are not those of the full corpus.")

e2 = pd.DataFrame({m: pd.Series((p_oth[m] >= 0.5).astype(float)).groupby(oth[LABEL].values).mean()
                   for m in MODELS})
e2.insert(0, "n", oth[LABEL].value_counts())
e2 = e2.sort_values("M0 binary (paper)", ascending=False)
e2.to_csv(f"{OUT_DIR}/E2_flagged_by_class.csv")
log("\n## E2 Share of each other attack class flagged as ARP spoofing\n")
log(e2.to_markdown(floatfmt=".4f"))

# ------------------------------------------------------------------ E4 bootstrap
log("\n## E4 Partition-level cluster bootstrap of F1 differences on the mixed set\n")
idx = {p: np.where(mixed_part == p)[0] for p in np.unique(mixed_part)}
H = {m: (np.concatenate([p_tgt[m], p_oth[m]]) >= 0.5).astype(int) for m in MODELS}
out = []
for a, b in [("M1 ARP-vs-rest", "M0 binary (paper)"), ("M2 ARP-vs-rest, no IAT", "M1 ARP-vs-rest")]:
    obs = f1_score(mixed_y, H[a]) - f1_score(mixed_y, H[b])
    bs = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[p] for p in rng.choice(list(idx), len(idx))])
        bs.append(f1_score(mixed_y[ii], H[a][ii]) - f1_score(mixed_y[ii], H[b][ii]))
    out.append({"comparison": f"{a} minus {b}", "dF1": obs,
                "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
e4 = pd.DataFrame(out)
e4.to_csv(f"{OUT_DIR}/E4_bootstrap.csv", index=False)
log(e4.to_markdown(index=False, floatfmt=".4f"))

# ------------------------------------------------------------------ E5 SHAP
log("\n## E5 Top-10 global SHAP features of M1 (fold 1, 2,000 benign+ARP test records)\n")
log(shap_m1.head(10).rename("mean |SHAP| (log-odds)").to_frame().to_markdown(floatfmt=".3f"))
shap_m1.to_csv(f"{OUT_DIR}/E5_shap_m1.csv")

pd.DataFrame({"partition": mixed_part, "y": mixed_y,
              **{m: np.concatenate([p_tgt[m], p_oth[m]]) for m in MODELS}}).to_csv(f"{OUT_DIR}/oof_mixed.csv", index=False)
with open(f"{OUT_DIR}/arp_vs_rest_report.md", "w") as fh:
    fh.write("\n".join(rep))
print(f"\nDone. Report: {OUT_DIR}/arp_vs_rest_report.md")
