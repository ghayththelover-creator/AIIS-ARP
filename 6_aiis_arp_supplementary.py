#!/usr/bin/env python3
"""
AIIS-ARP - supplementary analyses for the reviewer response (S1-S8).

Run on Kaggle with the "UNB CIC IOT 2023 Dataset" attached (CPU is enough).
Outputs go to OUT_DIR as CSV files plus a summary report (supp_report.md).

  S1  per-partition record counts
  S2  exact / near-duplicate screening across partitions and folds
  S3  per-fold confusion matrices of AIIS-ARP (XGB, T=1); pooled vs mean-of-fold metrics
  S4  adaptivity: static vs cumulative vs size-matched vs recent-window retraining
  S5  partition-level paired bootstrap CIs for F1 differences (T=1 models,
      plus any extra out-of-fold prediction files you provide, e.g. fusion/LSTM)
  S6  IAT distribution across partitions (per class, KS distance to the rest)
  S7  deletion test under alternative baselines (mean, median, benign mean, benign donor)
  S8  negative controls: share of other CICIoT2023 attack classes flagged as ARP spoofing

Settings mirror the manuscript (Table 2): XGBoost 400 trees, depth 6, lr 0.05,
subsample 0.9, colsample 0.8; RF 200 trees; GroupKFold(5) by partition; median
imputation + z-score fitted on training folds only; threshold 0.5; targets =
records with >= 19 predecessors in their partition (106,382 for the 15 partitions).
"""
import glob, os, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- config
DATA_GLOB = os.environ.get("DATA_GLOB", os.environ.get("CICIOT_GLOB", "/kaggle/input/**/part-*.csv"))
N_PARTS = 15                      # first 15 partitions in sorted file order
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working/supp")
BENIGN, ARP = "BenignTraffic", "MITM-ArpSpoofing"
LABEL_COL = "label"
MIN_PRED = 19                     # targets need >= 19 predecessors (T = 20)
SEED = 42
N_BOOT = 2000
N_ALERTS = 200                    # alerts used in the deletion test
NEG_PER_CLASS = 3000              # max other-attack records scored per class per fold
# Optional extra OOF prediction files for S5: CSV with columns
#   target_id, prob   (target_id = "<partition>_<row index in partition>")
EXTRA_OOF = {
    # "Fusion LSTM-XGB, T=20": "/kaggle/input/aiis-oof/fusion_t20.csv",
    # "LSTM, T=20": "/kaggle/input/aiis-oof/lstm_t20.csv",
}
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
report = ["# AIIS-ARP supplementary analyses\n",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n"]


def log(s=""):
    print(s)
    report.append(s)


def xgb():
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="logloss", random_state=SEED, n_jobs=-1)


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                Acc=(tp + tn) / len(y), Prec=tp / max(tp + fp, 1),
                TPR=tp / max(tp + fn, 1), FPR=fp / max(fp + tn, 1),
                F1=f1_score(y, yhat), MCC=matthews_corrcoef(y, yhat),
                ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p))


# ----------------------------------------------------------------- load
files = sorted(glob.glob(DATA_GLOB, recursive=True))[:N_PARTS]
assert len(files) == N_PARTS, f"found {len(files)} partition files with {DATA_GLOB}"
frames, others = [], []
for k, f in enumerate(files):
    df = pd.read_csv(f).replace([np.inf, -np.inf], np.nan)
    df["partition"] = k
    df["pfile"] = os.path.basename(f)
    keep = df[df[LABEL_COL].isin([BENIGN, ARP])].copy()
    keep["pos"] = np.arange(len(keep))                 # position inside partition (binary subset)
    frames.append(keep)
    oth = df[~df[LABEL_COL].isin([BENIGN, ARP])]
    if len(oth):
        oth = pd.concat([g.sample(min(len(g), NEG_PER_CLASS), random_state=SEED)
                         for _, g in oth.groupby(LABEL_COL)])
        others.append(oth)
data = pd.concat(frames, ignore_index=True)
other = pd.concat(others, ignore_index=True) if others else pd.DataFrame()
FEATS = [c for c in data.columns if c not in (LABEL_COL, "partition", "pfile", "pos")]
data["y"] = (data[LABEL_COL] == ARP).astype(int)
data["target_id"] = data["partition"].astype(str) + "_" + data["pos"].astype(str)
tgt = data[data["pos"] >= MIN_PRED].reset_index(drop=True)
log(f"Partitions: {N_PARTS}; binary records: {len(data):,}; targets: {len(tgt):,}; features: {len(FEATS)}")

# ----------------------------------------------------------------- S1
s1 = (data.groupby(["partition", "pfile"])
          .agg(records=("y", "size"), attacks=("y", "sum"))
          .assign(benign=lambda d: d.records - d.attacks,
                  attack_share=lambda d: d.attacks / d.records)
          .reset_index())
s1["targets"] = tgt.groupby("partition").size().values
s1.to_csv(f"{OUT_DIR}/S1_partition_counts.csv", index=False)
log("\n## S1 Per-partition counts\n")
log(s1.to_markdown(index=False, floatfmt=".4f"))

# ----------------------------------------------------------------- folds
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(tgt, tgt["y"], groups=tgt["partition"]))
fold_parts = [sorted(tgt.loc[te, "partition"].unique().tolist()) for _, te in folds]
log("\nTest partitions per fold: " + json.dumps(fold_parts))


def prep(tr_df, te_df):
    imp = SimpleImputer(strategy="median").fit(tr_df[FEATS])
    sc = StandardScaler().fit(imp.transform(tr_df[FEATS]))
    f = lambda d: sc.transform(imp.transform(d[FEATS]))
    return f(tr_df), f(te_df), imp, sc

# ----------------------------------------------------------------- S2
log("\n## S2 Duplicate screening\n")
Xr = data[FEATS].round(6)
h = pd.util.hash_pandas_object(Xr, index=False)
data["h"] = h.values
dup_within = data.duplicated(["h"], keep=False)
cross = data.groupby("h")["partition"].nunique()
cross_h = set(cross[cross > 1].index)
log(f"- Records with an exact feature-vector duplicate anywhere: {dup_within.sum():,} ({dup_within.mean():.2%})")
log(f"- Records whose exact duplicate occurs in another partition: {data['h'].isin(cross_h).sum():,}")
conf = data.groupby("h")["y"].nunique()
log(f"- Feature vectors carrying both labels (label conflicts): {(conf > 1).sum():,}")
tgt = tgt.merge(data[["target_id", "h"]], on="target_id", how="left")
rows = []
for k, (tr, te) in enumerate(folds):
    trh = set(tgt.loc[tr, "h"])
    te_dup = tgt.loc[te, "h"].isin(trh)
    Xtr, Xte, _, _ = prep(tgt.loc[tr], tgt.loc[te])
    samp = rng.choice(len(te), size=min(2000, len(te)), replace=False)
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
    d_te = nn.kneighbors(Xte[samp])[0][:, 0]
    tr_s = rng.choice(len(tr), size=min(2000, len(tr)), replace=False)
    d_tr = NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr[tr_s])[0][:, 1]
    eps = np.quantile(d_tr, 0.01)
    rows.append(dict(fold=k + 1, test_records=len(te),
                     exact_dup_in_train=int(te_dup.sum()), exact_dup_share=te_dup.mean(),
                     median_nn_dist_test_to_train=np.median(d_te),
                     median_nn_dist_train_to_train=np.median(d_tr),
                     near_dup_share_test=(d_te <= eps).mean(), eps_1pct_train=eps))
s2 = pd.DataFrame(rows)
s2.to_csv(f"{OUT_DIR}/S2_duplicates_by_fold.csv", index=False)
log(s2.to_markdown(index=False, floatfmt=".4f"))
log("Near-duplicate = test record whose nearest training neighbour (z-scored space) is closer than the 1st percentile of within-training nearest-neighbour distances.")

# ----------------------------------------------------------------- S3 (+ OOF for S5)
log("\n## S3 Per-fold confusion matrices (XGB, T=1)\n")
models = {
    "XGB, T=1": xgb,
    "RF, T=1": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
    "DT, T=1": lambda: DecisionTreeClassifier(random_state=SEED),
    "LR, T=1": lambda: LogisticRegression(max_iter=2000),
}
oof = {m: np.zeros(len(tgt)) for m in models}
fold_rows, cache = [], {}
for k, (tr, te) in enumerate(folds):
    Xtr, Xte, imp, sc = prep(tgt.loc[tr], tgt.loc[te])
    ytr, yte = tgt.loc[tr, "y"].values, tgt.loc[te, "y"].values
    for m, mk in models.items():
        clf = mk().fit(Xtr, ytr)
        oof[m][te] = clf.predict_proba(Xte)[:, 1]
        if m == "XGB, T=1":
            r = metrics(yte, oof[m][te]); r["fold"] = k + 1
            fold_rows.append(r)
            cache[k] = (clf, Xtr, Xte, ytr, yte, imp, sc)
s3 = pd.DataFrame(fold_rows)[["fold", "TP", "FN", "FP", "TN", "Acc", "Prec", "TPR", "FPR", "F1", "MCC", "ROC_AUC", "PR_AUC"]]
pooled = metrics(tgt["y"].values, oof["XGB, T=1"])
s3.to_csv(f"{OUT_DIR}/S3_per_fold_confusion_xgb_t1.csv", index=False)
log(s3.to_markdown(index=False, floatfmt=".4f"))
cmp_ = pd.DataFrame({"mean_of_folds": s3[["Acc", "Prec", "TPR", "FPR", "F1", "MCC", "ROC_AUC", "PR_AUC"]].mean(),
                     "pooled_OOF": pd.Series({k: pooled[k] for k in ["Acc", "Prec", "TPR", "FPR", "F1", "MCC", "ROC_AUC", "PR_AUC"]})})
cmp_.to_csv(f"{OUT_DIR}/S3_pooled_vs_mean.csv")
log("\nPooled vs mean-of-fold:\n")
log(cmp_.to_markdown(floatfmt=".4f"))
pd.DataFrame({"target_id": tgt["target_id"], "y": tgt["y"], **{m: oof[m] for m in models}}) \
  .to_csv(f"{OUT_DIR}/oof_predictions_t1.csv", index=False)

# ----------------------------------------------------------------- S5
log("\n## S5 Partition-level paired bootstrap of F1 differences (AIIS-ARP minus competitor)\n")
for name, path in EXTRA_OOF.items():
    e = pd.read_csv(path)[["target_id", "prob"]]
    oof[name] = tgt[["target_id"]].merge(e, on="target_id", how="left")["prob"].values
y_all, part = tgt["y"].values, tgt["partition"].values
idx_by_part = {p: np.where(part == p)[0] for p in np.unique(part)}
base = (oof["XGB, T=1"] >= 0.5).astype(int)
rows = []
for m in oof:
    if m == "XGB, T=1" or np.isnan(oof[m]).any():
        continue
    comp = (oof[m] >= 0.5).astype(int)
    d_obs = f1_score(y_all, base) - f1_score(y_all, comp)
    boots = []
    plist = list(idx_by_part)
    for _ in range(N_BOOT):
        pick = rng.choice(plist, size=len(plist), replace=True)
        ii = np.concatenate([idx_by_part[p] for p in pick])
        boots.append(f1_score(y_all[ii], base[ii]) - f1_score(y_all[ii], comp[ii]))
    per_p = [f1_score(y_all[i], base[i]) - f1_score(y_all[i], comp[i]) for i in idx_by_part.values()]
    w = stats.wilcoxon(per_p) if np.any(np.array(per_p) != 0) else None
    rows.append(dict(competitor=m, dF1=d_obs, ci_low=np.percentile(boots, 2.5),
                     ci_high=np.percentile(boots, 97.5),
                     partitions_favouring_AIIS=int(np.sum(np.array(per_p) > 0)),
                     wilcoxon_p_15_partitions=(w.pvalue if w else np.nan)))
s5 = pd.DataFrame(rows)
s5.to_csv(f"{OUT_DIR}/S5_bootstrap_dF1.csv", index=False)
log(s5.to_markdown(index=False, floatfmt=".4f"))
log(f"{N_BOOT} bootstrap resamples of the 15 partitions (cluster bootstrap); 95% percentile intervals.")

# ----------------------------------------------------------------- S4
log("\n## S4 Adaptivity with size-matched and recent-window baselines\n")
parts = sorted(tgt["partition"].unique())
static_parts = parts[:3]
static_size = int(tgt["partition"].isin(static_parts).sum())


def fit_eval(train_df, test_df):
    Xtr, Xte, _, _ = prep(train_df, test_df)
    clf = xgb().fit(Xtr, train_df["y"].values)
    return f1_score(test_df["y"].values, (clf.predict_proba(Xte)[:, 1] >= 0.5).astype(int))


rows = []
for b in parts[3:]:
    test_df = tgt[tgt["partition"] == b]
    seen = [p for p in parts if p < b]
    cum = tgt[tgt["partition"].isin(seen)]
    rows.append(dict(batch=b + 1,
                     static=fit_eval(tgt[tgt["partition"].isin(static_parts)], test_df),
                     cumulative=fit_eval(cum, test_df),
                     size_matched=fit_eval(cum.sample(static_size, random_state=SEED + b), test_df),
                     recent_window_3=fit_eval(tgt[tgt["partition"].isin(seen[-3:])], test_df)))
s4 = pd.DataFrame(rows)
s4.to_csv(f"{OUT_DIR}/S4_adaptivity_baselines.csv", index=False)
log(s4.to_markdown(index=False, floatfmt=".4f"))
for c in ["cumulative", "size_matched", "recent_window_3"]:
    d = s4[c] - s4["static"]
    bs = [rng.choice(d.values, len(d)).mean() for _ in range(N_BOOT)]
    nz = d[d != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) > 0 else np.nan
    log(f"- {c} - static: mean dF1 = {d.mean():+.4f} (95% bootstrap CI {np.percentile(bs, 2.5):+.4f} to {np.percentile(bs, 97.5):+.4f}); Wilcoxon p = {p:.4f}")
log("Size-matched = random sample of all partitions seen so far with the same number of records as the static model's training set; recent-window = the last three partitions seen.")

# ----------------------------------------------------------------- S6
log("\n## S6 IAT distribution across partitions\n")
rows = []
for cls, lab in [(0, "benign"), (1, "arp")]:
    sub = data[data["y"] == cls]
    for p in parts:
        a = sub.loc[sub["partition"] == p, "IAT"].dropna()
        b_ = sub.loc[sub["partition"] != p, "IAT"].dropna()
        rows.append(dict(cls=lab, partition=p + 1, n=len(a), median=a.median(),
                         q25=a.quantile(.25), q75=a.quantile(.75),
                         KS_vs_rest=stats.ks_2samp(a, b_).statistic))
s6 = pd.DataFrame(rows)
s6.to_csv(f"{OUT_DIR}/S6_iat_by_partition.csv", index=False)
summ = s6.groupby("cls").agg(median_min=("median", "min"), median_max=("median", "max"),
                            KS_mean=("KS_vs_rest", "mean"), KS_max=("KS_vs_rest", "max"))
log(summ.to_markdown(floatfmt=".4f"))
ks_cls = stats.ks_2samp(data.loc[data.y == 0, "IAT"].dropna(), data.loc[data.y == 1, "IAT"].dropna()).statistic
log(f"KS distance benign vs ARP (all partitions): {ks_cls:.4f}. Compare with the within-class KS_vs_rest values above.")

# ----------------------------------------------------------------- S7
log("\n## S7 Deletion test under alternative baselines (fold 1)\n")
clf, Xtr, Xte, ytr, yte, imp, sc = cache[0]
p_te = clf.predict_proba(Xte)[:, 1]
alerts = np.where(p_te >= 0.5)[0]
alerts = rng.choice(alerts, size=min(N_ALERTS, len(alerts)), replace=False)
expl = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent", model_output="raw")
sv = expl.shap_values(Xte[alerts])
log(f"TreeExplainer config: feature_perturbation={expl.feature_perturbation}, model_output={expl.model_output}")
chk = np.abs(sv.sum(1) + expl.expected_value - clf.predict(Xte[alerts], output_margin=True)).max()
log(f"Max |sum(SHAP)+E[f] - log-odds margin| = {chk:.2e} (additivity in log-odds)")
benign_tr = Xtr[ytr == 0]
baselines = {"mean (0)": np.zeros(Xtr.shape[1]), "median": np.median(Xtr, 0),
             "benign mean": benign_tr.mean(0)}
rows = []
for k_ in [1, 3, 5, 10]:
    top = np.argsort(-np.abs(sv), 1)[:, :k_]
    rnd = np.array([rng.choice(Xte.shape[1], k_, replace=False) for _ in alerts])
    for bname in list(baselines) + ["benign donor"]:
        res = {}
        for tag, cols in [("top", top), ("random", rnd)]:
            Xm = Xte[alerts].copy()
            for i in range(len(alerts)):
                if bname == "benign donor":
                    donor = benign_tr[rng.integers(len(benign_tr))]
                    Xm[i, cols[i]] = donor[cols[i]]
                else:
                    Xm[i, cols[i]] = baselines[bname][cols[i]]
            res[tag] = (p_te[alerts] - clf.predict_proba(Xm)[:, 1]).mean()
        rows.append(dict(k=k_, baseline=bname, drop_top=res["top"], drop_random=res["random"]))
s7 = pd.DataFrame(rows)
s7.to_csv(f"{OUT_DIR}/S7_deletion_baselines.csv", index=False)
log(s7.to_markdown(index=False, floatfmt=".4f"))
log("'benign donor' replaces the deleted features with the values of a real benign training record (a realistic, in-distribution perturbation).")

# ----------------------------------------------------------------- S8
log("\n## S8 Negative controls: other attack classes scored by AIIS-ARP\n")
if len(other):
    rows = []
    for k, (tr, te) in enumerate(folds):
        clf, *_ , imp, sc = cache[k]
        o = other[other["partition"].isin(fold_parts[k])]
        if not len(o):
            continue
        po = clf.predict_proba(sc.transform(imp.transform(o[FEATS])))[:, 1]
        for lab, g in pd.DataFrame({"lab": o[LABEL_COL].values, "flag": po >= 0.5}).groupby("lab"):
            rows.append(dict(fold=k + 1, attack=lab, n=len(g), flagged_as_ARP=g["flag"].mean()))
    r8 = pd.DataFrame(rows)
    r8["flagged"] = r8["n"] * r8["flagged_as_ARP"]
    s8 = r8.groupby("attack", as_index=False)[["n", "flagged"]].sum()
    s8["share_flagged_as_ARP"] = s8["flagged"] / s8["n"]
    s8 = s8.drop(columns="flagged").sort_values("share_flagged_as_ARP", ascending=False)
    s8.to_csv(f"{OUT_DIR}/S8_negative_controls.csv", index=False)
    log(s8.to_markdown(index=False, floatfmt=".4f"))
    log(f"Up to {NEG_PER_CLASS} records per attack class per partition, scored by the fold model whose test set contains that partition.")
else:
    log("No other attack classes found in the selected partitions.")

with open(f"{OUT_DIR}/supp_report.md", "w") as fh:
    fh.write("\n".join(report))
print(f"\nDone. Report: {OUT_DIR}/supp_report.md")
