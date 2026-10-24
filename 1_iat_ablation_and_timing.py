#!/usr/bin/env python3
"""
1_iat_ablation_and_timing.py - IAT ablation (Section 5.6.1, Table 5.5) and inference timing.
Complete, runnable version of the excerpt printed in Appendix B: it rebuilds the 15-partition
subset directly from the CICIoT2023 CSV files instead of the intermediate file arp_15parts.csv.gz.
Output: 1_iat_ablation_results.json
"""
import os, sys, glob, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiis_common import environment
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

GLOB = os.environ.get("DATA_GLOB", os.environ.get("CICIOT_GLOB", "/kaggle/input/**/part-*.csv"))
parts = []
for i, f in enumerate(sorted(glob.glob(GLOB, recursive=True))[:15]):
    d = pd.read_csv(f).replace([np.inf, -np.inf], np.nan)
    d = d[d["label"].isin(["BenignTraffic", "MITM-ArpSpoofing"])].copy()
    d["partition"] = i
    parts.append(d)
big = pd.concat(parts, ignore_index=True)
y = (big["label"] == "MITM-ArpSpoofing").astype(int).values
g = big["partition"].values
X = big.drop(columns=["label", "partition"])
pos = big.groupby("partition").cumcount().values
target = pos >= 19                                  # 106,382 common target records
FOLDS = list(GroupKFold(5).split(X, y, g))          # partition-level folds


def run(drop, timing=False):
    cols = [c for c in X.columns if c not in drop]
    f1s, cm, ys, ps, res = [], np.zeros(4, int), [], [], {}
    for tr, te in FOLDS:
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr][cols])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr][cols]))
        Xtr = sc.transform(imp.transform(X.iloc[tr][cols]))
        tt = te[target[te]]
        Xte = sc.transform(imp.transform(X.iloc[tt][cols]))
        m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.8, tree_method="hist", random_state=42)
        t0 = time.perf_counter(); m.fit(Xtr, y[tr]); train_s = time.perf_counter() - t0
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        f1s.append(f1_score(y[tt], pred))
        cm += confusion_matrix(y[tt], pred, labels=[0, 1]).ravel()      # tn, fp, fn, tp
        ys.append(y[tt]); ps.append(p)
        if timing:
            # (a) batched: whole test fold scored in one call, median of 5 repetitions after 2 warm-ups
            #     -> amortised model-scoring cost per record (NOT end-to-end latency)
            for _ in range(2):
                m.predict_proba(Xte[:100])
            reps = []
            for _ in range(5):
                t0 = time.perf_counter(); m.predict_proba(Xte); reps.append(time.perf_counter() - t0)
            # (b) single record scored alone (model call only), 200 calls after warm-up
            one = Xte[:1]; singles = []
            for _ in range(200):
                t0 = time.perf_counter(); m.predict_proba(one); singles.append(time.perf_counter() - t0)
            # (c) single raw record: imputation + scaling + model call
            raw = X.iloc[tt[:1]][cols]; e2e = []
            for _ in range(200):
                t0 = time.perf_counter(); m.predict_proba(sc.transform(imp.transform(raw)))
                e2e.append(time.perf_counter() - t0)
            res["amortised_us_per_record"] = float(np.median(reps)) / len(Xte) * 1e6
            res["single_record_ms"] = float(np.median(singles)) * 1e3
            res["single_record_ms_p95"] = float(np.percentile(singles, 95)) * 1e3
            res["single_raw_record_with_preprocessing_ms"] = float(np.median(e2e)) * 1e3
            res["timing_fold_records"] = int(len(Xte))
    tn, fp, fn, tp = cm
    res.update(F1=float(np.mean(f1s)), F1_SD=float(np.std(f1s, ddof=1)), TPR=tp / (tp + fn),
               FPR_pct=100 * fp / (fp + tn), Precision=tp / (tp + fp),
               ROC_AUC=float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))),
               TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn), train_s_last_fold=train_s)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}


out = {"All features": run([], timing=True),
       "Without IAT": run(["IAT"]),
       "Without IAT, Number, Weight": run(["IAT", "Number", "Weight"])}
out["environment"] = environment()
OUT = os.environ.get("OUT_DIR", ".")
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, "1_iat_ablation_results.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
