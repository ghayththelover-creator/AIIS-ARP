#!/usr/bin/env python3
"""Dominant-feature ablation (Table 5.5 / Table C.1) and inference timing (Section 5.3.2).
Input: arp_15parts.csv.gz   Output: 1_iat_ablation_results.json
Usage: python 1_iat_ablation_and_timing.py [arp_15parts.csv.gz]"""
import sys, time, json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score

big = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "arp_15parts.csv.gz")
y = (big["label"] == "MITM-ArpSpoofing").astype(int).values
g = big["partition"].values
X = big.drop(columns=["label", "partition"])
target = big.groupby("partition").cumcount().values >= 19      # 106,382 common target records


def model():
    return xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.8, tree_method="hist", random_state=42)


def run(drop):
    cols = [c for c in X.columns if c not in drop]
    f1s, yt, pt, train_s = [], [], [], 0.0
    for tr, te in GroupKFold(5).split(X, y, g):                  # partition-level folds
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr][cols])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr][cols]))
        Xtr = sc.transform(imp.transform(X.iloc[tr][cols]))
        tt = te[target[te]]
        Xte = sc.transform(imp.transform(X.iloc[tt][cols]))
        m = model(); t0 = time.perf_counter(); m.fit(Xtr, y[tr]); train_s = time.perf_counter() - t0
        p = m.predict_proba(Xte)[:, 1]
        f1s.append(f1_score(y[tt], (p >= .5).astype(int))); yt.append(y[tt]); pt.append(p)
    yt, pt = np.concatenate(yt), np.concatenate(pt); pred = (pt >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    return dict(f1=float(np.mean(f1s)), f1_sd=float(np.std(f1s, ddof=1)), tpr=tp / (tp + fn), fpr_pct=100 * fp / (fp + tn),
                precision=precision_score(yt, pred), roc_auc=roc_auc_score(yt, pt), tp=int(tp), fn=int(fn), fp=int(fp),
                tn=int(tn), train_s_last_fold=train_s), (m, Xte)


res = {}
for name, drop in [("All features", []), ("Without IAT", ["IAT"]), ("Without IAT, Number, Weight", ["IAT", "Number", "Weight"])]:
    res[name], last = run(drop)
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()})
# timing on the last fold of the full model
m, Xte = run([])[1]
m.predict_proba(Xte[:2]); m.predict_proba(Xte[:2])                 # warm-up
batch = []
for _ in range(5):
    t0 = time.perf_counter(); m.predict_proba(Xte); batch.append(time.perf_counter() - t0)
single = []
for i in range(200):
    t0 = time.perf_counter(); m.predict_proba(Xte[i:i + 1]); single.append(time.perf_counter() - t0)
res["timing"] = dict(n_records=int(len(Xte)), us_per_record_batch=1e6 * float(np.median(batch)) / len(Xte),
                     ms_single_record=1e3 * float(np.median(single)))
print(res["timing"])
json.dump(res, open("1_iat_ablation_results.json", "w"), indent=2, default=float)
