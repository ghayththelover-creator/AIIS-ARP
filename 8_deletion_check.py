#!/usr/bin/env python3
"""8_deletion_check.py - deletion test on 200 alerts in each of the five folds (Table 5.15).
Replacement value = training mean (0 after z-scoring); control = k random features.
NOTE: the deletion test measures the model's sensitivity to removing the features ranked highest
by TreeSHAP; replacement can create out-of-distribution inputs, so it is an indirect check, not
proof of causal or ground-truth correctness."""
import glob, os, numpy as np, pandas as pd, shap
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

GLOB = os.environ.get("DATA_GLOB", os.environ.get("CICIOT_GLOB", "/kaggle/input/**/part-*.csv"))
files = sorted(glob.glob(GLOB, recursive=True))[:15]
parts = []
for k, f in enumerate(files):
    d = pd.read_csv(f).replace([np.inf, -np.inf], np.nan)
    d = d[d["label"].isin(["BenignTraffic", "MITM-ArpSpoofing"])].copy()
    d["partition"], d["pos"] = k, np.arange(len(d))
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
df = df[df["pos"] >= 19].reset_index(drop=True)
FEATS = [c for c in df.columns if c not in ("label", "partition", "pos")]
y = (df["label"] == "MITM-ArpSpoofing").astype(int).values
rows = []
for fi, (tr, te) in enumerate(GroupKFold(5).split(df, y, groups=df["partition"])):
    imp = SimpleImputer(strategy="median").fit(df.loc[tr, FEATS])
    sc = StandardScaler().fit(imp.transform(df.loc[tr, FEATS]))
    Xtr = sc.transform(imp.transform(df.loc[tr, FEATS])); Xte = sc.transform(imp.transform(df.loc[te, FEATS]))
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                      tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=-1).fit(Xtr, y[tr])
    p = m.predict_proba(Xte)[:, 1]
    rng = np.random.default_rng(fi)
    al = rng.choice(np.where(p >= 0.5)[0], 200, replace=False)          # 200 alerts per fold
    sv = shap.TreeExplainer(m).shap_values(Xte[al])
    for k in [1, 3, 5, 10]:
        top = np.argsort(-np.abs(sv), 1)[:, :k]
        rnd = np.array([rng.choice(len(FEATS), k, replace=False) for _ in al])
        for name, cols in [("SHAP top-k", top), ("random k", rnd)]:
            Xd = Xte[al].copy()
            for i in range(len(al)):
                Xd[i, cols[i]] = 0.0                                     # 0 = training mean
            rows.append(dict(fold=fi + 1, k=k, features=name,
                             drop=float((p[al] - m.predict_proba(Xd)[:, 1]).mean())))
    print(f"fold {fi + 1} done", flush=True)
res = pd.DataFrame(rows)
out = res.groupby(["k", "features"])["drop"].agg(mean="mean", sd="std", min="min", max="max").reset_index()
OUT = os.environ.get("OUT_DIR", "/kaggle/working/supp")
os.makedirs(OUT, exist_ok=True)
res.to_csv(os.path.join(OUT, "S7b_deletion_all_folds.csv"), index=False)
print("\nDeletion test, 200 alerts per fold, replacement = training mean (0), 5 folds")
print(out.to_markdown(index=False, floatfmt=".3f"))
print("\nPer fold, k = 5:")
print(res[res.k == 5].pivot(index="fold", columns="features", values="drop").to_markdown(floatfmt=".3f"))
