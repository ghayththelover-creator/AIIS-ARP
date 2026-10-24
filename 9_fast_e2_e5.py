#!/usr/bin/env python3
"""
AIIS-ARP - fast versions of E2 (shuffle control) and E5 (nested tuning) for XGBoost only.
Same 15 partitions, partition-level GroupKFold(5), train-only preprocessing, 106,382 targets.
Runs on GPU if available (XGBoost device='cuda'), otherwise CPU.  Typical run: 20-30 min on GPU.
Output: /kaggle/working/fast_out/fast_results.json (+ printed summary)
"""
import os, glob, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, precision_score, roc_auc_score
import xgboost as xgb

GLOB = os.environ.get("DATA_GLOB", os.environ.get("CICIOT_GLOB", "/kaggle/input/**/part-*.csv"))
OUT = os.environ.get("OUT_DIR", "/kaggle/working/fast_out")
N_TUNE = int(os.environ.get("N_TUNE", 10))
os.makedirs(OUT, exist_ok=True)
SEED, MIN_PRED = 42, 19
BENIGN, ARP = "BenignTraffic", "MITM-ArpSpoofing"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# GPU detection
import shutil, subprocess, warnings
DEV = "cpu"
if shutil.which("nvidia-smi") and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
    DEV = "cuda"
warnings.filterwarnings("ignore")
log("XGBoost device:", DEV)
BASE = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
            tree_method="hist", random_state=SEED, eval_metric="logloss", device=DEV)


def fit(X, y, **kw):
    return xgb.XGBClassifier(**{**BASE, **kw}).fit(X, y)


def metrics(y, p, t=0.5):
    pr = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return dict(f1=float(f1_score(y, pr)), tpr=float(tp / (tp + fn)), fpr=float(fp / (fp + tn)),
                precision=float(precision_score(y, pr)), roc_auc=float(roc_auc_score(y, p)))


files = sorted(glob.glob(GLOB, recursive=True))[:15]
parts = []
for i, f in enumerate(files):
    d = pd.read_csv(f)
    d = d[d["label"].isin([BENIGN, ARP])].copy()
    d["partition"] = i
    parts.append(d)
big = pd.concat(parts, ignore_index=True)
FEATS = [c for c in big.columns if c not in ("label", "partition")]
X = big[FEATS].astype("float32").replace([np.inf, -np.inf], np.nan).values
y = (big["label"] == ARP).astype(int).values
g = big["partition"].values
pos = big.groupby("partition").cumcount().values
target = pos >= MIN_PRED
FOLDS = list(GroupKFold(5).split(X, y, g))
log(f"{len(big)} records, {target.sum()} targets, {len(FEATS)} features")
R = {"device": DEV}


def prep(idx):
    imp = SimpleImputer(strategy="median").fit(X[idx])
    sc = StandardScaler().fit(imp.transform(X[idx]))
    return lambda A: sc.transform(imp.transform(A)).astype("float32")


# ---------------------------------------------------------------- E2 shuffle control (XGB)
log("E2 shuffle control (XGBoost, T = 5, 10, 20)")
rng = np.random.default_rng(SEED)
seq = np.empty(len(big), dtype=np.int64)          # seq[k] = record placed at row-position k (within partition)
for p in np.unique(g):
    rows = np.where(g == p)[0]
    seq[rows] = rng.permutation(rows)


def win(Z, tidx, T, shuffled):
    lag = np.stack([tidx - k for k in range(T - 1, -1, -1)], axis=1)   # positions inside the partition
    rec = seq[lag] if shuffled else lag
    return Z[rec].reshape(len(tidx), -1), (seq[tidx] if shuffled else tidx)


e2 = {}
for T in [5, 10, 20]:
    for tag, sh in [("published", False), ("shuffled", True)]:
        ys, ps = [], []
        for tr, te in FOLDS:
            Z = prep(tr)(X)
            trT, teT = tr[target[tr]], te[target[te]]
            Xtr, rtr = win(Z, trT, T, sh)
            Xte, rte = win(Z, teT, T, sh)
            m = fit(Xtr, y[rtr])
            ps.append(m.predict_proba(Xte)[:, 1]); ys.append(y[rte])
        e2[f"XGB, T = {T}|{tag}"] = metrics(np.concatenate(ys), np.concatenate(ps))
        log(f"  XGB T={T:2d} {tag:9s} F1={e2[f'XGB, T = {T}|{tag}']['f1']:.4f}")
R["E2_shuffle_xgb"] = e2
json.dump(R, open(os.path.join(OUT, "fast_results.json"), "w"), indent=2)

# ---------------------------------------------------------------- E5 nested tuning (XGB)
log(f"E5 nested tuning (XGBoost, {N_TUNE} candidates)")
space = dict(n_estimators=[200, 400, 800], max_depth=[4, 6, 8, 10], learning_rate=[0.02, 0.05, 0.1, 0.2],
             subsample=[0.7, 0.8, 0.9, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
             min_child_weight=[1, 3, 5], reg_lambda=[0.5, 1, 2, 5])
cands = list(ParameterSampler(space, N_TUNE, random_state=SEED))
yy, pp, chosen = [], [], []
for fi, (tr, te) in enumerate(FOLDS):
    inner = list(GroupKFold(3).split(tr, y[tr], g[tr]))
    best, bs = None, -1
    for c in cands:
        s = []
        for itr, iva in inner:
            a, b = tr[itr], tr[iva]
            f = prep(a); m = fit(f(X[a]), y[a], **c)
            bt = b[target[b]]
            s.append(f1_score(y[bt], m.predict(f(X[bt]))))
        if np.mean(s) > bs:
            bs, best = float(np.mean(s)), c
    f = prep(tr); m = fit(f(X[tr]), y[tr], **best)
    tt = te[target[te]]
    pp.append(m.predict_proba(f(X[tt]))[:, 1]); yy.append(y[tt])
    chosen.append(best)
    log(f"  fold {fi}: inner F1={bs:.4f} best={best}")
R["E5_xgb_tuning"] = dict(n_candidates=N_TUNE, tuned=metrics(np.concatenate(yy), np.concatenate(pp)), chosen=chosen)
json.dump(R, open(os.path.join(OUT, "fast_results.json"), "w"), indent=2, default=str)
log("tuned XGB:", R["E5_xgb_tuning"]["tuned"])
log("done ->", OUT)
print(json.dumps(R, indent=1, default=str))
