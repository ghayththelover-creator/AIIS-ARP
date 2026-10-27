"""Round-3 re-run of the 16-configuration benchmark (E1) on the thesis subset, with the hybrid and fusion
second stages fitted BOTH on in-sample and on inner out-of-fold LSTM embeddings from the SAME LSTM fits.
Reuses the data loading, folds, pre-processing and model code of 4_revision_experiments.py (v2.0)."""
import os, sys, time, json, pickle
import numpy as np, pandas as pd

SRC = "/home/claude/aiis-arp/4_revision_experiments.py"
OUTD = os.environ.get("OUT_DIR", "/home/claude/runs/oof_out")
os.makedirs(OUTD, exist_ok=True)
lines = open(SRC).read().split("\n")
setup = "\n".join(lines[:228])           # imports, config, data, folds, helpers, models
block = "\n".join(lines[253:265])        # CONFIGS, TGT, tgt_pos ...
g = {"__name__": "aiis_rev"}
exec(compile(setup + "\n" + block, SRC, "exec"), g)
globals().update({k: v for k, v in g.items() if not k.startswith("__")})

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

T0 = time.time()
STATE = os.path.join(OUTD, "state.pkl")
st = pickle.load(open(STATE, "rb")) if os.path.exists(STATE) else {"oof": {}, "rows": [], "done": []}


def checkpoint():
    pickle.dump(st, open(STATE, "wb"))
    pd.DataFrame(st["rows"]).to_csv(os.path.join(OUTD, "fold_scores.csv"), index=False)


def put(name, fi, teT, p, secs):
    key = name
    if key not in st["oof"]:
        st["oof"][key] = np.full(len(TGT), np.nan)
    st["oof"][key][[tgt_pos[r] for r in teT]] = p
    st["rows"].append(dict(config=name, fold=fi, seconds=secs, **metrics(y_all[teT], p)))
    log(f"  {name:34s} fold {fi}: F1={st['rows'][-1]['f1']:.4f}  ({secs:.0f} s)")


def run_fast(name, T):
    cname = f"{name}, T = {T}"
    if cname in st["done"]:
        return
    for fi, (tr, te) in enumerate(FOLDS):
        if f"{cname} fold {fi}" in st["done"]:
            continue
        if time.time() - T0 > float(os.environ.get("BUDGET_S", "1e9")):
            checkpoint(); log("time budget used; stopping for resume"); sys.exit(0)
        f = prep(tr); Z = f(X_all)
        trT, teT = tr[target[tr]], te[target[te]]
        Xtr = windows(Z, trT, T) if T > 1 else Z[trT]
        Xte = windows(Z, teT, T) if T > 1 else Z[teT]
        t0 = time.perf_counter()
        if name == "LR":
            m = LogisticRegression(max_iter=2000).fit(Xtr, y_all[trT])
        elif name == "DT":
            m = DecisionTreeClassifier(random_state=SEED).fit(Xtr, y_all[trT])
        elif name == "RF":
            m = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(Xtr, y_all[trT])
        else:
            m = fit_xgb(Xtr, y_all[trT])
        put(cname, fi, teT, m.predict_proba(Xte)[:, 1], time.perf_counter() - t0)
        st["done"].append(f"{cname} fold {fi}"); checkpoint()
    st["done"].append(cname); checkpoint()


def run_deep(T):
    tag = f"deep T={T}"
    if tag in st["done"]:
        return
    names = [f"LSTM, T = {T}", f"Hybrid LSTM-RF (in-sample), T = {T}", f"Hybrid LSTM-RF (OOF), T = {T}",
             f"Fusion LSTM-XGB (in-sample), T = {T}", f"Fusion LSTM-XGB (OOF), T = {T}"]
    for fi, (tr, te) in enumerate(FOLDS):
        if f"{tag} fold {fi}" in st["done"]:
            continue
        f = prep(tr); Z = f(X_all)
        trT, teT = tr[target[tr]], te[target[te]]
        ytr = y_all[trT]
        Str, Ste = seq(Z, trT, T), seq(Z, teT, T)
        part = st.setdefault("partial", {}).setdefault(f"{T}-{fi}", {"seeds": [], "P": {n: [] for n in names},
                                                                   "secs": {n: 0.0 for n in names}})
        P, secs = part["P"], part["secs"]
        for s in DEEP_SEEDS:
            if s in part["seeds"]:
                continue
            if int(os.environ.get("MAX_SEEDS", "99")) <= 0:
                checkpoint(); log("seed budget used; stopping for resume"); sys.exit(0)
            os.environ["MAX_SEEDS"] = str(int(os.environ.get("MAX_SEEDS", "99")) - 1)
            t0 = time.perf_counter()
            lstm, emb = fit_lstm(Str, ytr, T, seed=s)
            t_lstm = time.perf_counter() - t0
            P[names[0]].append(lstm.predict(Ste, batch_size=2048, verbose=0).ravel()); secs[names[0]] += t_lstm
            Ete = emb.predict(Ste, batch_size=2048, verbose=0)
            Ein = emb.predict(Str, batch_size=2048, verbose=0)
            t0 = time.perf_counter()
            Eoof = oof_embeddings(Str, ytr, g_all[trT], T, s)
            t_oof = time.perf_counter() - t0
            for kind, Etr in (("in-sample", Ein), ("OOF", Eoof)):
                extra = t_lstm + (t_oof if kind == "OOF" else 0)
                t0 = time.perf_counter()
                rf = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(Etr, ytr)
                P[f"Hybrid LSTM-RF ({kind}), T = {T}"].append(rf.predict_proba(Ete)[:, 1])
                secs[f"Hybrid LSTM-RF ({kind}), T = {T}"] += extra + time.perf_counter() - t0
                t0 = time.perf_counter()
                fx = fit_xgb(np.hstack([Etr, Z[trT]]), ytr)
                P[f"Fusion LSTM-XGB ({kind}), T = {T}"].append(fx.predict_proba(np.hstack([Ete, Z[teT]]))[:, 1])
                secs[f"Fusion LSTM-XGB ({kind}), T = {T}"] += extra + time.perf_counter() - t0
            log(f"  T={T} fold {fi} seed {s} done (LSTM {t_lstm:.0f} s, OOF embeddings {t_oof:.0f} s)")
            part["seeds"].append(s); checkpoint()
        for n in names:
            put(n, fi, teT, np.mean(P[n], axis=0), secs[n])
        st["done"].append(f"{tag} fold {fi}"); checkpoint()
    st["done"].append(tag); checkpoint()


import platform, sklearn, xgboost, tensorflow as tf
env = dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
           tensorflow=tf.__version__, numpy=np.__version__, pandas=pd.__version__, cpu_count=os.cpu_count(),
           started=time.strftime("%Y-%m-%d %H:%M:%S"))
json.dump(env, open(os.path.join(OUTD, "environment.json"), "w"), indent=1)
PLAN = os.environ.get("PLAN", "fast,deep10,deep20,rfwin").split(",")
if "fast" in PLAN:
    for n, T in [("LR", 1), ("DT", 1), ("RF", 1), ("XGB", 1), ("XGB", 5), ("XGB", 10), ("XGB", 20)]:
        run_fast(n, T)
if "deep10" in PLAN:
    run_deep(10)
if "deep20" in PLAN:
    run_deep(20)
if "rfwin" in PLAN:
    for T in (5, 10, 20):
        run_fast("RF", T)
log("finished")
