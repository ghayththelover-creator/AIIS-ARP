#!/usr/bin/env python3
"""
AIIS-ARP - revision experiments for the MSc thesis
"An Adaptive and Interpretable Intelligent System for Early Detection of
ARP Spoofing and Man-in-the-Middle Attacks" (Northern Border University).

Runs every additional experiment requested in the thesis review, on the
same CICIoT2023 subset and the same partition-level folds as the thesis:

  E0  Duplicate audit (exact duplicates inside the subset and across folds)
  E1  16-configuration benchmark with fold-level scores  -> Friedman + Nemenyi + CD diagram
  E2  Shuffle control for temporal windows (RQ2)
  E3  IAT shortcut analysis (distributions, position correlation, IAT-only model)
  E4  External hold-out on unseen partitions + hard negatives (other attack classes)
      + adaptation over a stream of distant partitions
  E5  Nested hyper-parameter tuning (XGBoost and LSTM)
  E6  Empirical error analysis (FP / FN profiles and SHAP)
  E7  Decision-threshold analysis (nested threshold selection)
  E8  LIME vs TreeSHAP: stability, run-to-run consistency and fidelity on the same alerts

USAGE (Kaggle, add the CICIoT2023 dataset as input; GPU optional):
    !python 4_revision_experiments.py
Environment variables (optional):
    DATA_GLOB   glob of the CSV partitions (default: /kaggle/input/**/part-*.csv)
    OUT_DIR     output folder (default: /kaggle/working/revision_out)
    QUICK=1     tiny settings for a smoke test
    SKIP=E5,E8  comma-separated list of experiments to skip
All results are written to OUT_DIR/revision_results.json, plus CSV tables and PNG figures.
Send revision_results.json and the PNG files back to fill the thesis.
"""
import os, glob, json, time, math, random, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, precision_score, recall_score,
                             matthews_corrcoef)
import xgboost as xgb

# --------------------------------------------------------------------------- config
DATA_GLOB = os.environ.get("DATA_GLOB", "/kaggle/input/**/part-*.csv")
OUT = os.environ.get("OUT_DIR", "/kaggle/working/revision_out")
QUICK = os.environ.get("QUICK", "0") == "1"
SKIP = set(s.strip().upper() for s in os.environ.get("SKIP", "").split(",") if s.strip())
os.makedirs(OUT, exist_ok=True)

N_STUDY_PARTS = 15            # first 15 partitions = thesis subset
N_HOLDOUT_PARTS = 20          # unseen partitions spread over 16..169
OTHER_CAP = 2000              # max rows per other-attack class per hold-out partition
MIN_PRED = 19                 # targets need >= 19 predecessors (T = 20)
SEED = 42
BENIGN, ARP = "BenignTraffic", "MITM-ArpSpoofing"
XGB_FIXED = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                 colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1,
                 eval_metric="logloss")
DEEP_SEEDS = [0] if QUICK else [0, 1, 2]
DEEP_EPOCHS = 2 if QUICK else 15
N_LIME = 6 if QUICK else 50
N_TUNE = 3 if QUICK else 20

np.random.seed(SEED); random.seed(SEED)
R = {"meta": {"quick": QUICK, "started": time.strftime("%Y-%m-%d %H:%M:%S")}}


def save():
    with open(os.path.join(OUT, "revision_results.json"), "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(f1=f1_score(y, pred, zero_division=0), tpr=tp / max(tp + fn, 1),
               fpr=fp / max(fp + tn, 1), precision=precision_score(y, pred, zero_division=0),
               mcc=matthews_corrcoef(y, pred), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, p)
        out["pr_auc"] = average_precision_score(y, p)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


# --------------------------------------------------------------------------- data
files = sorted(glob.glob(DATA_GLOB, recursive=True))
assert len(files) >= N_STUDY_PARTS, f"Found {len(files)} partition files with {DATA_GLOB}"
log(f"{len(files)} partition files found")


def read_part(path):
    df = pd.read_csv(path)
    lab = "label" if "label" in df.columns else "Label"
    df = df.rename(columns={lab: "label"})
    num = [c for c in df.columns if c != "label"]
    df[num] = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return df


parts = []
for i, f in enumerate(files[:N_STUDY_PARTS]):
    d = read_part(f)
    d = d[d["label"].isin([BENIGN, ARP])].copy()
    d["partition"] = i
    parts.append(d)
big = pd.concat(parts, ignore_index=True)
FEATS = [c for c in big.columns if c not in ("label", "partition")]
y_all = (big["label"] == ARP).astype(int).values
g_all = big["partition"].values
pos = big.groupby("partition").cumcount().values
target = pos >= MIN_PRED
X_all = big[FEATS].values.astype("float32")
R["meta"].update(n_records=int(len(big)), n_attack=int(y_all.sum()), n_features=len(FEATS),
                 n_targets=int(target.sum()), n_files_total=len(files))
log(f"study subset: {len(big)} records, {y_all.sum()} attacks, {target.sum()} targets, {len(FEATS)} features")
FOLDS = list(GroupKFold(5).split(X_all, y_all, g_all))
save()


def prep(tr_idx):
    imp = SimpleImputer(strategy="median").fit(X_all[tr_idx])
    sc = StandardScaler().fit(imp.transform(X_all[tr_idx]))
    return lambda A: sc.transform(imp.transform(A)).astype("float32")


def windows(Z, idx, T, order=None):
    """Flattened window of T rows ending at each target position (inside its partition).
    With `order` (shuffle control) the window follows the shuffled record order."""
    lag = (np.stack([idx - k for k in range(T - 1, -1, -1)], axis=1) if order is None
           else order[idx][:, -T:])
    return Z[lag].reshape(len(idx), -1)


def rec(idx, order=None):
    """Record index of the target at each position."""
    return idx if order is None else order[idx][:, -1]


def seq(Z, idx, T, order=None):
    lag = (np.stack([idx - k for k in range(T - 1, -1, -1)], axis=1) if order is None
           else order[idx][:, -T:])
    return Z[lag]  # (n, T, d)


def make_shuffled_order(seed=SEED):
    """For every row: indices of its 20-row window after shuffling the row order
    inside each partition (labels move with their records)."""
    rng = np.random.default_rng(seed)
    order = np.zeros((len(big), 20), dtype=np.int64)
    perm_rows = np.empty(len(big), dtype=np.int64)
    for p in np.unique(g_all):
        rows = np.where(g_all == p)[0]
        perm = rng.permutation(rows)
        perm_rows[rows] = perm          # position k of partition now holds record perm[k]
    # after shuffling, the record at position j has predecessors at positions j-19..j
    for k in range(20):
        shifted = np.arange(len(big)) - (19 - k)
        valid = (shifted >= 0)
        same = np.zeros(len(big), bool)
        same[valid] = g_all[shifted[valid]] == g_all[np.arange(len(big))[valid]]
        order[:, k] = np.where(same, perm_rows[np.clip(shifted, 0, None)], perm_rows)
    return perm_rows, order


# --------------------------------------------------------------------------- models
def fit_xgb(Xtr, ytr, **kw):
    params = {**XGB_FIXED, **kw}
    if QUICK:
        params["n_estimators"] = 50
    return xgb.XGBClassifier(**params).fit(Xtr, ytr)


def build_lstm(T, d, units=64, seed=0):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input((T, d))
    h = tf.keras.layers.LSTM(units)(inp)
    h = tf.keras.layers.Dropout(0.2)(h)
    emb = tf.keras.layers.Dense(32, activation="relu", name="emb")(h)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(emb)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


def fit_lstm(S, y, T, units=64, seed=0, epochs=None, patience=3, class_weight=None):
    import tensorflow as tf
    m = build_lstm(T, S.shape[2], units, seed)
    cb = [tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)]
    m.fit(S, y, batch_size=256, epochs=epochs or DEEP_EPOCHS, validation_split=0.1,
          callbacks=cb, verbose=0, class_weight=class_weight)
    embm = tf.keras.Model(m.input, m.get_layer("emb").output)
    return m, embm


# =========================================================================== E0
if "E0" not in SKIP:
    log("E0 duplicate audit")
    key = pd.util.hash_pandas_object(pd.DataFrame(X_all).round(6), index=False).values
    dup_rows = pd.Series(key).duplicated(keep=False).values
    lab_by_key = pd.DataFrame({"k": key, "y": y_all}).groupby("k")["y"].nunique()
    conflicting = int((lab_by_key > 1).sum())
    cross = []
    oof_dedup_true, oof_dedup_p = [], []
    for fi, (tr, te) in enumerate(FOLDS):
        trk = set(key[tr])
        tt = te[target[te]]
        has_twin = np.array([k in trk for k in key[tt]])
        cross.append(dict(fold=fi, test_targets=int(len(tt)), with_train_duplicate=int(has_twin.sum())))
        f = prep(tr)
        m = fit_xgb(f(X_all[tr]), y_all[tr])
        keep = tt[~has_twin]
        oof_dedup_true.append(y_all[keep]); oof_dedup_p.append(m.predict_proba(f(X_all[keep]))[:, 1])
    R["E0_duplicates"] = dict(
        rows_in_duplicate_groups=int(dup_rows.sum()), share=float(dup_rows.mean()),
        conflicting_feature_vectors=conflicting, per_fold=cross,
        test_targets_with_train_duplicate=int(sum(c["with_train_duplicate"] for c in cross)),
        xgb_T1_on_deduplicated_targets=metrics(np.concatenate(oof_dedup_true), np.concatenate(oof_dedup_p)))
    save()

# =========================================================================== E1 + E2
CONFIGS = [("LR", 1), ("DT", 1), ("RF", 1), ("RF", 5), ("RF", 10), ("RF", 20),
           ("XGB", 1), ("XGB", 5), ("XGB", 10), ("XGB", 20),
           ("LSTM", 10), ("LSTM", 20), ("Hybrid LSTM-RF", 10), ("Hybrid LSTM-RF", 20),
           ("Fusion LSTM-XGB", 10), ("Fusion LSTM-XGB", 20)]
oof = {}              # (name, T, ordering) -> probabilities over all targets (aligned with TGT)
TGT = np.where(target)[0]
tgt_pos = {r: i for i, r in enumerate(TGT)}
fold_rows = []
LSTM_CACHE = {}


def run_config(name, T, order=None, tag="original"):
    probs = np.zeros(len(TGT), dtype="float64")
    for fi, (tr, te) in enumerate(FOLDS):
        f = prep(tr)
        Z = np.zeros_like(X_all)
        Z[:] = f(X_all)                          # scaler fitted on training rows only
        trT = tr[target[tr]]; teT = te[target[te]]
        ytr, yte = y_all[rec(trT, order)], y_all[rec(teT, order)]
        t0 = time.perf_counter()
        if name in ("LR", "DT", "RF", "XGB"):
            Xtr = windows(Z, trT, T, order) if T > 1 else Z[rec(trT, order)]
            Xte = windows(Z, teT, T, order) if T > 1 else Z[rec(teT, order)]
            if name == "LR":
                m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            elif name == "DT":
                m = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
            elif name == "RF":
                m = RandomForestClassifier(50 if QUICK else 200, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
            else:
                m = fit_xgb(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
        else:
            Str, Ste = seq(Z, trT, T, order), seq(Z, teT, T, order)
            ps = []
            for s in DEEP_SEEDS:
                ck = (T, fi, s, tag)
                if ck not in LSTM_CACHE:
                    LSTM_CACHE[ck] = fit_lstm(Str, ytr, T, seed=s)
                lstm, emb = LSTM_CACHE[ck]
                if name == "LSTM":
                    ps.append(lstm.predict(Ste, batch_size=2048, verbose=0).ravel())
                else:
                    Etr = emb.predict(Str, batch_size=2048, verbose=0)
                    Ete = emb.predict(Ste, batch_size=2048, verbose=0)
                    if name.startswith("Hybrid"):
                        rf = RandomForestClassifier(50 if QUICK else 200, n_jobs=-1, random_state=SEED).fit(Etr, ytr)
                        ps.append(rf.predict_proba(Ete)[:, 1])
                    else:
                        cur_tr, cur_te = rec(trT, order), rec(teT, order)
                        fx = fit_xgb(np.hstack([Etr, Z[cur_tr]]), ytr)
                        ps.append(fx.predict_proba(np.hstack([Ete, Z[cur_te]]))[:, 1])
            p = np.mean(ps, axis=0)
        secs = time.perf_counter() - t0
        idx = [tgt_pos[r] for r in teT]
        probs[idx] = p
        mm = metrics(yte, p)
        fold_rows.append(dict(config=f"{name}, T = {T}", ordering=tag, fold=fi, seconds=secs, **mm))
        log(f"  {tag:9s} {name:16s} T={T:2d} fold {fi}: F1={mm['f1']:.4f}")
    oof[(name, T, tag)] = probs
    for k_ in [k_ for k_ in LSTM_CACHE if k_[0] == T and k_[3] == tag and name.startswith("Fusion")]:
        LSTM_CACHE.pop(k_)          # free memory once LSTM, hybrid and fusion have used it
    return probs


def y_targets(order=None):
    return y_all[order[TGT][:, -1]] if order is not None else y_all[TGT]


if "E1" not in SKIP:
    log("E1 full benchmark (fold-level)")
    for name, T in CONFIGS:
        run_config(name, T)
        pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "E1_fold_scores.csv"), index=False)
    fr = pd.DataFrame(fold_rows)
    fr = fr[fr.ordering == "original"]
    piv = fr.pivot(index="fold", columns="config", values="f1")
    summary = {c: metrics(y_targets(), oof[(n, T, "original")]) | {"f1_fold_mean": float(piv[c].mean()),
               "f1_fold_sd": float(piv[c].std(ddof=1))}
               for (n, T) in CONFIGS for c in [f"{n}, T = {T}"]}
    # Friedman + Nemenyi + critical difference
    chi, pf = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    k, N = piv.shape[1], piv.shape[0]
    q = stats.studentized_range.ppf(0.95, k, 1e6) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * N))
    best = ranks.index[0]
    R["E1_benchmark"] = dict(summary=summary, friedman_chi2=float(chi), friedman_p=float(pf),
                             mean_ranks=ranks.to_dict(), nemenyi_q_alpha=q, critical_difference=cd,
                             not_significantly_different_from_best=[c for c in ranks.index
                                                                    if ranks[c] - ranks[best] <= cd])
    # CD diagram (Demsar layout: best half labelled on the left, worst half on the right)
    rs = ranks.values; names = ranks.index.tolist(); half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(10, 0.32 * half + 2.2))
    ax.set_xlim(-2.5, k + 3.5); ax.set_ylim(-(half + 1) * 0.5, 1.6); ax.axis("off")
    ax.hlines(0, 1, k, color="k", lw=1)
    for i in range(1, k + 1):
        ax.vlines(i, -0.06, 0.06, color="k"); ax.text(i, 0.15, str(i), ha="center", fontsize=8)
    for i, (n, r) in enumerate(zip(names, rs)):
        if i < half:
            yy = -(i + 1) * 0.45; ax.plot([r, r, 0.6], [0, yy, yy], color="0.35", lw=0.8)
            ax.text(0.5, yy, f"{n} ({r:.1f})", ha="right", va="center", fontsize=7.5)
        else:
            yy = -(k - i) * 0.45; ax.plot([r, r, k + 0.4], [0, yy, yy], color="0.35", lw=0.8)
            ax.text(k + 0.5, yy, f"({r:.1f}) {n}", ha="left", va="center", fontsize=7.5)
    ax.plot([1, 1 + cd], [0.9, 0.9], color="#c0392b", lw=2.5)
    ax.text(1 + cd / 2, 1.05, f"CD = {cd:.2f}", ha="center", fontsize=8, color="#c0392b")
    # bars joining configurations not significantly different from the best
    ns = [r for r in rs if r - rs[0] <= cd]
    ax.plot([rs[0], max(ns)], [-0.15, -0.15], color="k", lw=3)
    ax.set_title("Critical-difference diagram (Nemenyi, alpha = 0.05; lower mean rank = better)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cd_diagram.png"), dpi=300); plt.close(fig)
    save()

if "E2" not in SKIP:
    log("E2 shuffle control")
    perm_rows, order = make_shuffled_order()
    for name, T in [("XGB", 5), ("XGB", 10), ("XGB", 20), ("RF", 10), ("LSTM", 10), ("Fusion LSTM-XGB", 10)]:
        if (name, T, "original") not in oof:
            run_config(name, T)
        run_config(name, T, order=order, tag="shuffled")
    res = {}
    for (n, T, tag), p in oof.items():
        if tag == "shuffled":
            res[f"{n}, T = {T}"] = dict(original=metrics(y_targets(), oof[(n, T, "original")])["f1"],
                                        shuffled=metrics(y_targets(order), p)["f1"])
    R["E2_shuffle_control"] = res
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "E1_E2_fold_scores.csv"), index=False)
    lab = list(res); a = [res[c]["original"] for c in lab]; b = [res[c]["shuffled"] for c in lab]
    fig, ax = plt.subplots(figsize=(8, 3.8)); xx = np.arange(len(lab))
    ax.bar(xx - 0.2, a, 0.4, label="Published order", color="#2c6fbb")
    ax.bar(xx + 0.2, b, 0.4, label="Shuffled within partition", color="#e3a33b")
    ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=20, fontsize=8); ax.set_ylabel("F1-score")
    ax.set_ylim(min(a + b) - 0.03, 1); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_shuffle_control.png"), dpi=300); plt.close(fig)
    save()

# =========================================================================== E3
if "E3" not in SKIP:
    log("E3 IAT analysis")
    iat = big["IAT"].values
    desc = {}
    for c, v in [("benign", iat[y_all == 0]), ("arp", iat[y_all == 1])]:
        q1, med, q3 = np.nanpercentile(v, [25, 50, 75])
        desc[c] = dict(q1=float(q1), median=float(med), q3=float(q3), n_unique=int(pd.Series(v).nunique()))
    ks = stats.ks_2samp(iat[y_all == 0], iat[y_all == 1])
    rho = []
    for p in np.unique(g_all):
        m = g_all == p
        rho.append(float(stats.spearmanr(iat[m], np.arange(m.sum()))[0]))
    per_part = big.groupby(["partition", "label"])["IAT"].median().unstack().to_dict()
    # IAT-only detector under the same folds
    pi, yi = [], []
    for tr, te in FOLDS:
        tt = te[target[te]]
        m = fit_xgb(np.nan_to_num(iat[tr]).reshape(-1, 1), y_all[tr])
        pi.append(m.predict_proba(np.nan_to_num(iat[tt]).reshape(-1, 1))[:, 1]); yi.append(y_all[tt])
    # partition predictability: can IAT identify the partition (capture segment)?
    from sklearn.model_selection import cross_val_score
    acc_part = cross_val_score(DecisionTreeClassifier(max_depth=8, random_state=SEED),
                               np.nan_to_num(iat).reshape(-1, 1), g_all, cv=3).mean()
    R["E3_iat"] = dict(describe=desc, ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                       spearman_iat_vs_position=dict(mean=float(np.mean(rho)), min=float(np.min(rho)),
                                                     max=float(np.max(rho))),
                       median_by_partition=per_part,
                       iat_only_detector=metrics(np.concatenate(yi), np.concatenate(pi)),
                       partition_from_iat_accuracy=float(acc_part),
                       partition_chance_accuracy=float(1 / len(np.unique(g_all))))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    lo, hi = np.nanpercentile(iat, [0.5, 99.5]); bins = np.linspace(lo, hi, 80)
    ax[0].hist(iat[y_all == 0], bins, alpha=.6, density=True, label="Benign", color="#2c6fbb")
    ax[0].hist(iat[y_all == 1], bins, alpha=.6, density=True, label="MITM-ArpSpoofing", color="#c0392b")
    ax[0].set_xlabel("IAT"); ax[0].set_ylabel("Density"); ax[0].legend(fontsize=8)
    med = big.groupby(["partition", "label"])["IAT"].median().unstack()
    for c, col in [(BENIGN, "#2c6fbb"), (ARP, "#c0392b")]:
        ax[1].plot(med.index + 1, med[c], "o-", color=col, label=c, ms=3)
    ax[1].set_xlabel("Partition"); ax[1].set_ylabel("Median IAT"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_iat_analysis.png"), dpi=300); plt.close(fig)
    save()

# =========================================================================== E4
if "E4" not in SKIP and len(files) > N_STUDY_PARTS:
    log("E4 external hold-out + hard negatives + distant-stream adaptation")
    rest = files[N_STUDY_PARTS:]
    pick = np.unique(np.linspace(0, len(rest) - 1, min(N_HOLDOUT_PARTS, len(rest))).astype(int))
    hold = []
    for j in pick:
        d = read_part(rest[j]); d["partition"] = N_STUDY_PARTS + j
        keep = d[d["label"].isin([BENIGN, ARP])]
        other = d[~d["label"].isin([BENIGN, ARP])]
        oth = pd.concat([s.sample(min(len(s), OTHER_CAP), random_state=SEED)
                         for _, s in other.groupby("label")] or [other])
        hold.append(pd.concat([keep, oth]))
    H = pd.concat(hold, ignore_index=True)
    XH = H[FEATS].values.astype("float32")
    ba = H["label"].isin([BENIGN, ARP]).values
    yH = (H["label"] == ARP).astype(int).values
    out = {}
    for tag, drop in [("all_features", []), ("without_IAT", ["IAT"])]:
        cols = [i for i, c in enumerate(FEATS) if c not in drop]
        imp = SimpleImputer(strategy="median").fit(X_all[:, cols])
        sc = StandardScaler().fit(imp.transform(X_all[:, cols]))
        tf_ = lambda A: sc.transform(imp.transform(A[:, cols]))
        m = fit_xgb(tf_(X_all), y_all)
        p = m.predict_proba(tf_(XH))[:, 1]
        per_class = (pd.DataFrame({"label": H["label"].values, "alert": p >= 0.5})
                     [~ba].groupby("label")["alert"].agg(["mean", "size"]))
        out[tag] = dict(benign_vs_arp=metrics(yH[ba], p[ba]),
                        other_attacks_alert_rate=float((p[~ba] >= 0.5).mean()) if (~ba).any() else None,
                        per_other_class=per_class.rename(columns={"mean": "alert_rate", "size": "n"}).to_dict("index"))
    # distant stream: static (15 parts) vs cumulative retraining over hold-out partitions
    hp = H["partition"].values
    st_f1, ad_f1 = [], []
    Xpool, ypool = X_all.copy(), y_all.copy()
    imp0 = SimpleImputer(strategy="median").fit(X_all); sc0 = StandardScaler().fit(imp0.transform(X_all))
    static = fit_xgb(sc0.transform(imp0.transform(X_all)), y_all)
    adaptive, impA, scA = static, imp0, sc0
    for P in sorted(np.unique(hp)):
        mk = (hp == P) & ba
        if mk.sum() == 0 or len(np.unique(yH[mk])) < 2:
            continue
        st_f1.append(f1_score(yH[mk], static.predict(sc0.transform(imp0.transform(XH[mk])))))
        ad_f1.append(f1_score(yH[mk], adaptive.predict(scA.transform(impA.transform(XH[mk])))))
        Xpool = np.vstack([Xpool, XH[mk]]); ypool = np.concatenate([ypool, yH[mk]])
        impA = SimpleImputer(strategy="median").fit(Xpool); scA = StandardScaler().fit(impA.transform(Xpool))
        adaptive = fit_xgb(scA.transform(impA.transform(Xpool)), ypool)
    w = stats.wilcoxon(ad_f1[1:], st_f1[1:]) if len(st_f1) > 2 and np.any(np.array(ad_f1[1:]) != np.array(st_f1[1:])) else None
    out["distant_stream"] = dict(partitions=[int(N_STUDY_PARTS + j + 1) for j in pick],
                                 static_f1=st_f1, adaptive_f1=ad_f1,
                                 static_mean=float(np.mean(st_f1)) if st_f1 else None,
                                 adaptive_mean=float(np.mean(ad_f1)) if ad_f1 else None,
                                 wilcoxon_p=float(w.pvalue) if w is not None else None)
    R["E4_external"] = out
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(range(1, len(st_f1) + 1), st_f1, "o-", label="Static (trained on partitions 1-15)", color="#7f8c8d")
    ax.plot(range(1, len(ad_f1) + 1), ad_f1, "s-", label="Adaptive (cumulative retraining)", color="#2c6fbb")
    ax.set_xlabel("Unseen partition in stream order"); ax.set_ylabel("F1-score"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_distant_stream.png"), dpi=300); plt.close(fig)
    save()

# =========================================================================== E5
if "E5" not in SKIP:
    log("E5 nested tuning")
    space = dict(n_estimators=[200, 400, 800], max_depth=[4, 6, 8, 10], learning_rate=[0.02, 0.05, 0.1, 0.2],
                 subsample=[0.7, 0.8, 0.9, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
                 min_child_weight=[1, 3, 5], reg_lambda=[0.5, 1, 2, 5])
    cands = list(ParameterSampler(space, N_TUNE, random_state=SEED))
    xp, xyy, chosen = [], [], []
    lp, lyy, lchosen = [], [], []
    for fi, (tr, te) in enumerate(FOLDS):
        inner = list(GroupKFold(3).split(tr, y_all[tr], g_all[tr]))
        best, bs = None, -1
        for c in cands:
            sc_ = []
            for itr, iva in inner:
                a, b = tr[itr], tr[iva]
                f = prep(a); m = fit_xgb(f(X_all[a]), y_all[a], **c)
                bt = b[target[b]]
                sc_.append(f1_score(y_all[bt], m.predict(f(X_all[bt]))))
            if np.mean(sc_) > bs:
                bs, best = np.mean(sc_), c
        chosen.append(best)
        f = prep(tr); m = fit_xgb(f(X_all[tr]), y_all[tr], **best)
        tt = te[target[te]]
        xp.append(m.predict_proba(f(X_all[tt]))[:, 1]); xyy.append(y_all[tt])
        # LSTM (T = 10): small grid, validation on one inner group split
        Z = f(X_all); trT = tr[target[tr]]; teT = te[target[te]]
        itr, iva = inner[0]
        a, b = tr[itr], tr[iva]; aT, bT = a[target[a]], b[target[b]]
        Za = prep(a)(X_all)
        bestl, bsl = None, -1
        for units in [64, 128]:
            for cw in [None, "balanced"]:
                cwd = None
                if cw:
                    w1 = len(y_all[aT]) / (2 * max(y_all[aT].sum(), 1)); w0 = len(y_all[aT]) / (2 * max((1 - y_all[aT]).sum(), 1))
                    cwd = {0: w0, 1: w1}
                mdl, _ = fit_lstm(seq(Za, aT, 10), y_all[aT], 10, units=units, epochs=3 if QUICK else 40,
                                  patience=5, class_weight=cwd)
                s = f1_score(y_all[bT], (mdl.predict(seq(Za, bT, 10), batch_size=2048, verbose=0).ravel() >= .5).astype(int))
                if s > bsl:
                    bsl, bestl = s, dict(units=units, class_weight=cw, cwd=cwd)
        lchosen.append({k: v for k, v in bestl.items() if k != "cwd"})
        mdl, _ = fit_lstm(seq(Z, trT, 10), y_all[trT], 10, units=bestl["units"], epochs=3 if QUICK else 40,
                          patience=5, class_weight=bestl["cwd"])
        lp.append(mdl.predict(seq(Z, teT, 10), batch_size=2048, verbose=0).ravel()); lyy.append(y_all[teT])
        log(f"  fold {fi}: XGB {best} | LSTM {lchosen[-1]}")
    R["E5_tuning"] = dict(xgb_tuned=metrics(np.concatenate(xyy), np.concatenate(xp)), xgb_chosen=chosen,
                          lstm_T10_tuned=metrics(np.concatenate(lyy), np.concatenate(lp)), lstm_chosen=lchosen,
                          n_candidates_xgb=len(cands))
    save()

# =========================================================================== E6 + E7 + E8
if not {"E6", "E7", "E8"} <= SKIP:
    log("E6/E7/E8 record-level detector: errors, thresholds, explainers")
    import shap
    p_oof = np.zeros(len(big)); shap_rows = []
    lime_rows, lime_glob, shap_glob, rr = [], [], [], []
    thr_nested = []
    for fi, (tr, te) in enumerate(FOLDS):
        f = prep(tr); Xtr = f(X_all[tr]); m = fit_xgb(Xtr, y_all[tr])
        tt = te[target[te]]; Xte = f(X_all[tt]); p = m.predict_proba(Xte)[:, 1]; p_oof[tt] = p
        yt = y_all[tt]; pred = (p >= .5).astype(int)
        grp = np.where((yt == 1) & (pred == 1), "TP", np.where((yt == 0) & (pred == 0), "TN",
                        np.where(pred == 1, "FP", "FN")))
        ex = shap.TreeExplainer(m)
        rng = np.random.default_rng(fi)
        for gname in ["TP", "TN", "FP", "FN"]:
            ids = np.where(grp == gname)[0]
            ids = rng.choice(ids, min(len(ids), 100 if QUICK else 500), replace=False) if len(ids) else ids
            if len(ids):
                sv = ex.shap_values(Xte[ids])
                for r_, i_ in enumerate(ids):
                    shap_rows.append(dict(group=gname, fold=fi, **{c: float(sv[r_, j]) for j, c in enumerate(FEATS)}))
        # E7 nested threshold: choose on inner OOF of the training partitions
        inner = list(GroupKFold(3).split(tr, y_all[tr], g_all[tr]))
        ip, iy = [], []
        for itr, iva in inner:
            a, b = tr[itr], tr[iva]; fa = prep(a); mi = fit_xgb(fa(X_all[a]), y_all[a])
            bt = b[target[b]]; ip.append(mi.predict_proba(fa(X_all[bt]))[:, 1]); iy.append(y_all[bt])
        ip, iy = np.concatenate(ip), np.concatenate(iy)
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        best_t = grid[np.argmax([f1_score(iy, (ip >= t).astype(int)) for t in grid])]
        thr_nested.append(dict(fold=fi, threshold=float(best_t), **metrics(yt, p, best_t)))
        # E8 LIME vs SHAP on the same alerts
        from lime.lime_tabular import LimeTabularExplainer
        al = np.where(pred == 1)[0]
        al = rng.choice(al, min(len(al), N_LIME), replace=False)
        lime_ex = LimeTabularExplainer(Xtr[:5000], feature_names=FEATS, class_names=["benign", "arp"],
                                       discretize_continuous=False, random_state=fi)
        sv = ex.shap_values(Xte[al])
        lw = np.zeros((len(al), len(FEATS)))
        for r_, i_ in enumerate(al):
            e = lime_ex.explain_instance(Xte[i_], m.predict_proba, num_features=len(FEATS), num_samples=2000)
            for j, w in e.as_map()[1]:
                lw[r_, j] = w
        lime_glob.append(np.abs(lw).mean(0)); shap_glob.append(np.abs(sv).mean(0))
        # deletion fidelity on identical alerts (replace with 0 = training mean)
        for k in [1, 3, 5, 10]:
            for meth, W in [("SHAP", sv), ("LIME", lw)]:
                Xd = Xte[al].copy()
                for r_ in range(len(al)):
                    Xd[r_, np.argsort(-np.abs(W[r_]))[:k]] = 0
                lime_rows.append(dict(fold=fi, k=k, method=meth,
                                      drop=float((p[al] - m.predict_proba(Xd)[:, 1]).mean())))
        # run-to-run consistency of LIME (two seeds, same alerts)
        if fi == 0:
            ex2 = LimeTabularExplainer(Xtr[:5000], feature_names=FEATS, class_names=["benign", "arp"],
                                       discretize_continuous=False, random_state=999)
            for r_, i_ in enumerate(al[:min(30, len(al))]):
                e2 = ex2.explain_instance(Xte[i_], m.predict_proba, num_features=len(FEATS), num_samples=2000)
                w2 = np.zeros(len(FEATS))
                for j, w in e2.as_map()[1]:
                    w2[j] = w
                A = set(np.argsort(-np.abs(lw[r_]))[:5]); B = set(np.argsort(-np.abs(w2))[:5])
                rr.append(len(A & B) / len(A | B))
    yt_all = y_all[TGT]; pt = p_oof[TGT]
    sh = pd.DataFrame(shap_rows)
    # E6
    raw = big.loc[TGT, FEATS].copy(); pred = (pt >= .5).astype(int)
    raw["group"] = np.where((yt_all == 1) & (pred == 1), "TP", np.where((yt_all == 0) & (pred == 0), "TN",
                            np.where(pred == 1, "FP", "FN")))
    top = sh.drop(columns=["group", "fold"]).abs().mean().sort_values(ascending=False).index[:8].tolist()
    R["E6_errors"] = dict(group_sizes=raw["group"].value_counts().to_dict(),
                          median_raw_values=raw.groupby("group")[top].median().to_dict("index"),
                          mean_shap_by_group=sh.groupby("group")[top].mean().to_dict("index"),
                          top_features=top)
    raw.groupby("group")[top].median().to_csv(os.path.join(OUT, "E6_median_by_group.csv"))
    sh.groupby("group")[top].mean().to_csv(os.path.join(OUT, "E6_mean_shap_by_group.csv"))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    msh = sh.groupby("group")[top].mean().reindex(["TP", "FN", "FP", "TN"])
    msh.T.plot.bar(ax=ax, color=["#c0392b", "#e3a33b", "#8e44ad", "#2c6fbb"], width=.8)
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("Mean SHAP (log-odds)"); ax.legend(fontsize=8, ncol=4)
    ax.tick_params(axis="x", rotation=25, labelsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_error_shap.png"), dpi=300); plt.close(fig)
    # E7
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tab = [dict(threshold=float(t), **metrics(yt_all, pt, t)) for t in grid]
    pd.DataFrame(tab).to_csv(os.path.join(OUT, "E7_threshold_table.csv"), index=False)
    tn = pd.DataFrame(thr_nested)
    R["E7_threshold"] = dict(table=tab, nested=thr_nested,
                             nested_pooled_tp=int(tn.tp.sum()), nested_pooled_fp=int(tn.fp.sum()),
                             nested_pooled_fn=int(tn.fn.sum()), nested_pooled_tn=int(tn.tn.sum()))
    fig, ax = plt.subplots(figsize=(6.5, 3.4)); T_ = pd.DataFrame(tab)
    ax.plot(T_.threshold, T_.tpr, label="TPR", color="#c0392b"); ax.plot(T_.threshold, T_.f1, label="F1", color="#2c6fbb")
    ax2 = ax.twinx(); ax2.plot(T_.threshold, T_.fpr * 100, "--", label="FPR (%)", color="#7f8c8d"); ax2.set_ylabel("FPR (%)")
    ax.axvline(.5, color="k", lw=.5, ls=":"); ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_threshold.png"), dpi=300); plt.close(fig)
    # E8
    def pair_rho(G):
        return [stats.spearmanr(G[i], G[j])[0] for i in range(len(G)) for j in range(i + 1, len(G))]
    lr_, sr_ = pair_rho(lime_glob), pair_rho(shap_glob)
    fid = pd.DataFrame(lime_rows).groupby(["method", "k"])["drop"].mean().unstack(0)
    R["E8_lime_vs_shap"] = dict(alerts_per_fold=N_LIME,
                                cross_fold_spearman=dict(SHAP=dict(mean=float(np.mean(sr_)), min=float(np.min(sr_))),
                                                         LIME=dict(mean=float(np.mean(lr_)), min=float(np.min(lr_)))),
                                lime_run_to_run_jaccard_top5=dict(mean=float(np.mean(rr)), min=float(np.min(rr)), n=len(rr)),
                                deletion_fidelity=fid.to_dict())
    save()

R["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done ->", OUT)
