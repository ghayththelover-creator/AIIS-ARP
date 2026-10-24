#!/usr/bin/env python3
"""
10_round2_analyses.py - analyses added in the second review round (paper M09, M14, M15, M18,
M21-M23, M26; thesis Ch3-5 and Appendix B comments).

  R1  Provenance: exact file names, SHA-256 hashes and full per-class counts of the study
      partitions; fold -> partition/file mapping; split indices; software/hardware description.
  R2  Sampling sensitivity: the record-level detector (XGB, T = 1) re-evaluated with the same
      protocol on N_ALT alternative sets of 15 partitions drawn from partitions 16-169.
  R3  Partition-clustered paired bootstrap for all configurations in oof_predictions.csv
      (written by 4_revision_experiments.py, E1), with selection-aware summaries
      (how often each configuration is the best one in a resample), simultaneous (max-type)
      intervals and an equivalence (TOST-style) check for the XGB-versus-fusion comparison.
      Without that file it falls back to the four record-level models.
  R4  Re-assessment of the PREFERRED ARP-vs-rest detector: per-alert TreeSHAP latency, global
      attribution and cross-fold rank stability, deletion test (several replacement values),
      IAT ablation, and static / cumulative / size-matched / recent-window retraining.
  R5  Prevalence analysis: precision, F1 and false alerts per 10,000 flows of the benign-vs-ARP
      and ARP-vs-rest detectors at assumed deployment class mixes (from out-of-fold rates).
  R6  Timing protocol: TreeSHAP and LIME on the SAME alerts (same count), repeated runs,
      single-record end-to-end latency (pre-processing + scoring + explanation).

Usage (Kaggle, CICIoT2023 attached):   python 10_round2_analyses.py
Environment: DATA_GLOB, OUT_DIR (default /kaggle/working/round2), SKIP=R2,R4 ..., N_ALT (5),
OOF_FILE (path to oof_predictions.csv), OTHER_PER_CLASS (1000), QUICK=1 (smoke test).
Outputs: OUT_DIR/round2_results.json, CSV tables, partition_manifest.csv, split_indices.json.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiis_common import (ARP, BENIGN, N_STUDY_PARTS, SEED, XGB_FIXED, environment, list_files,
                         load_study, out_dir, read_part, sha256)

warnings.filterwarnings("ignore")
QUICK = os.environ.get("QUICK", "0") == "1"
SKIP = {s.strip().upper() for s in os.environ.get("SKIP", "").split(",") if s.strip()}
OUT = out_dir("/kaggle/working/round2")
N_ALT = int(os.environ.get("N_ALT", 2 if QUICK else 5))
N_BOOT = 200 if QUICK else 2000
OTHER_PER_CLASS = int(os.environ.get("OTHER_PER_CLASS", 1000))
N_ALERTS = 20 if QUICK else 200
N_LIME = 5 if QUICK else 30
rng = np.random.default_rng(SEED)
R = {"meta": dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), seed=SEED, quick=QUICK,
                  environment=environment())}


def save():
    json.dump(R, open(os.path.join(OUT, "round2_results.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def xgb(**kw):
    p = {**XGB_FIXED, **kw}
    if QUICK:
        p["n_estimators"] = 50
    return XGBClassifier(**p)


def rates(y, p, thr=0.5):
    h = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, h, labels=[0, 1]).ravel()
    return dict(f1=float(f1_score(y, h, zero_division=0)), tpr=tp / max(tp + fn, 1),
                fpr=fp / max(fp + tn, 1), precision=tp / max(tp + fp, 1),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def prep(train_df, feats):
    imp = SimpleImputer(strategy="median").fit(train_df[feats])
    sc = StandardScaler().fit(imp.transform(train_df[feats]))
    return lambda d: sc.transform(imp.transform(d[feats])), imp, sc


files = list_files()
study_files = files[:N_STUDY_PARTS]
log(f"{len(files)} partition files; study = first {N_STUDY_PARTS} in sorted order")

# =============================================================================== R1
if "R1" not in SKIP:
    log("R1 provenance")
    rows = []
    for k, f in enumerate(study_files):
        d = read_part(f)
        cnt = d["label"].value_counts()
        rows.append(dict(partition=k + 1, file=os.path.basename(f), sha256=sha256(f), rows=len(d),
                         benign=int(cnt.get(BENIGN, 0)), arp=int(cnt.get(ARP, 0)),
                         other_attacks=int(len(d) - cnt.get(BENIGN, 0) - cnt.get(ARP, 0)),
                         **{f"n[{c}]": int(v) for c, v in cnt.items()}))
    man = pd.DataFrame(rows).fillna(0)
    man.to_csv(os.path.join(OUT, "partition_manifest.csv"), index=False)
    big, feats = load_study(study_files)
    tg = big[big["target"]].reset_index(drop=True)
    folds = list(GroupKFold(5).split(tg, tg["y"], tg["partition"]))
    split = {f"fold_{i + 1}": dict(test_partitions=sorted(int(p) + 1 for p in tg.loc[te, "partition"].unique()),
                                   test_files=sorted(tg.loc[te, "pfile"].unique().tolist()),
                                   n_test_targets=int(len(te)),
                                   test_target_ids=tg.loc[te, "target_id"].tolist())
             for i, (_, te) in enumerate(folds)}
    json.dump(split, open(os.path.join(OUT, "split_indices.json"), "w"))
    R["R1_provenance"] = dict(files=man[["partition", "file", "sha256", "rows", "benign", "arp", "other_attacks"]]
                              .to_dict("records"),
                              folds={k: {kk: vv for kk, vv in v.items() if kk != "test_target_ids"}
                                     for k, v in split.items()},
                              n_records=int(len(big)), n_targets=int(len(tg)), features=feats,
                              selection_rule="first 15 files in sorted path order (part-00000 ... part-00014)")
    save()

# =============================================================================== R2
if "R2" not in SKIP and len(files) >= 2 * N_STUDY_PARTS:
    log("R2 sampling sensitivity")
    pool = np.arange(N_STUDY_PARTS, len(files))
    out = []
    for a in range(N_ALT):
        pick = np.sort(np.random.default_rng(1000 + a).choice(pool, N_STUDY_PARTS, replace=False))
        big, feats = load_study([files[i] for i in pick])
        tg = big[big["target"]].reset_index(drop=True)
        ys, ps = [], []
        for tr, te in GroupKFold(5).split(tg, tg["y"], tg["partition"]):
            T, _, _ = prep(tg.loc[tr], feats)
            m = xgb().fit(T(tg.loc[tr]), tg.loc[tr, "y"])
            ps.append(m.predict_proba(T(tg.loc[te]))[:, 1]); ys.append(tg.loc[te, "y"].values)
        r = rates(np.concatenate(ys), np.concatenate(ps))
        r.update(alt_set=a + 1, partitions=(pick + 1).tolist(), n_targets=int(len(tg)),
                 attack_share=float(tg["y"].mean()), roc_auc=float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))))
        out.append(r)
        log(f"  alt set {a + 1}: F1={r['f1']:.4f} TPR={r['tpr']:.3f} FPR={100 * r['fpr']:.2f}%")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "R2_sampling_sensitivity.csv"), index=False)
    R["R2_sampling_sensitivity"] = dict(sets=out, f1_mean=float(df.f1.mean()), f1_min=float(df.f1.min()),
                                        f1_max=float(df.f1.max()))
    save()

# =============================================================================== R3
if "R3" not in SKIP:
    log("R3 partition-clustered paired bootstrap")
    oof_file = os.environ.get("OOF_FILE", "")
    if oof_file and os.path.exists(oof_file):
        O = pd.read_csv(oof_file)
        src = oof_file
    else:     # fallback: record-level models computed here
        big, feats = load_study(study_files)
        tg = big[big["target"]].reset_index(drop=True)
        O = tg[["target_id", "partition", "y"]].copy()
        for name in ["LR, T = 1", "DT, T = 1", "RF, T = 1", "XGB, T = 1"]:
            p = np.zeros(len(tg))
            for tr, te in GroupKFold(5).split(tg, tg["y"], tg["partition"]):
                T, _, _ = prep(tg.loc[tr], feats)
                m = {"LR": LogisticRegression(max_iter=2000), "DT": DecisionTreeClassifier(random_state=SEED),
                     "RF": RandomForestClassifier(50 if QUICK else 200, n_jobs=-1, random_state=SEED),
                     "XGB": xgb()}[name.split(",")[0]]
                p[te] = m.fit(T(tg.loc[tr]), tg.loc[tr, "y"]).predict_proba(T(tg.loc[te]))[:, 1]
            O[name] = p
        src = "computed in R3 (record-level models only)"
    cfgs = [c for c in O.columns if c not in ("target_id", "partition", "y")]
    ref = "XGB, T = 1"
    y = O["y"].values
    H = {c: (O[c].values >= 0.5).astype(int) for c in cfgs}
    # per-partition sufficient statistics -> fast cluster bootstrap of pooled F1
    parts = np.unique(O["partition"])
    S = {c: np.array([[((H[c] == 1) & (y == 1) & (O.partition == p)).sum(),
                       ((H[c] == 1) & (y == 0) & (O.partition == p)).sum(),
                       ((H[c] == 0) & (y == 1) & (O.partition == p)).sum()] for p in parts]) for c in cfgs}
    f1 = lambda s: 2 * s[..., 0] / np.maximum(2 * s[..., 0] + s[..., 1] + s[..., 2], 1)
    obs = {c: float(f1(S[c].sum(0))) for c in cfgs}
    B = rng.integers(0, len(parts), size=(N_BOOT, len(parts)))
    boot = {c: f1(S[c][B].sum(1)) for c in cfgs}
    best = np.array(cfgs)[np.argmax(np.stack([boot[c] for c in cfgs]), 0)]
    comp = [c for c in cfgs if c != ref]
    D = {c: boot[ref] - boot[c] for c in comp}
    dobs = {c: obs[ref] - obs[c] for c in comp}
    sd = {c: float(np.std(D[c], ddof=1)) for c in comp}
    # simultaneous 95% band: max over comparisons of |standardised deviation|
    tmax = np.max(np.stack([np.abs(D[c] - dobs[c]) / max(sd[c], 1e-12) for c in comp]), 0)
    qsim = float(np.percentile(tmax, 95))
    rows = []
    for c in comp:
        lo, hi = np.percentile(D[c], [2.5, 97.5])
        lo90, hi90 = np.percentile(D[c], [5, 95])
        err_ref = int((H[ref] != y).sum()); err_c = int((H[c] != y).sum())
        rows.append({"competitor": c, "dF1 (XGB minus competitor)": dobs[c], "CI95 low": lo, "CI95 high": hi,
                     "simultaneous CI low": dobs[c] - qsim * sd[c], "simultaneous CI high": dobs[c] + qsim * sd[c],
                     "errors per 10,000 targets (XGB - competitor)": 1e4 * (err_ref - err_c) / len(y),
                     "partitions favouring XGB": int(sum(f1(S[ref][i]) > f1(S[c][i]) for i in range(len(parts)))),
                     "equivalent within +/-0.005 (90% CI inside margin)": bool(lo90 > -0.005 and hi90 < 0.005)})
    tab = pd.DataFrame(rows).sort_values("dF1 (XGB minus competitor)")
    tab.to_csv(os.path.join(OUT, "R3_cluster_bootstrap.csv"), index=False)
    sel = pd.Series(best).value_counts(normalize=True)
    R["R3_cluster_bootstrap"] = dict(source=src, n_boot=N_BOOT, unit="partition (15 clusters)",
                                     observed_f1=obs, comparisons=tab.to_dict("records"),
                                     selection_frequency=sel.to_dict(), simultaneous_quantile=qsim,
                                     note="Percentile intervals from resampling whole partitions with replacement; "
                                          "'selection_frequency' = share of resamples in which each configuration "
                                          "has the highest pooled F1 (selection uncertainty).")
    log(tab.to_string(index=False))
    save()

# =============================================================================== R4 + R5
if not {"R4", "R5"} <= SKIP:
    log("R4/R5 ARP-vs-rest detector re-assessment and prevalence analysis")
    import shap
    big, oth, feats = load_study(study_files, keep_other=True, other_cap=OTHER_PER_CLASS)
    tg = big[big["target"]].reset_index(drop=True)
    oth = oth.reset_index(drop=True)
    folds = list(GroupKFold(5).split(tg, tg["y"], tg["partition"]))
    fparts = [set(tg.loc[te, "partition"]) for _, te in folds]
    noiat = [c for c in feats if c != "IAT"]
    P = {m: (np.full(len(tg), np.nan), np.full(len(oth), np.nan)) for m in ["M0", "M1", "M1_noIAT"]}
    lat, glob_imp, dele = [], [], []
    for k, (tr, te) in enumerate(folds):
        trp = set(tg.loc[tr, "partition"])
        otr = oth["partition"].isin(trp).values
        ote = oth["partition"].isin(fparts[k]).values
        for m, F, use_oth in [("M0", feats, False), ("M1", feats, True), ("M1_noIAT", noiat, True)]:
            train = pd.concat([tg.loc[tr, F + ["y"]], oth.loc[otr, F + ["y"]]]) if use_oth else tg.loc[tr, F + ["y"]]
            T, imp, sc = prep(train, F)
            clf = xgb().fit(T(train), train["y"].values)
            P[m][0][te] = clf.predict_proba(T(tg.loc[te]))[:, 1]
            P[m][1][ote] = clf.predict_proba(T(oth.loc[ote]))[:, 1]
            if m != "M1":
                continue
            ex = shap.TreeExplainer(clf)
            Xte = T(tg.loc[te]); p = P[m][0][te]
            al = np.where(p >= 0.5)[0]
            al = np.random.default_rng(k).choice(al, min(N_ALERTS, len(al)), replace=False)
            for i in al[:3]:
                ex.shap_values(Xte[i:i + 1])                     # warm-up
            for i in al:
                t0 = time.perf_counter(); ex.shap_values(Xte[i:i + 1]); lat.append(1e3 * (time.perf_counter() - t0))
            samp = np.random.default_rng(SEED).choice(len(Xte), min(2000, len(Xte)), replace=False)
            glob_imp.append(np.abs(ex.shap_values(Xte[samp])).mean(0))
            sv = ex.shap_values(Xte[al])
            ben_tr = T(train[train["y"] == 0])
            repl = {"training mean (0)": np.zeros(len(F)),
                    "training median": sc.transform(imp.statistics_.reshape(1, -1))[0],
                    "benign-class mean": ben_tr.mean(0)}
            r_ = np.random.default_rng(100 + k)
            for kk in [1, 3, 5, 10]:
                top = np.argsort(-np.abs(sv), 1)[:, :kk]
                rnd = np.array([r_.choice(len(F), kk, replace=False) for _ in al])
                for rname, rv in repl.items():
                    for meth, cols in [("SHAP top-k", top), ("random k", rnd)]:
                        Xd = Xte[al].copy()
                        for i in range(len(al)):
                            Xd[i, cols[i]] = rv[cols[i]]
                        dele.append(dict(fold=k + 1, k=kk, replacement=rname, features=meth,
                                         drop=float((p[al] - clf.predict_proba(Xd)[:, 1]).mean())))
        log(f"  fold {k + 1} done")
    yt = tg["y"].values
    my = np.concatenate([yt, np.zeros(len(oth), int)])
    summ = {}
    for m in P:
        pt, po = P[m]
        mp = np.concatenate([pt, po])
        summ[m] = dict(benign_arp=rates(yt, pt), other_alert_rate=float((po >= 0.5).mean()),
                       n_other=int(len(po)), mixed=rates(my, mp))
    gi = pd.DataFrame(glob_imp, columns=feats)
    rho = [stats.spearmanr(gi.iloc[i], gi.iloc[j])[0] for i in range(len(gi)) for j in range(i + 1, len(gi))]
    dl = pd.DataFrame(dele).groupby(["replacement", "k", "features"])["drop"].agg(["mean", "std"]).reset_index()
    dl.to_csv(os.path.join(OUT, "R4_deletion_arp_vs_rest.csv"), index=False)
    # retraining stream on the ARP-vs-rest formulation (batches = partitions 4..15)
    allp = pd.concat([tg[feats + ["y", "partition"]], oth[feats + ["y", "partition"]]], ignore_index=True)
    np_ = sorted(allp["partition"].unique())
    stream = []
    base = allp[allp.partition.isin(np_[:3])]
    T0, _, _ = prep(base, feats); static = xgb().fit(T0(base), base["y"])
    for b in np_[3:]:
        test = allp[allp.partition == b]
        seen = allp[allp.partition < b]
        row = {"batch": int(b) + 1}
        sm = seen.sample(len(base), random_state=SEED + int(b))
        rc = allp[allp.partition.isin([b - 3, b - 2, b - 1])]
        for name, trn, mdl in [("static", base, static), ("cumulative", seen, None),
                               ("size_matched", sm, None), ("recent_window_3", rc, None)]:
            Tn, _, _ = prep(trn, feats)
            mdl = mdl or xgb().fit(Tn(trn), trn["y"])
            row[name] = float(f1_score(test["y"], (mdl.predict_proba(Tn(test))[:, 1] >= 0.5).astype(int), zero_division=0))
        stream.append(row)
    st = pd.DataFrame(stream)
    st.to_csv(os.path.join(OUT, "R4_retraining_arp_vs_rest.csv"), index=False)
    wil = {}
    for c in ["cumulative", "size_matched", "recent_window_3"]:
        d = st[c] - st["static"]
        wil[c] = dict(mean_gain=float(d.mean()),
                      wilcoxon_p=float(stats.wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() > 1 else None)
    imp_m1 = gi.mean().sort_values(ascending=False)
    R["R4_arp_vs_rest"] = dict(summary=summ,
                               shap_latency_ms=dict(mean=float(np.mean(lat)), median=float(np.median(lat)),
                                                    p95=float(np.percentile(lat, 95)), n=len(lat)),
                               global_importance_top10=imp_m1.head(10).to_dict(),
                               cross_fold_spearman=dict(mean=float(np.mean(rho)), min=float(np.min(rho))),
                               deletion=dl.to_dict("records"), retraining=dict(per_batch=stream, gains=wil))
    # ---------------- R5 prevalence: rates are taken from the out-of-fold predictions above
    mixes = [(0.001, 0.01), (0.001, 0.10), (0.01, 0.01), (0.01, 0.10), (0.05, 0.10), (0.05, 0.30)]
    prev = []
    for m in ["M0", "M1"]:
        tpr = summ[m]["benign_arp"]["tpr"]; fpr_b = summ[m]["benign_arp"]["fpr"]; fpr_o = summ[m]["other_alert_rate"]
        for pa, po in mixes:
            pb = 1 - pa - po
            tp, fp = tpr * pa, fpr_b * pb + fpr_o * po
            prec = tp / (tp + fp)
            prev.append(dict(model=m, arp_share=pa, other_attack_share=po, benign_share=pb, precision=prec,
                             f1=2 * prec * tpr / (prec + tpr), false_alerts_per_10k_flows=1e4 * fp,
                             true_alerts_per_10k_flows=1e4 * tp))
    pv = pd.DataFrame(prev)
    pv.to_csv(os.path.join(OUT, "R5_prevalence.csv"), index=False)
    R["R5_prevalence"] = prev
    log(pv.to_string(index=False))
    save()

# =============================================================================== R6
if "R6" not in SKIP:
    log("R6 timing protocol (same alerts for TreeSHAP and LIME, repeated)")
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    big, feats = load_study(study_files)
    tg = big[big["target"]].reset_index(drop=True)
    tr, te = next(iter(GroupKFold(5).split(tg, tg["y"], tg["partition"])))
    T, imp, sc = prep(tg.loc[tr], feats)
    Xtr, Xte = T(tg.loc[tr]), T(tg.loc[te])
    clf = xgb().fit(Xtr, tg.loc[tr, "y"])
    p = clf.predict_proba(Xte)[:, 1]
    al = np.random.default_rng(SEED).choice(np.where(p >= 0.5)[0], N_LIME, replace=False)
    ex = shap.TreeExplainer(clf)
    lx = LimeTabularExplainer(Xtr[:5000], feature_names=feats, class_names=["benign", "arp"],
                              discretize_continuous=False, random_state=SEED)
    ex.shap_values(Xte[al[:1]]); lx.explain_instance(Xte[al[0]], clf.predict_proba, num_samples=200)
    sh, li = [], []
    for run in range(3):
        for i in al:
            t0 = time.perf_counter(); ex.shap_values(Xte[i:i + 1]); sh.append(dict(run=run, ms=1e3 * (time.perf_counter() - t0)))
            t0 = time.perf_counter(); lx.explain_instance(Xte[i], clf.predict_proba, num_features=len(feats),
                                                          num_samples=2000)
            li.append(dict(run=run, ms=1e3 * (time.perf_counter() - t0)))
    raw = tg.loc[te, feats].iloc[al]
    e2e = []
    for _ in range(3):
        for j in range(len(raw)):
            t0 = time.perf_counter()
            z = sc.transform(imp.transform(raw.iloc[j:j + 1]))
            if clf.predict_proba(z)[0, 1] >= 0.5:
                ex.shap_values(z)
            e2e.append(1e3 * (time.perf_counter() - t0))

    def summ_t(v):
        v = np.asarray(v); bs = [np.mean(rng.choice(v, len(v))) for _ in range(1000)]
        return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                    ci95_mean=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(len(v)))
    R["R6_timing"] = dict(alerts=int(len(al)), repetitions=3,
                          treeshap_ms=summ_t([d["ms"] for d in sh]), lime_ms=summ_t([d["ms"] for d in li]),
                          end_to_end_single_record_ms=summ_t(e2e),
                          protocol="fold 1; identical alerts for both explainers; one warm-up call each; "
                                   "time.perf_counter wall-clock; single process; hardware in meta.environment")
    save()

R["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("done ->", OUT)
