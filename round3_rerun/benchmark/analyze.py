"""Summaries, Friedman/Nemenyi, McNemar and partition-level bootstrap for the round-3 re-run."""
import pickle, json, math, sys, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix, precision_score)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/runs/oof_out"
st = pickle.load(open(os.path.join(OUT, "state.pkl"), "rb"))
df = pd.read_csv("/home/claude/data/4efe2522-arp_15parts.csv.gz", usecols=["label", "partition"])
pos = df.groupby("partition").cumcount().values
tgt = np.where(pos >= 19)[0]
y = (df["label"].values[tgt] == "MITM-ArpSpoofing").astype(int)
part = df["partition"].values[tgt]
rows = pd.DataFrame(st["rows"])


def m(yv, p):
    pr = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(yv, pr, labels=[0, 1]).ravel()
    return dict(acc=(tp + tn) / len(yv), precision=tp / max(tp + fp, 1), tpr=tp / (tp + fn), fpr=fp / (fp + tn),
                f1=f1_score(yv, pr), mcc=matthews_corrcoef(yv, pr), roc_auc=roc_auc_score(yv, p),
                pr_auc=average_precision_score(yv, p), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))


summ = {}
for c, p in st["oof"].items():
    if np.isnan(p).any():
        continue
    r = rows[rows.config == c]
    summ[c] = m(y, p) | dict(f1_fold_mean=r.f1.mean(), f1_fold_sd=r.f1.std(ddof=1),
                            us_per_record_train_and_score=None, seconds_per_fold=r.seconds.mean())
S = pd.DataFrame(summ).T.sort_values("f1", ascending=False)


def friedman(cols):
    piv = rows[rows.config.isin(cols)].pivot(index="fold", columns="config", values="f1")[cols]
    chi, p = stats.friedmanchisquare(*[piv[c].values for c in cols])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    k, N = len(cols), len(piv)
    q = stats.studentized_range.ppf(0.95, k, 1e6) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * N))
    return dict(chi2=chi, p=p, cd=cd, ranks=ranks.to_dict(), piv=piv)


base = [c for c in st["oof"] if "(in-sample)" not in c and "(OOF)" not in c]
set_oof = base + [c for c in st["oof"] if "(OOF)" in c]
set_in = base + [c for c in st["oof"] if "(in-sample)" in c]
res = {"summary": S.drop(columns=["us_per_record_train_and_score"]).to_dict(orient="index")}
for name, cols in (("friedman_oof", set_oof), ("friedman_insample", set_in)):
    if len(cols) == 16:
        fr = friedman(cols); fr["piv"].to_csv(os.path.join(OUT, f"fold_f1_matrix_{name}.csv"))
        res[name] = {k: v for k, v in fr.items() if k != "piv"}

best = "XGB, T = 1"; pb = (st["oof"][best] >= .5).astype(int)
mc = {}
for c, p in st["oof"].items():
    if c == best or np.isnan(p).any():
        continue
    pc = (p >= .5).astype(int); okb, okc = pb == y, pc == y
    n10, n01 = int((okb & ~okc).sum()), int((~okb & okc).sum())
    chi = (abs(n10 - n01) - 1) ** 2 / max(n10 + n01, 1); pv = stats.chi2.sf(chi, 1)
    mc[c] = dict(n10=n10, n01=n01, p=pv, p_bonf=min(1, pv * 15), err_diff_per_10k=(n01 - n10) / len(y) * 1e4)
res["mcnemar_vs_xgb_t1"] = mc

# partition-level cluster bootstrap of F1(XGB T=1) - F1(competitor)
rng = np.random.default_rng(42); parts = np.unique(part); idx = {q: np.where(part == q)[0] for q in parts}
boot = {}
for c in [x for x in st["oof"] if x.startswith(("Fusion", "Hybrid", "LSTM", "RF, T = 1"))]:
    p = st["oof"][c]
    if np.isnan(p).any():
        continue
    obs = f1_score(y, pb) - f1_score(y, (p >= .5).astype(int)); d = []
    for _ in range(2000):
        s = np.concatenate([idx[q] for q in rng.choice(parts, len(parts))])
        d.append(f1_score(y[s], pb[s]) - f1_score(y[s], (p[s] >= .5).astype(int)))
    fav = sum(f1_score(y[idx[q]], pb[idx[q]]) > f1_score(y[idx[q]], (p[idx[q]] >= .5).astype(int)) for q in parts)
    boot[c] = dict(dF1=obs, ci_low=float(np.percentile(d, 2.5)), ci_high=float(np.percentile(d, 97.5)),
                   ci90_low=float(np.percentile(d, 5)), ci90_high=float(np.percentile(d, 95)),
                   partitions_favouring_xgb=int(fav))
res["cluster_bootstrap_xgb_minus"] = boot
json.dump(res, open(os.path.join(OUT, "round3_rerun_results.json"), "w"), indent=1, default=float)
oo = pd.DataFrame({"target_id": [f"{a}_{b}" for a, b in zip(part, pos[tgt])], "partition": part, "y": y})
for c, p in st["oof"].items():
    oo[c] = p
oo.to_csv(os.path.join(OUT, "oof_predictions.csv.gz"), index=False, compression="gzip")
pd.set_option("display.width", 250)
print(S[["acc", "precision", "tpr", "fpr", "f1", "f1_fold_mean", "f1_fold_sd", "roc_auc", "pr_auc", "mcc", "tp", "fn", "fp", "tn", "seconds_per_fold"]].round(4))
for k in ("friedman_oof", "friedman_insample"):
    if k in res: print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[k].items() if a != "ranks"}, res[k]["ranks"])
print(pd.DataFrame(mc).T.round(5)); print(pd.DataFrame(boot).T.round(4))
