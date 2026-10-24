#!/usr/bin/env python3
"""
5_aiis_arp_service.py - AIIS-ARP reference prototype (thesis Section 4.5.1, Appendix D).

An OFFLINE research prototype: it reads flow records (46 CICIoT2023 features) from CSV files,
scores every record, explains every alert with TreeSHAP and writes alerts as JSON lines.
It does NOT capture packets, build flows, deliver alerts to a SIEM or interact with analysts;
labelled batches are supplied as files (feedback command).

Commands
  train    --data FILE [--label-col label] [--arp-vs-rest]
  detect   --flows FILE [--follow] [--top-k 5] [--threshold 0.5] [--out alerts.jsonl]
  feedback --labelled FILE [--tolerance 0.005]
  status
  rollback
Registry: ./aiis_registry (override with --registry DIR or AIIS_REGISTRY).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ARP = "MITM-ArpSpoofing"
BENIGN = "BenignTraffic"
XGB = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
           tree_method="hist", random_state=42, eval_metric="logloss", n_jobs=-1)


# ----------------------------------------------------------------------------- registry
class Registry:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index = os.path.join(root, "registry.json")
        self.data = json.load(open(self.index)) if os.path.exists(self.index) else {"active": None, "versions": []}

    def save(self):
        json.dump(self.data, open(self.index, "w"), indent=2)

    @staticmethod
    def _hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()

    def add(self, bundle, meta, activate):
        v = f"v{len(self.data['versions']) + 1:04d}"
        path = os.path.join(self.root, f"{v}.joblib")
        joblib.dump(bundle, path)
        meta = dict(version=v, file=os.path.basename(path), sha256=self._hash(path),
                    created=time.strftime("%Y-%m-%dT%H:%M:%S"), **meta)
        self.data["versions"].append(meta)
        if activate:
            self.data["active"] = v
        self.save()
        return v

    def load(self, version=None):
        v = version or self.data["active"]
        if v is None:
            sys.exit("No model registered - run `train` first.")
        meta = next(m for m in self.data["versions"] if m["version"] == v)
        path = os.path.join(self.root, meta["file"])
        if self._hash(path) != meta["sha256"]:
            sys.exit(f"Integrity check failed for {v}: file does not match its SHA-256 hash; refusing to load.")
        return joblib.load(path), meta


# ----------------------------------------------------------------------------- model
def read_labelled(path, label_col, arp_vs_rest):
    df = pd.read_csv(path)
    if not arp_vs_rest:
        df = df[df[label_col].isin([BENIGN, ARP])]
    y = (df[label_col] == ARP).astype(int).values
    X = df.drop(columns=[c for c in df.columns if c == label_col or c.lower() in ("partition", "pfile", "pos")])
    return X.replace([np.inf, -np.inf], np.nan), y


def fit_bundle(X, y):
    feats = list(X.columns)
    imp = SimpleImputer(strategy="median").fit(X)
    sc = StandardScaler().fit(imp.transform(X))
    clf = XGBClassifier(**XGB).fit(sc.transform(imp.transform(X)), y)
    return dict(features=feats, imputer=imp, scaler=sc, model=clf)


def transform(bundle, X):
    X = X[bundle["features"]].replace([np.inf, -np.inf], np.nan)
    return bundle["scaler"].transform(bundle["imputer"].transform(X))


# ----------------------------------------------------------------------------- commands
def cmd_train(a, reg):
    X, y = read_labelled(a.data, a.label_col, a.arp_vs_rest)
    t0 = time.perf_counter()
    b = fit_bundle(X, y)
    b["arp_vs_rest"] = a.arp_vs_rest
    b["buffer"] = (X, y)
    v = reg.add(b, dict(reason="initial training", n_train=int(len(y)), n_attack=int(y.sum()),
                        arp_vs_rest=a.arp_vs_rest, train_seconds=round(time.perf_counter() - t0, 3)),
                activate=True)
    print(json.dumps({"version": v, "n_train": int(len(y)), "active": True}))


def _explainer(bundle):
    import shap
    return shap.TreeExplainer(bundle["model"])


def score_frame(df, bundle, version, ex, k, thr, out, offset=0):
    t0 = time.perf_counter()
    Z = transform(bundle, df)
    p = bundle["model"].predict_proba(Z)[:, 1]
    t_det = (time.perf_counter() - t0) / max(len(df), 1)
    n_alert, t_exp = 0, []
    for i in np.where(p >= thr)[0]:
        t1 = time.perf_counter()
        sv = ex.shap_values(Z[i:i + 1])[0]
        t_exp.append(time.perf_counter() - t1)
        top = np.argsort(-np.abs(sv))[:k]
        rec = {"record_id": int(df.index[i]) + offset if "record_id" not in df.columns else df.iloc[i]["record_id"],
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "probability": round(float(p[i]), 4), "threshold": thr, "model_version": version,
               "top_features": [{"feature": bundle["features"][j],
                                 "value": None if pd.isna(df.iloc[i][bundle["features"][j]])
                                 else round(float(df.iloc[i][bundle["features"][j]]), 6),
                                 "shap": round(float(sv[j]), 4),
                                 "direction": "towards ARP spoofing" if sv[j] > 0 else "towards benign"}
                                for j in top]}
        out.write(json.dumps(rec) + "\n")
        n_alert += 1
    out.flush()
    return len(df), n_alert, t_det, t_exp


def cmd_detect(a, reg):
    bundle, meta = reg.load()
    ex = _explainer(bundle)
    tot, alerts, dets, exps = 0, 0, [], []
    with open(a.out, "a") as out:
        pos = 0
        while True:
            df = pd.read_csv(a.flows, skiprows=range(1, pos + 1)) if pos else pd.read_csv(a.flows)
            if len(df):
                df = df.drop(columns=[c for c in df.columns if c.lower() in ("label", "partition", "pfile", "pos")])
                n, na, td, te = score_frame(df, bundle, meta["version"], ex, a.top_k, a.threshold, out, offset=pos)
                tot += n; alerts += na; dets.append(td); exps += te; pos += n
            if not a.follow:
                break
            time.sleep(a.poll)          # tail mode: re-read the file and score only appended rows
    summ = {"records": tot, "alerts": alerts, "model_version": meta["version"],
            "detection_us_per_record_batched": round(1e6 * float(np.mean(dets)), 2) if dets else None,
            "explanation_ms_per_alert_mean": round(1e3 * float(np.mean(exps)), 3) if exps else None,
            "alerts_file": a.out}
    print(json.dumps(summ))


def cmd_feedback(a, reg):
    bundle, meta = reg.load()
    Xn, yn = read_labelled(a.labelled, a.label_col, bundle.get("arp_vs_rest", False))
    Xo, yo = bundle["buffer"]
    X = pd.concat([Xo, Xn[Xo.columns]], ignore_index=True)
    y = np.concatenate([yo, yn])
    new = fit_bundle(X, y)
    new["arp_vs_rest"] = bundle.get("arp_vs_rest", False)
    new["buffer"] = (X, y)
    # promotion guard: the retrained model must not be worse than the current one on the new batch
    # (by more than the tolerance).  NOTE: the new batch is also part of the retraining buffer, so
    # this guard is a sanity check against degraded/poisoned updates, not an independent evaluation.
    f_old = f1_score(yn, (bundle["model"].predict_proba(transform(bundle, Xn))[:, 1] >= 0.5).astype(int), zero_division=0)
    f_new = f1_score(yn, (new["model"].predict_proba(transform(new, Xn))[:, 1] >= 0.5).astype(int), zero_division=0)
    promote = f_new >= f_old - a.tolerance
    v = reg.add(new, dict(reason=f"feedback {os.path.basename(a.labelled)}", n_train=int(len(y)),
                          n_attack=int(y.sum()), f1_current_on_batch=round(float(f_old), 4),
                          f1_candidate_on_batch=round(float(f_new), 4), promoted=bool(promote),
                          parent=meta["version"]), activate=promote)
    print(json.dumps({"candidate": v, "promoted": bool(promote), "f1_current": round(float(f_old), 4),
                      "f1_candidate": round(float(f_new), 4), "active": reg.data["active"]}))


def cmd_status(a, reg):
    print(json.dumps(reg.data, indent=2))


def cmd_rollback(a, reg):
    vs = [m["version"] for m in reg.data["versions"]]
    act = reg.data["active"]
    if act is None or vs.index(act) == 0:
        sys.exit("Nothing to roll back to.")
    prev = [m["version"] for m in reg.data["versions"][:vs.index(act)]
            if m.get("promoted", True)]
    reg.data["active"] = prev[-1] if prev else vs[0]
    reg.save()
    reg.load()               # integrity check of the restored version
    print(json.dumps({"active": reg.data["active"], "rolled_back_from": act}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.environ.get("AIIS_REGISTRY", "aiis_registry"))
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--data", required=True)
    t.add_argument("--label-col", default="label"); t.add_argument("--arp-vs-rest", action="store_true")
    d = sp.add_parser("detect"); d.add_argument("--flows", required=True)
    d.add_argument("--follow", action="store_true"); d.add_argument("--poll", type=float, default=2.0)
    d.add_argument("--top-k", type=int, default=5); d.add_argument("--threshold", type=float, default=0.5)
    d.add_argument("--out", default="alerts.jsonl")
    f = sp.add_parser("feedback"); f.add_argument("--labelled", required=True)
    f.add_argument("--label-col", default="label"); f.add_argument("--tolerance", type=float, default=0.005)
    sp.add_parser("status"); sp.add_parser("rollback")
    a = ap.parse_args()
    reg = Registry(a.registry)
    dict(train=cmd_train, detect=cmd_detect, feedback=cmd_feedback, status=cmd_status,
         rollback=cmd_rollback)[a.cmd](a, reg)


if __name__ == "__main__":
    main()
