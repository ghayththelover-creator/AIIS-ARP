#!/usr/bin/env python3
"""
AIIS-ARP reference prototype (replay / streaming mode).

Turns the three modules of the thesis into one runnable component:
  detection  - record-level XGBoost on the 46 CICIoT2023 flow features
  explanation- exact TreeSHAP attached to every alert (top-k features with sign)
  adaptation - cumulative retraining on analyst-labelled batches, with a
               promotion guard, model versioning, rollback and integrity hashes

Sub-commands
  train    --data labelled.csv [more.csv ...]          create model version v0001
  detect   --flows flows.csv [--follow] [--top-k 5]    score flow records, write alerts (JSON lines)
  feedback --labelled batch.csv                        append an analyst-labelled batch, retrain,
                                                       promote only if F1 on that batch does not drop
  rollback                                              return to the previous model version
  status                                                list versions and the active one

Input: CSV files with the 46 CICIoT2023 flow-feature columns (as produced by the dataset's
feature extractor); labelled files also carry a `label` column. Payloads are never read.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

ARP = "MITM-ArpSpoofing"
XGB_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1)


class Store:
    """Versioned model store with SHA-256 integrity checks."""

    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "versions"), exist_ok=True)
        self.reg = os.path.join(root, "registry.json")
        self.state = json.load(open(self.reg)) if os.path.exists(self.reg) else {"active": None, "versions": []}

    def _save_state(self):
        json.dump(self.state, open(self.reg, "w"), indent=2)

    def add(self, bundle, meta, activate=True):
        vid = f"v{len(self.state['versions']) + 1:04d}"
        path = os.path.join(self.root, "versions", vid + ".joblib")
        joblib.dump(bundle, path)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        self.state["versions"].append(dict(id=vid, path=path, sha256=digest,
                                           created=time.strftime("%Y-%m-%d %H:%M:%S"), **meta))
        if activate:
            self.state["active"] = vid
        self._save_state()
        return vid

    def load(self, vid=None):
        vid = vid or self.state["active"]
        if vid is None:
            sys.exit("no model: run `train` first")
        v = next(v for v in self.state["versions"] if v["id"] == vid)
        if hashlib.sha256(open(v["path"], "rb").read()).hexdigest() != v["sha256"]:
            sys.exit(f"integrity check failed for {vid}: model file was modified")
        return vid, joblib.load(v["path"])


def read_labelled(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    lab = "label" if "label" in df.columns else "Label"
    y = (df[lab] == ARP).astype(int).values
    X = df.drop(columns=[lab]).replace([np.inf, -np.inf], np.nan)
    return X, y


def fit_bundle(X, y):
    pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
    model = xgb.XGBClassifier(**XGB_PARAMS).fit(pre.transform(X), y)
    return dict(features=list(X.columns), pre=pre, model=model)


def cmd_train(a):
    X, y = read_labelled(a.data)
    st = Store(a.store)
    buf = os.path.join(a.store, "buffer.pkl")
    pd.concat([X, pd.Series(y, name="y")], axis=1).to_pickle(buf)
    vid = st.add(fit_bundle(X, y), dict(n_train=int(len(y)), reason="initial training"))
    print(json.dumps(dict(version=vid, n_train=int(len(y)), attacks=int(y.sum()))))


def explain_top(explainer, z, feats, k):
    phi = explainer.shap_values(z.reshape(1, -1))[0]
    order = np.argsort(-np.abs(phi))[:k]
    return [dict(feature=feats[j], shap=round(float(phi[j]), 4),
                 direction="towards ARP spoofing" if phi[j] > 0 else "towards benign") for j in order]


def cmd_detect(a):
    import shap
    st = Store(a.store)
    vid, b = st.load()
    feats, pre, model = b["features"], b["pre"], b["model"]
    explainer = shap.TreeExplainer(model)
    out = open(a.alerts, "a")
    seen, n_alerts, lat_d, lat_e = 0, 0, [], []

    def process(chunk):
        nonlocal seen, n_alerts
        X = chunk[feats].replace([np.inf, -np.inf], np.nan)
        Z = pre.transform(X)
        for i in range(len(Z)):
            t0 = time.perf_counter()
            p = float(model.predict_proba(Z[i:i + 1])[0, 1])
            lat_d.append((time.perf_counter() - t0) * 1e3)
            seen += 1
            if p >= a.threshold:
                t1 = time.perf_counter()
                top = explain_top(explainer, Z[i], feats, a.top_k)
                lat_e.append((time.perf_counter() - t1) * 1e3)
                n_alerts += 1
                out.write(json.dumps(dict(record=int(chunk.index[i]), model=vid, probability=round(p, 4),
                                          time=time.strftime("%Y-%m-%d %H:%M:%S"), explanation=top)) + "\n")
        out.flush()

    if a.follow:                                   # streaming: process lines appended to the file
        pos = 0
        header = pd.read_csv(a.flows, nrows=0).columns
        while True:
            df = pd.read_csv(a.flows, skiprows=range(1, pos + 1))
            if len(df):
                df.index = range(pos, pos + len(df)); pos += len(df); process(df)
            time.sleep(a.poll)
    else:                                          # replay
        for chunk in pd.read_csv(a.flows, chunksize=5000):
            process(chunk)
    print(json.dumps(dict(model=vid, records=seen, alerts=n_alerts,
                          detect_ms_median=round(float(np.median(lat_d)), 3) if lat_d else None,
                          explain_ms_median=round(float(np.median(lat_e)), 3) if lat_e else None,
                          alerts_file=a.alerts)))


def cmd_feedback(a):
    st = Store(a.store)
    vid, cur = st.load()
    Xb, yb = read_labelled([a.labelled])
    Xb = Xb[cur["features"]]
    cur_f1 = f1_score(yb, (cur["model"].predict_proba(cur["pre"].transform(Xb))[:, 1] >= .5).astype(int))
    buf = os.path.join(a.store, "buffer.pkl")
    B = pd.read_pickle(buf)
    B = pd.concat([B, pd.concat([Xb.reset_index(drop=True), pd.Series(yb, name="y")], axis=1)], ignore_index=True)
    B.to_pickle(buf)
    # candidate is trained on everything seen before this batch (the batch is then used as a
    # fresh check), mirroring the evaluate-then-train protocol of the thesis
    prev = B.iloc[:len(B) - len(yb)]
    cand = fit_bundle(prev.drop(columns="y"), prev["y"].values)
    cand_f1 = f1_score(yb, (cand["model"].predict_proba(cand["pre"].transform(Xb))[:, 1] >= .5).astype(int))
    final = fit_bundle(B.drop(columns="y"), B["y"].values)
    promote = cand_f1 >= cur_f1 - a.tolerance
    new = st.add(final, dict(n_train=int(len(B)), reason="cumulative retraining",
                             check_batch_f1_current=round(cur_f1, 4), check_batch_f1_candidate=round(cand_f1, 4)),
                 activate=promote)
    print(json.dumps(dict(current=vid, current_f1=round(cur_f1, 4), candidate_f1=round(cand_f1, 4),
                          new_version=new, promoted=bool(promote))))


def cmd_rollback(a):
    st = Store(a.store)
    ids = [v["id"] for v in st.state["versions"]]
    i = ids.index(st.state["active"])
    if i == 0:
        sys.exit("already at the first version")
    st.state["active"] = ids[i - 1]; st._save_state()
    print(json.dumps(dict(active=ids[i - 1])))


def cmd_status(a):
    print(json.dumps(Store(a.store).state, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="aiis_store")
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--data", nargs="+", required=True)
    d = sp.add_parser("detect"); d.add_argument("--flows", required=True)
    d.add_argument("--alerts", default="alerts.jsonl"); d.add_argument("--top-k", type=int, default=5)
    d.add_argument("--threshold", type=float, default=0.5); d.add_argument("--follow", action="store_true")
    d.add_argument("--poll", type=float, default=1.0)
    f = sp.add_parser("feedback"); f.add_argument("--labelled", required=True)
    f.add_argument("--tolerance", type=float, default=0.005)
    sp.add_parser("rollback"); sp.add_parser("status")
    a = ap.parse_args()
    dict(train=cmd_train, detect=cmd_detect, feedback=cmd_feedback, rollback=cmd_rollback, status=cmd_status)[a.cmd](a)
