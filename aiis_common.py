"""
aiis_common.py - shared data access for the AIIS-ARP replication scripts.

* One environment variable for the input files: DATA_GLOB (CICIOT_GLOB is accepted as an alias).
* The study subset is always the first N_STUDY_PARTS files in *sorted* path order
  (CICIoT2023 "part-00000-…csv" … "part-00014-…csv"); the exact file names and SHA-256
  hashes are written by 10_round2_analyses.py (R1) to partition_manifest.csv.
* Non-finite values (+/-inf) are converted to NaN and imputed with the training-fold median.
"""
import glob
import hashlib
import os

import numpy as np
import pandas as pd

BENIGN, ARP = "BenignTraffic", "MITM-ArpSpoofing"
SEED = 42
MIN_PRED = 19          # targets need >= 19 predecessors in their partition (T = 20 windows)
N_STUDY_PARTS = 15

XGB_FIXED = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                 colsample_bytree=0.8, tree_method="hist", random_state=SEED,
                 eval_metric="logloss", n_jobs=-1)


def data_glob():
    return os.environ.get("DATA_GLOB", os.environ.get("CICIOT_GLOB", "/kaggle/input/**/part-*.csv"))


def out_dir(default):
    d = os.environ.get("OUT_DIR", default)
    os.makedirs(d, exist_ok=True)
    return d


def list_files():
    files = sorted(glob.glob(data_glob(), recursive=True))
    if len(files) < N_STUDY_PARTS:
        raise SystemExit(f"Found {len(files)} CSV partitions with DATA_GLOB={data_glob()!r}; "
                         f"at least {N_STUDY_PARTS} are needed.")
    return files


def read_part(path):
    df = pd.read_csv(path)
    lab = "label" if "label" in df.columns else "Label"
    df = df.rename(columns={lab: "label"})
    num = [c for c in df.columns if c != "label"]
    df[num] = df[num].astype("float64").replace([np.inf, -np.inf], np.nan)
    return df


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_study(files=None, keep_other=False, other_cap=None, seed=SEED):
    """Benign + ARP records of the study partitions (with partition id and in-partition position).
    Optionally also returns the other-attack records (capped per class and partition)."""
    files = files if files is not None else list_files()[:N_STUDY_PARTS]
    parts, others = [], []
    for k, f in enumerate(files):
        d = read_part(f)
        d["partition"] = k
        d["pfile"] = os.path.basename(f)
        b = d[d["label"].isin([BENIGN, ARP])].copy()
        b["pos"] = np.arange(len(b))
        parts.append(b)
        if keep_other:
            o = d[~d["label"].isin([BENIGN, ARP])]
            if other_cap is not None and len(o):
                o = pd.concat([g.sample(min(len(g), other_cap), random_state=seed + k)
                               for _, g in o.groupby("label")])
            others.append(o)
    big = pd.concat(parts, ignore_index=True)
    feats = [c for c in big.columns if c not in ("label", "partition", "pfile", "pos")]
    big["y"] = (big["label"] == ARP).astype(int)
    big["target"] = big["pos"] >= MIN_PRED
    big["target_id"] = big["partition"].astype(str) + "_" + big["pos"].astype(str)
    if keep_other:
        oth = pd.concat(others, ignore_index=True) if others else pd.DataFrame(columns=big.columns)
        oth["y"] = 0
        return big, oth, feats
    return big, feats


def environment():
    """Software / hardware description written next to every result file."""
    import platform
    info = dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), cpu_count=os.cpu_count())
    for m in ["numpy", "pandas", "sklearn", "scipy", "xgboost", "shap", "lime", "tensorflow"]:
        try:
            info[m] = __import__(m).__version__
        except Exception:
            info[m] = None
    try:
        with open("/proc/cpuinfo") as fh:
            names = [l.split(":", 1)[1].strip() for l in fh if l.startswith("model name")]
        info["cpu_model"] = names[0] if names else None
    except OSError:
        pass
    return info
