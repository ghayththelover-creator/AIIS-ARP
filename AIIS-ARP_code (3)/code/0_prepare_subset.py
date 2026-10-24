#!/usr/bin/env python3
"""Builds the study subset of the thesis: Benign + MITM-ArpSpoofing records of the first 15
CICIoT2023 CSV partitions (sorted by file name), with a `partition` column (0-14).
Output: arp_15parts.csv.gz (106,667 records, 46 features).   Usage: python 0_prepare_subset.py"""
import glob, pandas as pd
files = sorted(glob.glob("/kaggle/input/**/CICIoT2023/part-*.csv", recursive=True))[:15]
print(len(files), "files")
parts = []
for i, f in enumerate(files):
    d = pd.read_csv(f)
    d = d[d["label"].isin(["BenignTraffic", "MITM-ArpSpoofing"])].copy()
    d["partition"] = i
    parts.append(d)
big = pd.concat(parts, ignore_index=True)
print(big.shape, big["label"].value_counts().to_dict())
big.to_csv("arp_15parts.csv.gz", index=False, compression="gzip")
