#!/usr/bin/env python3
"""
Writes SYNTHETIC CSV partitions with the CICIoT2023 column layout (46 features + label) so that
every replication script can be smoke-tested without the real data.  The numbers produced on
these files are meaningless and must never be reported.
Usage: python tests/make_synthetic_ciciot.py OUTDIR [N_PARTS=20] [ROWS=1500]
"""
import os
import sys

import numpy as np
import pandas as pd

COLS = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
        "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
        "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count",
        "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
        "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
        "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"]
OTHER = ["DDoS-UDP_Fragmentation", "Mirai-udpplain", "DNS_Spoofing", "Recon-HostDiscovery",
         "DoS-SYN_Flood", "Backdoor_Malware"]


def main(out, n_parts=20, rows=1500):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    for k in range(n_parts):
        lab = rng.choice(["BenignTraffic", "MITM-ArpSpoofing"] + OTHER, rows,
                         p=[0.45, 0.13] + [0.42 / len(OTHER)] * len(OTHER))
        X = rng.normal(size=(rows, len(COLS)))
        arp = lab == "MITM-ArpSpoofing"
        oth = ~np.isin(lab, ["BenignTraffic", "MITM-ArpSpoofing"])
        X[arp, 39] += 2.0                 # IAT shifted for ARP
        X[arp, 18] -= 1.0
        X[oth, 39] += 1.0 * rng.random(oth.sum())
        X[:, 39] = np.where(rng.random(rows) < 0.5, X[:, 39], X[:, 39] + 8)   # bimodal
        X[rng.random(rows) < 0.002, 4] = np.inf
        df = pd.DataFrame(X, columns=COLS)
        df["label"] = lab
        df.to_csv(os.path.join(out, f"part-{k:05d}-synthetic.csv"), index=False)
    print(f"wrote {n_parts} synthetic partitions to {out}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:]))
