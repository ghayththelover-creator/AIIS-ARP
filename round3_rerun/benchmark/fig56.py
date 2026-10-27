import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
r = json.load(open('/home/claude/runs/oof_out/round3_rerun_results.json'))['mcnemar_vs_xgb_t1']
order = [("Fusion LSTM-XGB (OOF), T = 10", "Fusion\nOOF T=10"), ("Fusion LSTM-XGB (OOF), T = 20", "Fusion\nOOF T=20"),
         ("XGB, T = 5", "XGB\nT=5"), ("RF, T = 1", "RF\nT=1"), ("XGB, T = 20", "XGB\nT=20"), ("DT, T = 1", "DT\nT=1"),
         ("Hybrid LSTM-RF (OOF), T = 20", "Hybrid\nOOF T=20"), ("LSTM, T = 20", "LSTM\nT=20"), ("LR, T = 1", "LR\nT=1")]
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=200)
x = np.arange(len(order)); w = .38
a = [r[k]["n10"] for k, _ in order]; b = [r[k]["n01"] for k, _ in order]
ax.bar(x - w/2, a, w, color="#1f6fc5", label="n10: only AIIS-ARP correct")
ax.bar(x + w/2, b, w, color="#bcc1c8", label="n01: only competitor correct")
ax.set_yscale("log"); ax.set_ylim(100, 2e4)
for i, (k, _) in enumerate(order):
    pb = min(1, r[k]["p"] * 19)
    ax.text(i, max(a[i], b[i]) * 1.25, "*" if pb < .05 else "n.s.", ha="center", fontsize=13 if pb < .05 else 11)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=9.5)
ax.set_ylabel("Discordant records (log scale)"); ax.grid(True, color="#e3e3e3", lw=.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("/home/claude/runs/oof_out/fig_discordant_round3.png")
