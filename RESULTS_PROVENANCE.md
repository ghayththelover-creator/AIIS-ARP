# Provenance of the reported results (thesis and paper, revision round 3)

Every number in the thesis (*An Adaptive and Interpretable Intelligent System for Early Detection of ARP
Spoofing and Man-in-the-Middle Attacks*) and in the companion paper comes from one of the runs below.
All runs use the same 15 CICIoT2023 partitions (`part-00000` … `part-00014`), the same GroupKFold(5)
partition-level folds and the same 106,382 target records (`round2/partition_manifest.csv`,
`round2/split_indices.json`).

No experiment was modified after the second review round. **The hybrid and fusion models were not re-run
with out-of-fold embeddings**: every reported hybrid/fusion value (runs A and B) uses in-sample LSTM
embeddings. `EMB_MODE=oof` (default of the root `4_revision_experiments.py`, v2.0) is implemented but has
no reported results.

| Run | Code | Output files in this repository | Thesis | Paper |
|---|---|---|---|---|
| **A** – original 16-configuration benchmark (Kaggle notebook `notebookbb18ce43db`, version 1; 2-core CPU) | notebook v1 | `original_run/` | Tables 5.1–5.4; Figures 5.1–5.11; Sections 5.3.1–5.3.6, 5.5 | Tables 3–5; Figures 2–7; Sections 6.1, 6.3 (Friedman χ² = 74.47, McNemar), 6.4, 6.5 |
| Supplementary runs (scripts 6–8; 2-core CPU) | `6_aiis_arp_supplementary.py`, `7_aiis_arp_vs_rest.py`, `8_deletion_check.py` | `supp_out/` (`supp_report.md`, `arp_vs_rest_report.md`, `S7b_deletion_all_folds.csv`, CSV tables) | Tables 5.12, 5.14, 5.15; Section 5.6.8 (IAT KS stability); Section 5.6.17 (31.7 % / 29.2 % negative controls) | Sections 6.4 (controls), 6.5 (five-fold deletion), 6.8 (negative controls) |
| Ablation and timing run (2-core CPU) | `1_iat_ablation_and_timing.py` | `1_iat_ablation_results.json` | Table 5.5; Figure 5.12; Appendix C; 2.31 µs / 0.286 ms | Section 6.6; Section 7.4 and Table 8 |
| **B** – revision run E0–E8 (notebook version 2, 21 Oct 2026, 08:56–11:16) | `kaggle_v2_run/code/` | `revision_out/` | Tables 5.7–5.11; Figures 5.14–5.19; Sections 5.6.5–5.6.13 | Sections 5.1 (hold-out), 6.2 (shuffle), 6.3 (re-run, Nemenyi), 6.6, 6.8 (unseen partitions), 7.6 (tuning) |
| **Round 2** – R1–R6 (notebook `aiis-arp-round-2-analyses`, version 1, 25 Oct 2026, 4-core CPU) | `10_round2_analyses.py` (release v2.1) | `round2/` | Tables 5.13, 5.16, 5.17; Sections 5.6.9 (R2), 5.6.13 (R6), 5.6.17 (R4) | Section 6.3 (bootstrap), 6.5 (R6), 6.8 and Tables 6–7, 7.6 (R2) |
| Prototype run (same notebook, 25 Oct 2026) | `5_aiis_arp_service.py` | `prototype_demo/` | Appendix D | — |

## Values updated in revision round 3 (and where)

| Value | Old | New (source file) | Thesis | Paper |
|---|---|---|---|---|
| TPR of the ARP-versus-rest detector | 0.822 | 0.821 (`round2/round2_results.json`, R4 M1: 19,160 / 23,324) | 5.6.17, 5.7, 6.4, 7.1, Tables 5.16–5.17 | Abstract, Sec. 1, 6.8, 8, Tables 6–7 |
| F1 of ARP-versus-rest without IAT (benign + ARP) | 0.785 | 0.784 (R4 M1_noIAT: 0.78449) | Table 5.16, 5.6.17 | Table 6, 6.8 |
| Mixed-set precision without IAT | 0.911 (derived) | 0.911 (read directly from R4) | — | Table 6 (footnote removed) |
| Prevalence table | rounded rates | unrounded rates (`round2/R5_prevalence.csv`); max. change 0.004 in precision/F1, 1 alert per 10,000 flows | Table 5.17, 5.6.17, 6.2 | Table 7, 6.8 |
| Bootstrap 95 % CI (RF, DT, LR) | 0.0049–0.0082; 0.0336–0.0388; 0.1741–0.1813 | 0.0050–0.0082; 0.0336–0.0385; 0.1739–0.1813 (`round2/R3_cluster_bootstrap.csv`) | Table 5.13, 5.6.14 | 6.3 |
| Friedman p of run A (χ² = 74.47, df = 15) | 7.1 × 10⁻¹¹ (typo) | 7.1 × 10⁻¹⁰ (= `chi2.sf(74.47, 15)`) | 5.3.4, Appendix B | 6.3 |

Run A and the record-level part of the round-2 run agree exactly (XGB, T = 1: TP 20,655, FN 2,669, FP 505,
TN 82,553; `round2_results.json`, R4 M0), so Table 5.12 / the pooled confusion matrix are reproduced by files
already in `round2/`.
