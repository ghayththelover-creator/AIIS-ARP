# Provenance of the reported results (thesis and paper, revision round 3)

Every number in the thesis (*An Adaptive and Interpretable Intelligent System for Early Detection of ARP
Spoofing and Man-in-the-Middle Attacks*) and in the companion paper is listed below with the run and the output
file it comes from. All runs use the same 15 CICIoT2023 partitions (`part-00000` … `part-00014`), the same
GroupKFold(5) partition-level folds and the same 106,382 target records (`round2/partition_manifest.csv`,
`round2/split_indices.json`).

| Run | Code | Output files | Thesis | Paper |
|---|---|---|---|---|
| **A** – original 16-configuration benchmark (Kaggle notebook `notebookbb18ce43db`, version 1; 2-core CPU) | notebook v1 | not archived; record-level predictions reproduced exactly by `round2/` (R3, R4) and `round3_rerun/supp/` (S3); all configurations re-run in `round3_rerun/benchmark/` | Tables 5.1, 5.3, 5.4; Figures 5.1–5.5, 5.7–5.11; Sections 5.3.1–5.3.6, 5.5 | Tables 3, 5; Figures 2–7; Sections 6.1, 6.4, 6.5 |
| **B** – revision run E0–E8 (notebook version 2, 21 Oct 2026) | `kaggle_v2_run/code/` | `revision_out/` | Tables 5.7–5.11; Figures 5.14–5.19; Sections 5.6.5–5.6.13 | Sections 5.1 (hold-out), 6.2 (shuffle), 6.3 (run-B re-run), 6.6, 6.8 (unseen partitions), 7.6 (tuning) |
| **Round 2** – R1–R6 (notebook `aiis-arp-round-2-analyses`, version 1, 25 Oct 2026, 4-core CPU) | `10_round2_analyses.py` (release v2.1) | `round2/` | Tables 5.13, 5.16, 5.17; Sections 5.6.9 (R2), 5.6.13 (R6), 5.6.17 (R4, incl. 31.7 % negative control) | Section 6.3 (record-level bootstrap), 6.5 (R6), 6.8 and Tables 6–7, 7.6 (R2) |
| Prototype run (same notebook, 25 Oct 2026) | `5_aiis_arp_service.py` | `prototype_demo/` | Appendix D | — |
| **C** – round-3 re-run of all 16 configurations, hybrid/fusion with in-sample **and** out-of-fold second stages (27 Oct 2026, 2-core CPU) | `round3_rerun/benchmark/oof_rerun.py` (+ `analyze.py`, `boot_extra.py`, `fig56.py`) | `round3_rerun/benchmark/` (fold scores, all per-record predictions, statistics, figure) | Table 5.2; Figure 5.6; Sections 3.4.5, 5.3.3, 5.3.4, 5.4, 6.1, 6.4, 7.1 | Table 4; Sections 5.3, 6.2, 6.3, 7.2, 7.6, 8 |
| Round-3 re-run of the ablation and timing script | `1_iat_ablation_and_timing.py` | `round3_rerun/ablation/` | Table 5.5; Figure 5.12; Appendix C (identical confusion matrices) | Section 6.6 |
| Round-3 re-run of the supplementary script | `6_aiis_arp_supplementary.py` | `round3_rerun/supp/` | Tables 5.12 (S3) and 5.14 (S4); Section 5.6.8 (S6); Section 5.6.16 (S7) | Sections 6.4, 6.5 |
| Round-3 re-run of the deletion check | `8_deletion_check.py` | `round3_rerun/deletion/` | Table 5.15 | Section 6.5 |

Known remaining gaps: the 29.2 % negative-control rate (up to 3,000 records per class and partition; thesis
Section 5.6.17, paper Section 6.8) requires the other CICIoT2023 attack classes and was not re-run (script 6,
S8); the timings of run A (5.9 µs per record, 2.7 ms TreeSHAP, 690.6 ms LIME) are hardware-specific and are
superseded for reproduction by `round2/` (R6) and `round3_rerun/ablation/`.

## Values updated in revision round 3

| Value | Old | New (source file) | Thesis | Paper |
|---|---|---|---|---|
| Pairwise comparison of AIIS-ARP with all configurations | run A, in-sample second stages, McNemar only | run C, in-sample and OOF second stages, McNemar (m = 19) + partition bootstrap (`round3_rerun/benchmark/round3_rerun_results.json`) | Table 5.2, Figure 5.6, 3.4.5, 5.3.3, 5.3.4, 5.4, 6.1, 6.4, 7.1, Abstract | Table 4, Abstract, 5.3, 6.2, 6.3, 7.2, 7.6, 8 |
| TPR of the ARP-versus-rest detector | 0.822 | 0.821 (`round2/round2_results.json`, R4 M1: 19,160 / 23,324) | 5.6.17, 5.7, 6.4, 7.1, Tables 5.16–5.17 | Abstract, Sec. 1, 6.8, 8, Tables 6–7 |
| F1 of ARP-versus-rest without IAT (benign + ARP) | 0.785 | 0.784 (R4 M1_noIAT: 0.78449) | Table 5.16, 5.6.17 | Table 6, 6.8 |
| Mixed-set precision without IAT | 0.911 (derived) | 0.911 (read directly from R4) | — | Table 6 (footnote removed) |
| Prevalence table | rounded rates | unrounded rates (`round2/R5_prevalence.csv`); max. change 0.004 in precision/F1, 1 alert per 10,000 flows | Table 5.17, 5.6.17, 6.2 | Table 7, 6.8 |
| Bootstrap 95 % CI (RF, DT, LR) | 0.0049–0.0082; 0.0336–0.0388; 0.1741–0.1813 | 0.0050–0.0082; 0.0336–0.0385; 0.1739–0.1813 (`round2/R3_cluster_bootstrap.csv`) | Table 5.13, 5.6.14 | 6.3 |
| Friedman p of run A (χ² = 74.47, df = 15) | 7.1 × 10⁻¹¹ (typo) | 7.1 × 10⁻¹⁰ (= `chi2.sf(74.47, 15)`) | 5.3.4, Appendix B | 6.3 |
