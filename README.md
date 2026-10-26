# AIIS-ARP — replication package, version 2.3 (third review round)

Thesis: *An Adaptive and Interpretable Intelligent System for Early Detection of ARP Spoofing and
Man-in-the-Middle Attacks* (MSc Cybersecurity, Northern Border University) and the companion journal paper.

AIIS-ARP is an **offline research prototype**. All scripts read the pre-computed CICIoT2023 flow-feature CSV
partitions; nothing here captures packets or runs in a live network.

## Data access
* Kaggle: add the dataset **"UNB CIC IOT 2023 Dataset"** (`madhavmalhotra/unb-cic-iot-dataset`) to a notebook, or
* download the CSV release from the Canadian Institute for Cybersecurity and set `DATA_GLOB` to the `part-*.csv` files.

The study subset is the **first 15 files in sorted path order** (`part-00000` … `part-00014`); benign and
MITM-ArpSpoofing records are extracted, giving 106,667 records and 106,382 targets (>= 19 predecessors).
`10_round2_analyses.py` (R1) writes the exact file names, SHA-256 hashes, per-class counts, the
fold-to-partition mapping and all target ids per fold (`partition_manifest.csv`, `split_indices.json`).

## Quick start
```bash
pip install -r requirements.txt
bash tests/smoke_test.sh            # executes every script and prototype command on synthetic data (~30 min CPU)
export DATA_GLOB="/kaggle/input/**/part-*.csv"
python 1_iat_ablation_and_timing.py # then the other scripts in numerical order
```
Environment variables: `DATA_GLOB` (alias `CICIOT_GLOB`), `OUT_DIR`, `QUICK=1` (smoke settings), `SKIP=E1,E5`
(scripts 4 and 10), `N_TUNE`, `N_TUNE_LSTM`, `EMB_MODE=oof|insample`, `OOF_FILE`, `N_ALT`, `OTHER_PER_CLASS`.

## Scripts → thesis / paper outputs
| Script | Reproduces | Runtime (Kaggle, 2-core CPU) |
|---|---|---|
| `1_iat_ablation_and_timing.py` | Thesis 5.6.1, Table 5.5, Appendix C (IAT ablation; batched, single-record and pre-processing+scoring timing; environment) | ~5 min |
| `3_statistics_checks.py` | MCC, Wilson CIs, Friedman p, McNemar + Bonferroni, exact Wilcoxon (Chapter 5) | seconds |
| `4_revision_experiments.py` | 5.7.1–5.7.9: E0 duplicates, E1 16-config benchmark (fold scores, **oof_predictions.csv**, CD diagram), E2 shuffle, E3 IAT, E4 external hold-out, E5 nested tuning (XGBoost **and LSTM, equal budget**), E6–E8 | full: 2–4 h GPU |
| `5_aiis_arp_service.py` | Appendix D reference prototype (train / detect / feedback / status / rollback) | seconds–minutes |
| `6_aiis_arp_supplementary.py` | 5.7.10–5.7.12, Tables 5.12–5.15 | ~30 min |
| `7_aiis_arp_vs_rest.py` | 5.7.13, Table 5.16 (ARP-vs-rest) | ~10 min |
| `8_deletion_check.py` | Table 5.15 (deletion test, 5 folds × 200 alerts) | ~10 min |
| `9_fast_e2_e5.py` | Tables 5.7 and 5.9 (shortened shuffle control and XGBoost tuning) | ~50 min (N_TUNE=6) |
| `10_round2_analyses.py` | **New**: R1 provenance, R2 sampling sensitivity (alternative 15-partition sets), R3 partition-clustered bootstrap with selection frequency, simultaneous CIs and equivalence check, R4 re-assessment of the ARP-vs-rest detector (SHAP latency, attribution, deletion, IAT ablation, retraining controls), R5 prevalence analysis, R6 timing protocol | ~1–2 h |
| `11_cd_diagram.py` | **New**: Friedman / Nemenyi CD diagram (Figure for 5.7.2) from a fold × configuration F1 matrix | seconds |

## Changes in version 2 (response to the second review)
* Hybrid / fusion second stage now fitted on **inner out-of-fold LSTM embeddings** (`EMB_MODE=oof`, default);
  `EMB_MODE=insample` reproduces the original thesis run.
* E1 writes `oof_predictions.csv` and `E1_fold_f1_matrix.csv`, so the clustered bootstrap (R3) and the
  CD diagram (11) can be computed for all 16 configurations, including the XGB-versus-fusion comparison.
* E5 tunes the LSTM by random search with the **same number of candidates** as XGBoost.
* One input variable (`DATA_GLOB`, alias `CICIOT_GLOB`) and `OUT_DIR` in all scripts; ±inf treated as missing.
* Timing protocol made explicit (warm-up, repetitions, medians, single record with and without pre-processing) and
  every run records its software/hardware environment.
* Added the reference prototype `5_aiis_arp_service.py`, the shared module `aiis_common.py`, `requirements.txt`
  and `tests/` (synthetic data generator + smoke test). `2_make_figure1.py` is not part of the package
  (Figure 4.1 was drawn manually and contains no data).

## Seeds
42 for XGBoost, Random Forest, decision tree, all sampling and bootstrap; 0, 1, 2 for the LSTM runs;
GroupKFold(5) on the partition index is deterministic.

## Results files to archive with the code (from the Kaggle runs)
`1_iat_ablation_results.json`, `revision_results.json` (+ CSV/PNG), `supp_report.md` (+ CSV),
`arp_vs_rest_report.md` (+ CSV), `S7b_deletion_all_folds.csv`, `fast_results.json`, and — after running them —
`round2_results.json` and `cd_summary.json`.

## Provenance of the reported revision results (E0–E8)
The numbers reported in thesis Section 5.7 and in the paper (E0–E8, including the CD diagram,
Tables 5.7 and 5.9) come from **version 2 of the Kaggle revision notebook**
(full run, `quick: False`, 21 October 2026, 08:56–11:16).

* `kaggle_v2_run/code/` — the exact code attached to that notebook run (unchanged).
* `revision_out/` — the outputs of that run, unchanged: `revision_results.json`, fold scores,
  E6/E7 tables and figures (`fig_cd_diagram.png` = thesis Figure 5.14).

The scripts in the repository root (v2.0) are the updated package prepared for the second review
round. They differ from `kaggle_v2_run/code/`; in particular, in the reported run XGBoost was tuned
with 20 candidates and the LSTM with 4, as stated in thesis Section 5.7.6, whereas root script 4
supports an equal tuning budget. To reproduce the reported numbers, use `kaggle_v2_run/code/`.

## Second-round outputs (release v2.2)
* `round2/` — unchanged outputs of `10_round2_analyses.py` (R1–R6), run from release v2.1 in the Kaggle notebook
  https://www.kaggle.com/code/ghaythasad/aiis-arp-round-2-analyses (version 1, 25 October 2026; 4-core CPU,
  Python 3.13.15, scikit-learn 1.6.1, XGBoost 3.4.1, SHAP 0.52.0, LIME 0.2.0.1), with the run log `round2_log.txt`.
* `prototype_demo/` — the reference prototype (`5_aiis_arp_service.py`) run on CICIoT2023 in the same notebook
  (25 October 2026): `train` on partitions 1–12, `detect` on partitions 13–15, then `feedback`, `status` and
  `rollback`. Contains every alert (`alerts.jsonl`, 4,174 alerts), the session log and the model registry
  (model files not included). The first alert is the example line in thesis Appendix D.

## Release v2.3 (third review round): complete output files and provenance

* `RESULTS_PROVENANCE.md` maps every table and figure of the thesis and the paper to the run and output file
  that produced it, and lists the values that were updated in the third revision round.
* New folders with the **unchanged** output files of the runs that produced the remaining reported results:
  * `original_run/` – outputs of the original 16-configuration benchmark (Kaggle notebook
    `notebookbb18ce43db`, **version 1**): per-configuration metrics and fold scores, McNemar inputs,
    adaptivity per-batch scores, SHAP/LIME timing, deletion test and the figures of thesis Chapter 5.
  * `supp_out/` – `supp_report.md`, `arp_vs_rest_report.md`, `S7b_deletion_all_folds.csv` and the CSV tables of
    scripts 6–8.
  * `1_iat_ablation_results.json` – output of `1_iat_ablation_and_timing.py` (thesis Table 5.5 / Appendix C).
* `3_statistics_checks.py`: comment corrected (`chi2.sf(74.47, 15)` = 7.1e-10).
* Hybrid and fusion models: all reported results use **in-sample** LSTM embeddings; no out-of-fold re-run has
  been reported. To run it: `EMB_MODE=oof SKIP=E0,E2,E3,E4,E5,E6,E7,E8 python 4_revision_experiments.py`
  (writes `oof_predictions.csv` and `E1_fold_f1_matrix.csv`; then `python 11_cd_diagram.py`).

Repository: https://github.com/ghayththelover-creator/AIIS-ARP

The original 16-configuration notebook: https://www.kaggle.com/code/ghaythasad/notebookbb18ce43db
