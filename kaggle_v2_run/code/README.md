# AIIS-ARP — Replication Code

Code for the MSc thesis *An Adaptive and Interpretable Intelligent System for Early Detection of
ARP Spoofing and Man-in-the-Middle Attacks* (Northern Border University). Data: CICIoT2023.

| Script | Purpose | Thesis |
|---|---|---|
| `0_prepare_subset.py` | Builds `arp_15parts.csv.gz` (Benign + ARP spoofing, first 15 partitions, 106,667 records) | Section 3.4.2 |
| `1_iat_ablation_and_timing.py` | IAT ablation and inference timing | Table 5.5, Table C.1, Section 5.3.2 |
| `3_statistics_checks.py` | MCC, Wilson CIs, Friedman, McNemar + Bonferroni, exact Wilcoxon | Chapter 5 |
| `4_revision_experiments.py` | Additional validity analyses E0–E8 (Kaggle, a few hours on GPU T4 x2) | Section 5.7, Figures 5.14–5.19 |
| `5_aiis_arp_service.py` | Reference prototype: detect + explain + guarded retraining + rollback | Section 4.5.1, Appendix D |
| `6_iat_mode_analysis.py` | IAT artefact analysis (two value ranges, masked IAT) | Section 5.7.4, Table 5.8, Figures 5.16–5.17 |

`2_make_figure1.py` (architecture diagram, Figure 4.1) belongs to the student's original files and is not included here.

## Requirements
Python 3.10+, pandas, numpy, scipy, scikit-learn, xgboost, shap, lime, tensorflow (script 4 only), matplotlib, joblib.

## Running
```bash
# on Kaggle with the CICIoT2023 dataset attached
python 0_prepare_subset.py                       # -> arp_15parts.csv.gz
python 1_iat_ablation_and_timing.py arp_15parts.csv.gz
python 3_statistics_checks.py
python 4_revision_experiments.py                 # -> revision_out/ (JSON, CSV, figures)
python 6_iat_mode_analysis.py arp_15parts.csv.gz revision_out/revision_results.json

# prototype
python 5_aiis_arp_service.py train    --data labelled.csv
python 5_aiis_arp_service.py detect   --flows new_flows.csv --top-k 5
python 5_aiis_arp_service.py feedback --labelled analyst_batch.csv
python 5_aiis_arp_service.py status | rollback
```
Options for script 4: `DATA_GLOB`, `OUT_DIR`, `QUICK=1` (smoke test), `SKIP=E5,E8`.
