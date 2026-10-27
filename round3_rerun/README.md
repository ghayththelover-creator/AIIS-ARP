# Round-3 re-runs (third review round, 27 October 2026)

All runs used the study subset `arp_15parts.csv.gz` (Benign + MITM-ArpSpoofing records of `part-00000` …
`part-00014`; 106,667 records, written by `kaggle_v2_run/code/0_prepare_subset.py`), split back into one CSV per
partition, on a two-core Intel Xeon CPU without GPU (Python 3.13.17, NumPy 2.5.3, pandas 3.0.5,
scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.54.1, TensorFlow 2.21.0; see `benchmark/environment.json` and
`ablation/1_iat_ablation_results.json`). Folds (GroupKFold(5) on the partition), targets (106,382 records with
≥ 19 predecessors) and seeds are the same as in all earlier runs.

## `benchmark/` — all 16 configurations, hybrid/fusion with in-sample **and** out-of-fold second stages

* `oof_rerun.py` reuses the data loading, pre-processing, window construction and model code of
  `4_revision_experiments.py` (v2.0) and runs E1 only. For every outer fold and LSTM seed (0, 1, 2) the same
  LSTM is used to build two second stages: one fitted on in-sample embeddings (as in the thesis and runs A/B)
  and one fitted on inner out-of-fold embeddings (`oof_embeddings`, three partition-disjoint inner folds).
  The run was executed in resumable steps (state checkpointed after every fold and seed; see `run_log.txt`).
* `fold_scores.csv` – metrics per configuration and fold; `oof_predictions.csv.gz` – out-of-fold
  probabilities of all 20 configurations for every target (`target_id` = partition_row);
  `fold_f1_matrix_friedman_*.csv` – fold × configuration F1 matrices.
* `analyze.py`, `boot_extra.py` – summaries, Friedman/Nemenyi, McNemar (Bonferroni, m = 19) and the
  partition-level cluster bootstrap (2,000 resamples of the 15 partitions) → `round3_rerun_results.json`
  (thesis Table 5.2, paper Table 4). `fig56.py` → `fig_discordant_round3.png` (thesis Figure 5.6).

Main results (F1, pooled): AIIS-ARP (XGB, T = 1) 0.9285; fusion LSTM–XGB OOF 0.9293 (T = 10) / 0.9282 (T = 20),
in-sample 0.9279 / 0.9284; hybrid LSTM–RF OOF 0.8106 / 0.8126, in-sample 0.8158 / 0.8168; LSTM 0.8103 / 0.8102.
AIIS-ARP minus fusion (OOF, T = 10): ΔF1 = −0.0008, 95% CI −0.0019 to 0.0004, McNemar p = 0.40.
Friedman (OOF set): χ² = 73.64, p = 1.0e-9.

## `ablation/` — `1_iat_ablation_and_timing.py`
Confusion matrices identical to thesis Table 5.5 / Appendix C. The fold SD of F1 is now computed with
ddof = 1 (0.0022 / 0.0040 / 0.0055); the original run used ddof = 0 (0.0019 / 0.0036 / 0.0049). Timings differ
because the hardware differs.

## `supp/`, `deletion/` — `6_aiis_arp_supplementary.py`, `8_deletion_check.py`
Re-run on the benign/ARP subset, so S8 (other attack classes) is skipped; the within-study negative control
(31.7 %) is reproduced by `round2/` (R4). Reproduced exactly: Table 5.12 (S3), Table 5.14 (S4, all 48 values),
the IAT stability values (S6), the alternative-baseline deletion ranges and SHAP additivity (S7) and Table 5.15
(`deletion/S7b_deletion_all_folds.csv`). Random Forest and decision tree
outputs differ slightly from the original run under scikit-learn 1.9.1 (S5: ΔF1 vs RF 0.0063 instead of
0.0066); the reported Table 5.13 is taken from `round2/R3_cluster_bootstrap.csv`, computed with the original
library versions.

Not reproducible here: the outputs of the original Kaggle notebook (run A, version 1) were not archived. Its
record-level predictions are reproduced exactly by `round2/` and by `supp/S3`, and all 16 configurations are
re-run and archived in `benchmark/`.
