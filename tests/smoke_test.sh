#!/usr/bin/env bash
# Smoke test: runs every script and every documented prototype command on SYNTHETIC data in the
# CICIoT2023 layout.  It checks that the code executes end to end; the numbers are meaningless.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
mkdir -p "$WORK"
python "$HERE/tests/make_synthetic_ciciot.py" "$WORK/data" 32 1400
export DATA_GLOB="$WORK/data/part-*.csv" QUICK=1
cd "$WORK"
echo "== 1"; OUT_DIR="$WORK/out1" python "$HERE/1_iat_ablation_and_timing.py" > log1.txt
echo "== 3"; python "$HERE/3_statistics_checks.py" > log3.txt
echo "== 4"; OUT_DIR="$WORK/out4" N_TUNE=2 N_TUNE_LSTM=2 python "$HERE/4_revision_experiments.py" > log4.txt
echo "== 6"; OUT_DIR="$WORK/out6" python "$HERE/6_aiis_arp_supplementary.py" > log6.txt
echo "== 7"; OUT_DIR="$WORK/out7" python "$HERE/7_aiis_arp_vs_rest.py" > log7.txt
echo "== 8"; OUT_DIR="$WORK/out8" python "$HERE/8_deletion_check.py" > log8.txt
echo "== 9"; OUT_DIR="$WORK/out9" N_TUNE=2 python "$HERE/9_fast_e2_e5.py" > log9.txt
echo "== 10"; OUT_DIR="$WORK/out10" OOF_FILE="$WORK/out4/oof_predictions.csv" python "$HERE/10_round2_analyses.py" > log10.txt
echo "== 11"; OUT_DIR="$WORK/out11" python "$HERE/11_cd_diagram.py" "$WORK/out4/E1_fold_f1_matrix.csv" > log11.txt
echo "== 5 prototype"
python - "$WORK" <<'EOF'
import sys, pandas as pd, glob
w = sys.argv[1]; f = sorted(glob.glob(w + "/data/part-*.csv"))
pd.concat([pd.read_csv(x) for x in f[:3]]).to_csv(w + "/labelled_partitions_01_15.csv", index=False)
pd.read_csv(f[3]).drop(columns="label").to_csv(w + "/new_flows.csv", index=False)
pd.read_csv(f[4]).to_csv(w + "/analyst_batch_01.csv", index=False)
EOF
P="python $HERE/5_aiis_arp_service.py"
$P train --data labelled_partitions_01_15.csv
$P detect --flows new_flows.csv --top-k 5
$P feedback --labelled analyst_batch_01.csv
$P status > status.json
$P rollback
head -c 600 alerts.jsonl; echo
echo "SMOKE TEST PASSED ($WORK)"
