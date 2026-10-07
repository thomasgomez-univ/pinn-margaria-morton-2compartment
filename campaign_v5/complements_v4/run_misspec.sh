#!/usr/bin/env bash
# Controle de mauvaise specification (AR1 + DRIFT), 20 athletes, sigma_P = 5 W.
# bash run_misspec.sh [--discret|--equilibre|--max]   -- suivi : tail -f misspec_*.log
set -u; cd "$(dirname "$0")"; MODE="${1:---equilibre}"
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc); E=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)
case "$MODE" in
  --discret)   NPROC=$E; [ "$NPROC" -lt 1 ] && NPROC=2; WRAP="taskpolicy -b nice -n 20" ;;
  --equilibre) NPROC=$((P-2)); [ "$NPROC" -lt 1 ] && NPROC=1; WRAP="nice -n 10" ;;
  --max)       NPROC=$P; WRAP="" ;;
  *) echo "mode inconnu : $MODE"; exit 1 ;;
esac
command -v taskpolicy >/dev/null 2>&1 || WRAP="${WRAP/taskpolicy -b /}"
H=$((NPROC/2)); [ "$H" -lt 1 ] && H=1; R=$((NPROC-H)); [ "$R" -lt 1 ] && R=1
for K in $(seq 0 $((H-1))); do nohup $WRAP python3 misspec_v4.py AR1   $K $H > misspec_AR1_$K.log   2>&1 & done
for K in $(seq 0 $((R-1))); do nohup $WRAP python3 misspec_v4.py DRIFT $K $R > misspec_DRIFT_$K.log 2>&1 & done
echo "lances : $H shards AR1, $R shards DRIFT (20 athletes chacun, LM + DE + PINN v4). Suivi : tail -f misspec_*.log"
