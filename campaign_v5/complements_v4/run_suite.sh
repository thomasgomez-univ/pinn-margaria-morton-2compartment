#!/usr/bin/env bash
# Deuxieme vague : inverse_crime_v3 et wr_sweep_v4 (apres rep_lm).  bash run_suite.sh [--discret|--equilibre|--max]
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
H=$((NPROC/2)); [ "$H" -lt 1 ] && H=1
for K in $(seq 0 $((H-1))); do nohup $WRAP python3 inverse_crime_v3.py $K $H > inverse_crime_$K.log 2>&1 & done
for K in $(seq 0 $((NPROC-H-1))); do nohup $WRAP python3 wr_sweep_v4.py $K $((NPROC-H)) > wr_sweep_$K.log 2>&1 & done
echo "lances : $H shards inverse_crime, $((NPROC-H)) shards wr_sweep. Suivi : tail -f *.log"
