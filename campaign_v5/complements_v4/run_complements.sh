#!/usr/bin/env bash
# Complements v4 (population recalibree). A lancer avec bash depuis complements_v4/ :
#   bash run_complements.sh [--discret|--equilibre|--max]
# Phase 1 (avant-plan, ~35 min sur 1 coeur) : fim_v3.py, protocoles_v3.py
# Phase 2 (arriere-plan, shards)            : rep_lm_v3, inverse_crime_v3, wr_sweep_v4
set -u
cd "$(dirname "$0")"
MODE="${1:---equilibre}"
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc); E=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)
case "$MODE" in
  --discret)   NPROC=$E; [ "$NPROC" -lt 1 ] && NPROC=2; WRAP="taskpolicy -b nice -n 20" ;;
  --equilibre) NPROC=$((P-2)); [ "$NPROC" -lt 1 ] && NPROC=1; WRAP="nice -n 10" ;;
  --max)       NPROC=$P; WRAP="" ;;
  *) echo "mode inconnu : $MODE"; exit 1 ;;
esac
command -v taskpolicy >/dev/null 2>&1 || WRAP="${WRAP/taskpolicy -b /}"
python3 -c "import numpy, scipy, torch" || { echo "dependances manquantes : pip3 install numpy scipy torch"; exit 1; }
{ ls camp_v3_*.jsonl >/dev/null 2>&1 || ls ../camp_v3_*.jsonl >/dev/null 2>&1; } || { echo "camp_v3_*.jsonl introuvables (ici ou dans ../)"; exit 1; }
echo "mode $MODE : $NPROC processus"
# Phase 2 en arriere-plan d'abord (longue), phase 1 en avant-plan
for K in $(seq 0 $((NPROC-1))); do
  nohup $WRAP python3 rep_lm_v3.py $K $NPROC       > rep_lm_$K.log 2>&1 &
done
nohup $WRAP bash -c "python3 fim_v3.py > fim_v3.log 2>&1; python3 protocoles_v3.py 15 > protocoles_v3.log 2>&1" &
echo "phase 1 + rep_lm lances. Quand rep_lm_*.log affichent TERMINE, lancez :  bash run_suite.sh $MODE"
