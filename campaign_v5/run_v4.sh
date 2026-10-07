#!/usr/bin/env bash
# Campagne v4 : PINN variante H2 seul, sur les 150 runs de la campagne v3.
# A lancer avec bash (pas zsh) depuis campagne_v3/ :
#   bash run_v4.sh --discret     # coeurs efficience, machine utilisable   (~1 h)
#   bash run_v4.sh --equilibre   # P-2 coeurs, nice 10                      (~40 min)
#   bash run_v4.sh --max         # tous les coeurs                          (~30 min)
#   bash run_v4.sh --graines     # 10 athletes x 3 graines a sigma=5 (variance)
# Reprend automatiquement la ou il s'est arrete. Resultats : camp_v4_K.jsonl
set -u
cd "$(dirname "$0")"
MODE="${1:---equilibre}"
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)
E=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)
EXTRA=""
case "$MODE" in
  --discret)   NPROC=$E; [ "$NPROC" -lt 1 ] && NPROC=2; WRAP="taskpolicy -b nice -n 20" ;;
  --equilibre) NPROC=$((P-2)); [ "$NPROC" -lt 1 ] && NPROC=1; WRAP="nice -n 10" ;;
  --max)       NPROC=$P; WRAP="" ;;
  --graines)   NPROC=$((P-2)); [ "$NPROC" -lt 1 ] && NPROC=1; WRAP="nice -n 10"
               EXTRA="--seeds 3 --sigma 5 --athletes 0-9"; TAG="g" ;;
  *) echo "mode inconnu : $MODE"; exit 1 ;;
esac
command -v taskpolicy >/dev/null 2>&1 || WRAP="${WRAP/taskpolicy -b /}"
python3 - <<'PY' || { echo "dependances manquantes : pip3 install numpy scipy torch"; exit 1; }
import numpy, scipy, torch
PY
ls camp_v3_*.jsonl >/dev/null 2>&1 || { echo "camp_v3_*.jsonl introuvables ici"; exit 1; }
echo "mode $MODE : $NPROC processus ($WRAP)"
for K in $(seq 0 $((NPROC-1))); do
  OUT="camp_v4_${TAG:-}${K}.jsonl"
  nohup $WRAP python3 pipeline_v4.py --shard $K --nshards $NPROC $EXTRA --out "$OUT" \
      > "camp_v4_${TAG:-}${K}.log" 2>&1 &
done
echo "lance. Suivi :  tail -f camp_v4_*.log     Avancement :  cat camp_v4_*.jsonl | wc -l"
