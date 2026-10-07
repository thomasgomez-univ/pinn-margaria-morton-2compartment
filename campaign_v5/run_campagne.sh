#!/usr/bin/env bash
# Campagne v5 : calibration du bruit corrigee (M-1), population recalibree, intensites en fractions de la CP vraie.
# Reprend automatiquement la ou elle s'est arretee.
#
#   bash run_campagne.sh            equilibre : laisse 2 coeurs a l'interface (defaut)
#   bash run_campagne.sh --discret  coeurs d'efficacite seulement, machine intacte, plus lent
#   bash run_campagne.sh --max      tous les coeurs performance, machine peu reactive
#   bash run_campagne.sh -j 4       force 4 processus
#
# Arreter :  pkill -f pipeline_v3
# Relancer :  la meme commande. Ce qui est fait est saute.
set -u
cd "$(dirname "$0")"

MODE=equilibre
FORCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --discret|--discreet) MODE=discret ;;
    --max)                MODE=max ;;
    -j)                   FORCE="${2:-}"; shift ;;
    ''|*[!0-9]*)          echo "option inconnue : $1"; exit 1 ;;
    *)                    FORCE="$1" ;;
  esac
  shift
done

perf_cores() {
  local n
  n=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null) && [ -n "$n" ] && { echo "$n"; return; }
  n=$(sysctl -n hw.physicalcpu 2>/dev/null)            && [ -n "$n" ] && { echo "$n"; return; }
  n=$(nproc 2>/dev/null)                               && [ -n "$n" ] && { echo "$n"; return; }
  echo 2
}
P=$(perf_cores)

WRAP=""
case "$MODE" in
  discret)
    NPROC=$(( P / 2 )); [ "$NPROC" -lt 1 ] && NPROC=1
    if command -v taskpolicy >/dev/null 2>&1; then WRAP="taskpolicy -b nice -n 20"; else WRAP="nice -n 20"; fi ;;
  max)
    NPROC=$P ;;
  equilibre)
    NPROC=$(( P - 2 )); [ "$NPROC" -lt 1 ] && NPROC=1
    WRAP="nice -n 10" ;;
esac
[ -n "$FORCE" ] && NPROC="$FORCE"
[ "$NPROC" -gt 50 ] && NPROC=50

echo "Coeurs performance detectes : $P"
echo "Mode : $MODE  ->  $NPROC processus${WRAP:+, priorite abaissee}"
[ "$MODE" = max ] && echo "ATTENTION : la machine sera peu reactive pendant le calcul."
echo "150 executions au total."
if [ "$MODE" = discret ]; then
  echo "Estimation : $(( 150 * 10 / NPROC )) min (coeurs d'efficacite, 2 a 3 fois plus lents)."
else
  echo "Estimation : $(( 150 * 4 / NPROC )) min."
fi
echo

python3 - <<'FINPY' || { echo "ERREUR : dependances manquantes."; echo "  pip3 install numpy scipy torch"; exit 1; }
import importlib.util as u, sys
m = [x for x in ("numpy", "scipy", "torch") if u.find_spec(x) is None]
if m:
    print("modules absents :", ", ".join(m)); sys.exit(1)
print("dependances : OK")
FINPY
echo

BASE=$(( 50 / NPROC )); RESTE=$(( 50 % NPROC )); START=0
for k in $(seq 0 $((NPROC-1))); do
  N=$BASE; [ "$k" -lt "$RESTE" ] && N=$(( BASE + 1 ))
  [ "$N" -eq 0 ] && continue
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    nohup $WRAP python3 pipeline_v3.py \
      --athletes "$N" --start "$START" --sigma 5 2 10 --rtol 1e-8 \
      --out "camp_v3_$k.jsonl" > "camp_v3_$k.log" 2>&1 &
  echo "  processus $k : athletes $START a $(( START + N - 1 ))"
  START=$(( START + N ))
done

echo
echo "Avancement :  cat camp_v3_*.jsonl | wc -l      (150 = termine)"
echo "Detail     :  tail -f camp_v3_0.log"
echo "Arret      :  pkill -f pipeline_v3"
wait
echo "Termine. Renvoyez les fichiers camp_v3_*.jsonl"
