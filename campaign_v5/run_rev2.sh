#!/bin/bash
# Second tour d'expertise : complements de calcul. A lancer depuis campagne_v5/ :  nohup bash run_rev2.sh > run_rev2.log 2>&1 &
# (1) graines du reseau corrige (seed_k 1 et 2, 50 athletes, sigma = 5 W) ; (2) graines de DE (20 athletes) ; (3) GLS sous AR(1) ; (4) profils de vraisemblance.
cd "$(dirname "$0")"; HERE=$(pwd)
echo "=== (1) PINN v4, graines 1-2, sigma 5 W : 6 shards  $(date) ==="
# les fichiers de sortie sont separes (camp_v4_seeds_K.jsonl) pour ne pas modifier camp_v4_K.jsonl lus par tables_v4/make_figs ;
# ils sont pre-remplis avec les runs seed_k = 0 de la campagne (sigma 5 W) pour que pipeline_v4 ne les recalcule pas.
python3 - <<'PY'
import json, glob
rec0 = [l for f in sorted(glob.glob("camp_v4_[0-9].jsonl")) for l in open(f) if json.loads(l)["sigma_P"] == 5.0 and json.loads(l).get("seed_k", 0) == 0]
for k in range(6):
    open("camp_v4_seeds_%d.jsonl" % k, "a").writelines(rec0)
print("pre-rempli :", len(rec0), "runs seed 0 par shard")
PY
for k in 0 1 2 3 4 5; do nohup nice -n 10 python3 pipeline_v4.py --shard $k --nshards 6 --seeds 3 --sigma 5 --out camp_v4_seeds_$k.jsonl > camp_v4_seeds_$k.log 2>&1 & done
cd complements_v4
echo "=== (2) DE graines 1-2 : 3 shards  $(date) ==="
for k in 0 1 2; do nohup nice -n 10 python3 de_seeds_v5.py $k 3 > de_seeds_$k.log 2>&1 & done
wait
echo "=== (3) GLS AR(1) : 3 shards  $(date) ==="
for k in 0 1 2; do nohup nice -n 10 python3 gls_ar1_v5.py $k 3 > gls_ar1_$k.log 2>&1 & done
echo "=== (4) profils de vraisemblance, athletes 0 1 2  $(date) ==="
nohup nice -n 10 python3 profile_v5.py 0 > profile_0.log 2>&1 &
nohup nice -n 10 python3 profile_v5.py 1 > profile_1.log 2>&1 &
nohup nice -n 10 python3 profile_v5.py 2 > profile_2.log 2>&1 &
wait
echo "=== syntheses  $(date) ==="
python3 de_seeds_v5.py --analyse > de_seeds_v5.log 2>&1
python3 gls_ar1_v5.py --analyse > gls_ar1_v5.log 2>&1
python3 pinn_seeds_v5.py > pinn_seeds_v5.log 2>&1
echo "TERMINE run_rev2 $(date)"
