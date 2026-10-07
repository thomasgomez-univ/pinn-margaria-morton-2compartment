#!/usr/bin/env bash
# Campagne v5 complete (calibration du bruit corrigee, M-1). A lancer depuis campagne_v5/ avec bash :
#   nohup bash run_v5_all.sh --equilibre > run_v5_all.log 2>&1 &
# Enchaine : (1) campagne principale LM/DE/PINN soumis + baselines (camp_v3_K.jsonl) ; (2) PINN v4 (camp_v4_K.jsonl) ;
# (3) complements : FIM, protocoles, rep_lm, crime inverse, balayage w_r, mauvaise specification ; (4) prediction, P-D.
# Les noms de fichiers sont ceux de la v3 pour que les scripts d'analyse fonctionnent sans modification ;
# le repertoire campagne_v5/ les distingue. Reprise automatique de ce qui est deja fait.
set -u; cd "$(dirname "$0")"; MODE="${1:---equilibre}"
HERE=$(pwd)
waitfor() { sleep 20; while pgrep -f "$HERE.*python3|python3 .*(pipeline_v3|pipeline_v4|rep_lm_v3|fim_v3|protocoles_v3|inverse_crime_v3|wr_sweep_v4|misspec_v4)\.py" >/dev/null 2>&1; do sleep 60; done; }
echo "=== (1) campagne principale $(date) ===";  bash run_campagne.sh ${MODE/--equilibre/};  waitfor
echo "=== (2) PINN v4 $(date) ===";              bash run_v4.sh "$MODE";                      waitfor
cd complements_v4
echo "=== (3a) FIM, protocoles, rep_lm $(date) ===";   bash run_complements.sh "$MODE";     waitfor
echo "=== (3b) crime inverse, balayage w_r $(date) ==="; bash run_suite.sh "$MODE";         waitfor
echo "=== (3c) mauvaise specification $(date) ===";     bash run_misspec.sh "$MODE";        waitfor
echo "=== (3d) relation puissance-duree $(date) ===";   python3 pd_curve_v4.py > pd_curve.log 2>&1
cd ..
echo "=== (4) prediction $(date) ===";           python3 prediction_v4.py > prediction_v4.log 2>&1
echo "TERMINE campagne v5 $(date) : $(cat camp_v3_*.jsonl | wc -l) runs principaux, $(cat camp_v4_*.jsonl | wc -l) runs PINN v4"
