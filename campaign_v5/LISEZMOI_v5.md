# Campagne v5 — calibration du bruit corrigée (point M-1 de l'expertise)

## Pourquoi
Le contrôle M-1 a montré que `calibrate_sigma_AP` de la campagne v3 produisait, pour le protocole intermittent B,
un écart-type σ_AP de ~240 J **à σ_P = 0** : la puissance bruitée était échantillonnée à 1 s puis interpolée
linéairement, ce qui transformait chaque commutation en rampe de 1 s (déviation systématique ~240 J, athlète 0),
et les points de grille postérieurs à l'épuisement d'une réalisation bruitée (trajectoire tronquée) entraient dans
le RMS (athlète 1 : 721 J). Résultat : σ_AP(B) = 242 / 270 / 348 J à σ_P = 2 / 5 / 10 W au lieu de ~42 / 105 / 210 J.
Les données de B étaient donc 2,5× plus bruitées que l'équivalent capteur à 5 W (6× à 2 W), les bornes et les
tableaux à faible bruit en dépendent, et la comparaison de protocoles (S3) pénalisait les designs intermittents.

## Correctif (pipeline_v3.py, complements_v4/pipeline_v3.py, complements_v4/protocoles_v3.py)
Bruit ajouté à la puissance exacte (commutations conservées) ; déviation mesurée sur les points de grille antérieurs
à l'épuisement le plus précoce des réalisations (grid < 0,98·min t_end). Vérification (6 athlètes, 5 W) :
A 103→94, 98→94, 95→102, 79→81, 126→144, 107→115 J ; B 264→105, 721→112, 269→106, 267→86, 235→157, 262→117 J.

## Lancement (depuis ce répertoire, environnement avec torch)
    nohup bash run_v5_all.sh --equilibre > run_v5_all.log 2>&1 &
    tail -f run_v5_all.log
Enchaîne campagne principale → PINN v4 → compléments (FIM, protocoles, rep_lm, crime inverse, w_r, mauvaise
spécification, P–D) → prédiction. Les fichiers gardent les noms v3/v4 (camp_v3_K.jsonl, camp_v4_K.jsonl) pour que
les scripts d'analyse (`tables_v4.py`, `make_figs_v4.py`, `analyse_misspec.py`, …) tournent sans modification.
Durée estimée : 3–5 h en mode --equilibre.

## Ensuite
`python3 tables_v4.py && python3 make_figs_v4.py` puis mise à jour des chiffres du manuscrit à partir de
`numbers_v4.json` (v3 → v5), des tableaux S3, S5–S6 et de la Section S6.

## Exploitation (7 octobre 2026, apres TERMINE campagne v5 a 15 h 04)

- `tables_v4.py` et `make_figs_v4.py` corriges (cles `fit`/`vfit` de `flat_traj_v3.npz` au lieu de `fitx`/`expo` ; sauvegardes `.bak_v5`) ; figures regenerees dans `../figures/`.
- `complements_v4/analyse_misspec.py`, `complements_v4/analyse_protocoles_v5.py` executes ; nouveaux scripts :
  - `cos_align_v5.py` : cosinus entre le vecteur d'erreur log et la direction la moins contrainte de la FIM (sigma_P = 5 W) -> `cos_align_v5.json` ;
  - `complements_v4/misspec_extra_v5.py` : misfit (mediane, IQR, fraction > 1.1), Wilcoxon apparies, erreurs signees, correlation des erreurs log ;
  - `complements_v4/analyse_protocoles_v5b.py` : rapports par athlete a A+B (kappa, lambda_min, volume, bornes) avec Wilcoxon -> `analyse_protocoles_v5b.log`.
- Chiffres integres dans `manuscript_v3_short.tex` et `supplement_v3.tex` par `edit_v5_part{1,2,3}.py` (sauvegardes `*.bak_before_v5`) ; `PeerJ_submission/` reconstruit (manuscript, Tables 1-5, Figures 2-4, Supplemental Article S1).
- Non recalcules (hors campagne) : diagnostic du reseau sur l'athlete 0 (Table S2, donnees d'avant recalibration ; note ajoutee en S2), variabilite entre graines d'entrainement (10 athletes), analyse VO2 + T_lim (Table 2 ; note ajoutee dans les Limites).

## vo2_tlim_v5.py (7 octobre, soir) — regeneration de l'analyse VO2 + T_lim perdue
`complements_v4/vo2_tlim_v5.py` -> `vo2_tlim_v5.json`, `vo2_tlim_v5.log`. 10 athletes, protocoles A et B, grille de la campagne (60 points),
DOP853 1e-10, sensibilites relatives par differences centrees (1e-4), T_lim par derivee implicite (validee a 3,9e-7 ; l'athlete 4
s'epuise sur une commutation du protocole B : T_lim y est discontinu, derivee unilaterale conservee). Blocs normalises (c_V = 1, c_T = r)
et bornes absolues (sigma_V = 100 / 200 mL/min a 20,9 J/mL, cv_T = 3 / 10 %). Differences avec l'ancienne analyse (vo2_tlim.md,
rapport3_reponses.md) : grille de 60 points au lieu de 20 pour le bloc normalise ; A_P avec sigma_AP v5 au lieu de 300 J constant ;
projections toutes en coordonnees logarithmiques. Integre dans Table 2 (bloc inferieur), §2.2, §3.1, abstract, §4.2, Limites
(`edit_v5_tlim.py`, sauvegarde `manuscript_v3_short.tex.bak_before_tlim_v5`) et dans les diapositives 11 et 19.

## fim_vo2_single_v5.py — bloc superieur de Table 2 (partie Fisher) refait sur la population recalibree
Athlete 0, protocoles A et B, grille de la campagne, sigma_P = 5 W, FIM 5 x 5 : A_P lambda_min/lambda_max 2,0e-6, projections CP 0,007 / W' 0,002 ;
VO2 (phi1) 4,0e-18, 0,460 / 0,500 ; VO2 (recharge) 1,3e-17, 0,460 / 0,500. Les rangs structurels (FISPO, ordre 8) sont inchanges :
ils ne dependent pas de la population. `vo2_tlim_v5.py` a ete place derriere `if __name__ == "__main__"` pour etre importable.
Integre dans Table 2, §3.1 (« twelve orders of magnitude », 0,460 / 0,500) et la diapositive 11. Point M-16 de l'expertise clos.

## Second tour d'expertise (7 octobre, soir) — complements de calcul
Faits en direct :
- `complements_v4/fim_vo2_single_v5.py` : boucle sur 10 athletes (M-22) ; `vo2_tlim_v5.json` relu sans l'athlete 4 (epuisement sur une
  commutation du protocole B, M-24) -> Table 2 en deux blocs (rang structurel ; Fisher 5 x 5, mediane sur neuf athletes), §2.2, §3.1.
- M-4 : medianes du reseau corrige sur les 35 athletes hors selection (15-49) : CP 1,25 / 1,25 / 1,47 %, W' 2,47 / 2,21 / 2,44 % ->
  §2.5 (declaration de la selection) et §3.3.
- `complements_v4/vo2_kinetics_v5.py` : cinetique de phi1 sur la population recalibree (tau mono-exp a 80 % CP : 53 s [33-81] ;
  1/M_R 51 s [18-89] ; r = 0,00 ; figure `fig_vo2_kinetics_v5.pdf`) -> Limites, nouvelle Section S7 + Figure S3 du supplement (P-2, P-4).
A lancer sur le Mac (`nohup bash run_rev2.sh > run_rev2.log 2>&1 &` depuis `campagne_v5/`, ~3 h) :
- (1) `pipeline_v4.py --seeds 3 --sigma 5 --out camp_v4_seeds_K.jsonl` : graines 1-2 du reseau corrige, 50 athletes (M-19, M-14) ;
  synthese `complements_v4/pinn_seeds_v5.py`.
- (2) `complements_v4/de_seeds_v5.py` : graines 1-2 de DE, 20 athletes (M-19, M-11) ; synthese `--analyse`.
- (3) `complements_v4/gls_ar1_v5.py` : GLS sous AR(1) (residus blanchis, F = S^T Sigma^-1 S), 20 athletes (M-10) ; synthese `--analyse`.
- (4) `complements_v4/profile_v5.py` : profils de vraisemblance de A_Omax et M_R, athletes 0-2 (M-2) -> `profile_v5.json`, `fig_profile_v5.pdf`.

## Resultats de run_rev2.sh (TERMINE 18 h 56) et integration
- `profile_v5.py` lance en 3 processus ecrivait le meme profile_v5.json (seul l'athlete 1 restait) et la figure echouait (format) :
  `profile_rebuild_v5.py` reconstruit profile_v5.json depuis profile_{0,1,2}.log et trace `fig_profile_v5.pdf`. Profils quadratiques
  sur +/-4 sigma_CR : demi-largeur 95 % = 0,99-1,03 x 1,96 sigma_CR, symetriques, vraie valeur dedans (6/6).
- `gls_ar1_v5.py --analyse` : GLS efficacite 0,85-1,23, couverture 85-95 % ; borne GLS ~2x bruit blanc ; OLS efficace a 1,09-1,44 contre elle.
- `pinn_seeds_v5.py` : medianes CP 1,36/1,24/1,38 % ; e.-t. par athlete 0,27 pt (max 1,86) ; W' 2,75/2,61/2,11.
- `de_seeds_v5.py --analyse` : medianes CP 0,99/0,70/0,84 % (20 athletes) ; e.-t. par athlete 0,61 pt (max 7,51).
- Integre par `edit_rev2_runs.py` (sauvegardes *.bak_before_rev2c). Tables PeerJ autonomes : \ref et \cite remplaces en dur.
