# Compléments v4 — population recalibrée

Dossier à placer dans `campagne_v3/complements_v4/` (les `camp_v3_*.jsonl` sont lus dans `../`).

```bash
cd ~/Documents/Pro/Recherche/MyPapers/ArticlePINNAI/Work_Claude/Revision_v2/campagne_v3/complements_v4
bash run_complements.sh --equilibre     # phase 1 (FIM 50 athlètes, 17 protocoles) + LM 8×20 réalisations
# quand rep_lm_*.log affichent TERMINE (~30 min) :
bash run_suite.sh --equilibre           # crime inverse (15 athlètes, LM/DE/PINN v4) + balayage w_r du PINN v4
```

| script | produit | usage dans l'article |
|---|---|---|
| `fim_v3.py` | `fim_v3.json`, `fim_v3.npz`, `flat_traj_v3.npz` | § Practical identifiability, Figure identifiabilité |
| `protocoles_v3.py` | `protocoles_v3.jsonl` | Tableau protocoles, § Protocol design |
| `rep_lm_v3.py` | `rep_lm_v3_K.jsonl` | Tableau per-parameter : lignes « 20 réalisations × 8 athlètes » |
| `inverse_crime_v3.py` | `inverse_crime_v3_K.jsonl` | Methods, contrôle du crime inverse |
| `wr_sweep_v4.py` | `wr_sweep_v4_K.jsonl` | § Estimation accuracy, balayage w_r (PINN v4) |

Durées indicatives sur 1 cœur : fim 2 min, protocoles 33 min, rep_lm 160 × 65 s ≈ 2 h 50,
inverse_crime 15 × 165 s ≈ 40 min, wr_sweep 75 × 40 s ≈ 50 min. Tout est parallélisé
et reprend là où il s'est arrêté.

## Controle de mauvaise specification (06/10, `misspec_v4.py`)
Donnees generees avec un ecart que le modele ajuste ignore, 20 athletes, sigma_P = 5 W, memes graines :
- `AR1` : bruit d'observation AR(1), phi = 0,8 par pas de grille, meme sigma_AP marginal ;
- `DRIFT` : eta(t) = eta0 (1 - 0,05 t / T0), T0 = temps d'epuisement nominal (eta perd 5 % a l'horizon de l'essai).
LM (avec SE), DE et PINN v4 ajustes avec le modele exact de la campagne. Sorties `misspec_v4_<CASE>_<shard>.jsonl`
(est, LM_se, crlb_pct bruit blanc, rmse ajuste / a theta_true, sigma_AP).
Lancement : `bash run_misspec.sh [--discret|--equilibre|--max]` ; suivi `tail -f misspec_*.log` ;
synthese : `python3 analyse_misspec.py` -> `misspec_v4_summary.json`, `misspec_v4_table.tex`
(colonnes : CP, W', erreurs par parametre, efficacite vs CRLB bruit blanc, couverture LM, ratio de misfit).
Test unitaire (athlete 0, LM seul) : AR1 CP 0,56 % ; DRIFT CP 3,97 %, A_Omax -52 %, M_R +41 % (compensation le long
de la direction plate), misfit 1,15 x plancher.

## Relation puissance-duree (P-1 de l'expertise, `pd_curve_v4.py`, 06/10)
50 athletes : T_lim du modele a 102/105/110/125/150/200 % de CP_m (medianes 340/253/189/117/76/46 s) contre 3053/1221/611/244/122/61 s pour
l'hyperbole de memes CP et W' ; (P-CP) T_lim de 2,1 a 14,3 kJ ; ajustement a deux parametres sur 105-150 % : CP_test/CP_m 0,80-0,88 (med 0,83),
W'_test/W'_m 0,78-0,84 (med 0,81). Sorties `pd_curve_v4.json`, `pd_curve_v4_summary.json`, `figures/fig_pd_curve.pdf` (Figure S2, Section S6).

## Revision rev.3 (09/10, scripts `rev3_*.py`)
Analyses demandees par la relecture ; estimateur renomme « trust-region reflective » (TRF, scipy `method="trf"`), le label `LM` des fichiers est conserve.
- `rev3_cp_pd.py` : bornes de Cramer-Rao sur CP_PD et W'_PD (regression P = CP + W'/T a 105/110/125/150 % CP), 50 athletes ; `rev3_cp_pd_0.jsonl`.
- `rev3_sigma_fim.py K 3 --n 20 --mc 400` : information de Fisher avec la covariance Sigma des deviations de A_P (400 tirages Monte-Carlo du bruit de puissance), retrait vers la diagonale (alpha) et pepite (nu) ; `rev3_sigma_fim_K.jsonl`.
- `rev3_tlim_div.py` : equilibres a puissance constante, valeurs propres du jacobien, divergence logarithmique de T_lim pres de CP ; `rev3_tlim_div_0.jsonl`.
- `rev3_censoring.py` : censure a 3600 s dans les predictions de la Table 5 ; `rev3_censoring.json`.
- `rev3_binom.py` : intervalles de Clopper-Pearson sur les couvertures ; `rev3_binom.json`.
- `misspec_v4.py AR1 --phi 0.3|0.6|0.95 --tag _phiX --methods LM --n 10` et `DRIFT --delta 0.025|0.10 --tag _d0025|_d010` : balayages ; `rev3_fgls.py --phi X --tag _phiX [--tagols _phiX]` : moindres carres generalises faisables (phi estime sur les residus, Cochrane-Orcutt itere) ; `rev3_fgls_0_phiX.jsonl`.
- `rev3_population.py seed8|seed9|rho05|tau_wide --fit` : populations alternatives de 20 athletes (generateur `../make_population_v5.py`), Fisher sur 20, TRF sur 10 ; `rev3_pop_<pop>_0.jsonl`.
Toutes les syntheses par `--analyse`. Logs cloud (numpy 2.2.6, scipy 1.15.3) joints.

## Revision rev.3 bis (09/10, second rapport)
- `rev3_eiv.py 0 2 --n 10` et `1 2 --n 10` : erreurs dans les variables, A_P exact sous la puissance vraie, estimateur alimente par la puissance enregistree (bruit blanc 5 W aux noeuds de 1 s, interpole) ; pas de difference finie 1e-5 et rtol 1e-10 (sinon le jacobien numerique est du bruit et l'iteration s'arrete loin du minimum) ; `rev3_eiv_K.jsonl`, synthese `--analyse` (Table S8).
- `rev3_chi2.py` : test chi2 (115 ddl) du residu pondere sur la campagne et sur chaque ecart au modele ; `rev3_chi2.json`, `rev3_chi2_null.json`.
- `rev3_population.py eta_low --fit` : cinquieme population, eta ~ N(0.21, 0.02) (options `eta_mu`, `eta_sd` de `make_population_v5.py`) ; `rev3_pop_eta_low_0.jsonl` (Table S11).
- `make_fig_summary.py` : Figure 3 du manuscrit (bornes sous les trois observables) a partir de `camp_v3_*.jsonl` et `rev3_sigma_fim_*.jsonl`, les bornes VO2 + T_lim et CP, W', flux, CP_PD, W'_PD sous bruit blanc etant les valeurs du manuscrit codees en dur.
Les calculs ont tourne dans le conteneur cloud (numpy 2.2.6, scipy 1.15.3) ; logs joints.
