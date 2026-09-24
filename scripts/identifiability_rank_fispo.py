#!/usr/bin/env python3
"""
Identifiabilite structurelle LOCALE du modele bioenergetique a deux compartiments
(Gomez, manuscrit CMBBE, Eqs. 1-5), par le critere de rang de la matrice
d'observabilite-identifiabilite generalisee (approche FISPO, cf. Villaverde,
Barreiro & Papachristodoulou 2016 ; implementee dans STRIKE-GOLDD).

SYSTEME
    dA_O/dt = M_R (A_Omax - A_O) - M_O (A_O/A_Omax)(1 - A_P/A_Pmax)
    dA_P/dt =                      M_O (A_O/A_Omax)(1 - A_P/A_Pmax) - u/eta
    y       = A_P
    u(t)    = P_ext(t), entree connue

INCONNUES : etats (A_O, A_P) + parametres (M_O, A_Omax, A_Pmax, M_R, eta) -> n = 7

CRITERE
    rang(O) = n  => systeme localement identifiable et observable
    rang(O) < n  => non identifiable ; le noyau donne les directions plates

NOTE DE RIGUEUR
    Le rang est evalue en un point rationnel tire au hasard. Le rang en un point
    est une BORNE INFERIEURE du rang generique ; obtenir n en un point demontre
    donc rang generique = n. L'inverse n'est pas vrai : un rang < n en un point
    doit etre confirme sur plusieurs points, et idealement par un calcul
    symbolique du noyau.

    Ce critere donne l'identifiabilite LOCALE. Pour l'identifiabilite GLOBALE
    (unicite et non seulement isolement des solutions), utiliser
    StructuralIdentifiability.jl ou SIAN (voir le script Julia joint).

Dependance : sympy
"""

import random
import sympy as sp
from sympy import Rational as R

# ----------------------------------------------------------------- symboles
AO, AP = sp.symbols("A_O A_P", positive=True)
MO, AOm, APm, MR, eta = sp.symbols("M_O A_Omax A_Pmax M_R eta", positive=True)

NDER = 9                                   # u, u', u'', ... traitees en symboles
U = sp.symbols("u0:%d" % NDER)

STATES = [AO, AP]
PARAMS = [MO, AOm, APm, MR, eta]
UNKNOWNS = STATES + PARAMS
N = len(UNKNOWNS)


def build_field(n_free_input_derivatives):
    """Champ de vecteurs, avec les derivees de u au-dela de l'ordre donne mises a 0.

    n_free_input_derivatives = NDER-1 : entree generique (persistante excitante)
    n_free_input_derivatives = 1      : entree affine (rampe)
    n_free_input_derivatives = 0      : entree CONSTANTE (protocole a puissance fixe)
    """
    zeros = {U[j]: 0 for j in range(n_free_input_derivatives + 1, NDER)}
    phi1 = MO * (AO / AOm) * (1 - AP / APm)
    f = [MR * (AOm - AO) - phi1, phi1 - U[0] / eta]
    return [sp.simplify(fi.subs(zeros)) for fi in f]


def observability_matrix(f, n_free):
    """Matrice des jacobiens de h, L_f h, ..., L_f^{N-1} h par rapport aux inconnues."""

    def lie(expr):
        out = sum(sp.diff(expr, s) * fi for s, fi in zip(STATES, f))
        for j in range(NDER - 1):
            out += sp.diff(expr, U[j]) * (U[j + 1] if j + 1 <= n_free else 0)
        return sp.together(out)

    rows, cur = [], AP                      # h = A_P
    for k in range(N):
        rows.append([sp.diff(cur, v) for v in UNKNOWNS])
        if k < N - 1:
            cur = lie(cur)
    return sp.Matrix(rows)


def generic_rank(O, n_free, trials=5, seed=0):
    """Rang maximal obtenu sur `trials` points rationnels aleatoires."""
    random.seed(seed)
    best, best_M = 0, None
    for _ in range(trials):
        sub = {v: R(random.randint(2, 97), random.randint(1, 13)) for v in UNKNOWNS}
        for j in range(NDER):
            sub[U[j]] = (R(random.randint(-97, 97), random.randint(1, 13))
                         if j <= n_free else 0)
        M = O.subs(sub)
        rk = M.rank()
        if rk > best:
            best, best_M = rk, M
    return best, best_M


def main():
    cases = [("entree generique (u, u', u'', ... libres)", NDER - 1),
             ("entree affine    (rampe de puissance)    ", 1),
             ("entree CONSTANTE (puissance fixe)        ", 0)]

    for label, n_free in cases:
        O = observability_matrix(build_field(n_free), n_free)
        rk, M = generic_rank(O, n_free)
        print("%s : rang = %d / %d" % (label, rk, N))
        if rk < N:
            for v in M.nullspace():
                nz = [abs(c) for c in v if c != 0]
                print("    direction plate (coord. sur %s) :"
                      % [str(s) for s in UNKNOWNS])
                print("   ", [sp.nsimplify(c) for c in (v / max(nz))])
        print()

    print("Lecture : rang = 7/7 signifie que les deux etats ET les cinq parametres")
    print("sont localement identifiables a partir de la seule observation de A_P(t),")
    print("avec une entree u(t) connue -- y compris constante. Toute difficulte")
    print("d'estimation observee releve alors de l'identifiabilite PRATIQUE")
    print("(conditionnement / bruit), non de la structure du modele.")


if __name__ == "__main__":
    main()
