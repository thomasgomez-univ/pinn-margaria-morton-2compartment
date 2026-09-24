#!/usr/bin/env python3
"""
Identifiabilite structurelle GLOBALE par elimination differentielle
(approche entree-sortie de Ljung & Glad, celle qu'implemente DAISY).

Principe
--------
On elimine l'etat latent A_O pour obtenir une equation entree-sortie unique
reliant y = A_P, ses derivees, l'entree u = P_ext et sa derivee. Les
coefficients de cette equation sont des fonctions des parametres. Le modele
est GLOBALEMENT identifiable si et seulement si l'application

    theta = (M_O, A_Omax, A_Pmax, M_R, eta)  -->  coefficients

est injective. On le teste en resolvant exactement le systeme
coeff(theta) = coeff(theta') et en examinant les solutions.

Derivation
----------
Etat : y = A_P, et l'auxiliaire w = M_O * A_O / A_Omax (flux oxydatif courant).
Avec s = 1 - y / A_Pmax :

    y'  = w * s - u / eta                                          (i)
    w'  = M_R * (M_O - w) - (M_O / A_Omax) * w * s                 (ii)

De (i) : w = (y' + u/eta) / s. En derivant et en reportant dans (ii), puis en
multipliant par s^2, on obtient l'equation entree-sortie :

    (y'' + u'/eta) * s + (y'/A_Pmax) * (y' + u/eta)
        = M_R*M_O*s^2 - M_R*s*(y' + u/eta) - (M_O/A_Omax)*s^2*(y' + u/eta)

Parametrisation reduite : a = 1/A_Pmax, b = 1/eta, c = M_R, d = M_R*M_O,
e = M_O/A_Omax. La bijection avec theta est immediate :
A_Pmax = 1/a, eta = 1/b, M_R = c, M_O = d/c, A_Omax = d/(c*e).

Dependance : sympy
"""

import sympy as sp

# ---------------------------------------------------------------- variables
y, dy, ddy, u, du = sp.symbols("y dy ddy u du")          # indeterminees
a, b, c, d, e = sp.symbols("a b c d e", positive=True)   # parametres reduits
a2, b2, c2, d2, e2 = sp.symbols("a2 b2 c2 d2 e2", positive=True)

MONOMES = [y, dy, ddy, u, du]


def io_equation(a, b, c, d, e):
    """Membre de gauche de l'equation entree-sortie, mis a zero."""
    s = 1 - a * y
    v = dy + b * u                      # = w * s
    return sp.expand((ddy + b * du) * s + a * dy * v
                     - d * s**2 + c * s * v + e * s**2 * v)


def coefficients(expr):
    """Dictionnaire monome -> coefficient."""
    p = sp.Poly(expr, *MONOMES)
    return {m: sp.simplify(co) for m, co in zip(p.monoms(), p.coeffs())}


def main():
    E1 = io_equation(a, b, c, d, e)
    E2 = io_equation(a2, b2, c2, d2, e2)

    print("Equation entree-sortie, developpee :")
    sp.pprint(sp.collect(E1, MONOMES))

    C1, C2 = coefficients(E1), coefficients(E2)
    assert set(C1) == set(C2)

    print("\nCoefficients (monome en (y, y', y'', u, u') -> expression) :")
    names = {}
    for m in sorted(C1, reverse=True):
        lab = "*".join("%s^%d" % (s, k) for s, k in zip(MONOMES, m) if k)
        names[m] = lab or "1"
        print("   %-14s : %s" % (names[m], C1[m]))

    eqs = [sp.Eq(C1[m], C2[m]) for m in C1 if sp.simplify(C1[m] - C2[m]) != 0]
    print("\n%d equations non triviales pour 5 inconnues." % len(eqs))

    sol = sp.solve(eqs, [a2, b2, c2, d2, e2], dict=True)
    print("\nSolutions de coeff(theta) = coeff(theta') :")
    for s_ in sol:
        print("   ", {str(k): sp.simplify(v) for k, v in s_.items()})

    ident = (len(sol) == 1 and
             all(sp.simplify(sol[0][k] - orig) == 0
                 for k, orig in zip([a2, b2, c2, d2, e2], [a, b, c, d, e])))
    print("\n=> %s" % ("GLOBALEMENT IDENTIFIABLE : l'unique solution est theta' = theta."
                       if ident else
                       "PAS globalement identifiable : plusieurs solutions ci-dessus."))
    print("""
Retour aux parametres physiques : A_Pmax = 1/a, eta = 1/b, M_R = c,
M_O = d/c, A_Omax = d/(c*e) -- bijection, donc la conclusion vaut pour
theta = (M_O, A_Omax, A_Pmax, M_R, eta).

Hypotheses : entree u(t) connue et suffisamment excitante, observation de
y = A_P sans bruit, s = 1 - y/A_Pmax non identiquement nul (vrai hors
epuisement). Conditions initiales traitees comme generiques : les exploiter
(A_O(0) = A_Omax, A_P(0) = A_Pmax) ne peut qu'ameliorer l'identifiabilite.""")


if __name__ == "__main__":
    main()


# ============================================================================
# Inversion explicite : chaque parametre se lit sur les coefficients
# ============================================================================

def inversion_explicite():
    """Formules d'inversion et verification numerique.

    L'identifiabilite globale est ici CONSTRUCTIVE : chaque parametre
    s'obtient par operations rationnelles sur les coefficients de l'equation
    entree-sortie.

        [y'^2]   = a              -> A_Pmax = 1 / [y'^2]
        [u']     = b              -> eta    = 1 / [u']
        [1]      = -d             -> M_R * M_O = -[1]
        [y^2 y'] = a^2 e          -> e = [y^2 y'] / [y'^2]^2
        [y']     = c + e          -> M_R = [y'] - e
                                     M_O = -[1] / M_R
                                     A_Omax = M_O / e
    """
    import numpy as np

    def coeffs_num(theta):
        M_O, A_Om, A_Pm, M_R, eta = theta
        av, bv, cv, dv, ev = 1 / A_Pm, 1 / eta, M_R, M_R * M_O, M_O / A_Om
        return {"dy2": av, "du": bv, "un": -dv, "y2dy": av**2 * ev,
                "dy": cv + ev}

    def invert(C):
        a_ = C["dy2"]
        A_Pm = 1 / a_
        eta = 1 / C["du"]
        e_ = C["y2dy"] / a_**2
        M_R = C["dy"] - e_
        M_O = -C["un"] / M_R
        A_Om = M_O / e_
        return np.array([M_O, A_Om, A_Pm, M_R, eta])

    print("\n" + "=" * 72)
    print("Verification numerique de l'inversion, 5 jeux de parametres tires")
    print("=" * 72)
    rng = np.random.default_rng(0)
    lb = np.array([600.0, 25_000.0, 30_000.0, 0.005, 0.18])
    ub = np.array([1800.0, 120_000.0, 160_000.0, 0.080, 0.32])
    worst = 0.0
    for k in range(5):
        th = rng.uniform(lb, ub)
        rec = invert(coeffs_num(th))
        err = np.max(np.abs(rec - th) / th)
        worst = max(worst, err)
        print("   jeu %d : ecart relatif max = %.2e" % (k, err))
    print("\n   ecart maximal sur les 5 jeux : %.2e (precision machine)" % worst)


if __name__ == "__main__":
    inversion_explicite()
