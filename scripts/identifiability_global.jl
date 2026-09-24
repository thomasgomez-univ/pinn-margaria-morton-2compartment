# =============================================================================
# Identifiabilite structurelle GLOBALE du modele bioenergetique a deux
# compartiments (Gomez, manuscrit CMBBE, Eqs. 1-5).
#
# Outil : StructuralIdentifiability.jl (algebre differentielle, SciML).
#   Dong, Goodbrake, Harrington & Pogudin, "Differential elimination for
#   dynamical models via projections with applications to structural
#   identifiability", SIAM J. Appl. Algebra Geom. (2023).
#   https://github.com/SciML/StructuralIdentifiability.jl
#
# Le script Python joint (identifiabilite_structurelle_fispo.py) donne
# l'identifiabilite LOCALE par critere de rang. Ce script-ci tranche la
# question GLOBALE (unicite et non seulement isolement des solutions) et
# fournit le resultat sous une forme citable pour la revue.
#
# Installation :
#   julia -e 'using Pkg; Pkg.add("StructuralIdentifiability")'
# Execution :
#   julia identifiabilite_globale.jl
# Duree attendue : quelques secondes a quelques minutes sur ce systeme.
# =============================================================================

using StructuralIdentifiability

# -----------------------------------------------------------------------------
# 1. Formulation du manuscrit, observable y = A_P, entree u = P_ext
#    Conditions initiales traitees comme generiques inconnues (hypothese la plus
#    conservatrice : on n'exploite pas A_O(0) = A_Omax ni A_P(0) = A_Pmax).
# -----------------------------------------------------------------------------
ode = @ODEmodel(
    AO'(t) = MR * (AOm - AO(t)) - MO * (AO(t) / AOm) * (1 - AP(t) / APm),
    AP'(t) =                      MO * (AO(t) / AOm) * (1 - AP(t) / APm) - u(t) / eta,
    y(t)   = AP(t)
)

println("="^70)
println("1. Identifiabilite locale (rapide)")
println("="^70)
display(assess_local_identifiability(ode))

println("\n" * "="^70)
println("2. Identifiabilite globale par parametre")
println("="^70)
display(assess_identifiability(ode))

println("\n" * "="^70)
println("3. Fonctions identifiables (generateurs du corps des invariants)")
println("="^70)
# Si un parametre est non identifiable, cette sortie donne les COMBINAISONS
# qui le sont -- c'est le resultat a rapporter dans l'article.
display(find_identifiable_functions(ode))

println("\n" * "="^70)
println("4. Fonctions identifiables incluant les etats initiaux")
println("="^70)
display(find_identifiable_functions(ode, with_states = true))

# -----------------------------------------------------------------------------
# 5. Variante exploitant les conditions initiales au repos
#
#    Le manuscrit pose A_O(0) = A_Omax et A_P(0) = A_Pmax (Eq. 8) : les CI ne
#    sont pas des inconnues independantes mais des fonctions des parametres.
#    StructuralIdentifiability traite par defaut les CI comme generiques, ce qui
#    SOUS-ESTIME l'identifiabilite. On normalise donc les etats pour que les CI
#    deviennent des constantes numeriques :
#
#        xO = A_O / A_Omax,  xP = A_P / A_Pmax   =>   xO(0) = xP(0) = 1
#
#        xO' = MR (1 - xO) - k  xO (1 - xP),        k = M_O / A_Omax
#        xP' =               m  xO (1 - xP) - u / (eta * A_Pmax),  m = M_O / A_Pmax
#        y   = A_Pmax * xP
#
#    La correspondance (MR, k, m, eta, APm) <-> (M_O, A_Omax, A_Pmax, M_R, eta)
#    est bijective :  A_Pmax = APm,  M_O = m*APm,  A_Omax = m*APm/k,  M_R = MR.
#    Les deux formulations ont donc la meme identifiabilite.
#
#    NB : le mot-cle `known_ic` n'existe que dans les versions recentes du
#    paquet. Verifier avec  ?assess_identifiability  avant d'executer ce bloc.
# -----------------------------------------------------------------------------
ode_norm = @ODEmodel(
    xO'(t) = MR * (1 - xO(t)) - k * xO(t) * (1 - xP(t)),
    xP'(t) =                    m * xO(t) * (1 - xP(t)) - u(t) / (eta * APm),
    y(t)   = APm * xP(t)
)

println("\n" * "="^70)
println("5. Formulation normalisee (CI connues xO(0) = xP(0) = 1)")
println("="^70)
display(assess_identifiability(ode_norm))
display(find_identifiable_functions(ode_norm))

try
    println("\n--- avec CI connues explicitement ---")
    display(assess_identifiability(ode_norm, known_ic = [xO, xP]))
catch err
    println("known_ic indisponible dans cette version : ", err)
end
