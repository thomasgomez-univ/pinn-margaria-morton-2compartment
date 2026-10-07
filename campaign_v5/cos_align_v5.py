"""Alignement de l'erreur log avec la direction la moins contrainte de la FIM (sigma_P = 5 W, 50 athletes)."""
import json, glob, numpy as np
Z = np.load("complements_v4/fim_v3.npz"); soft = Z["soft"]            # (50,5), vecteur propre min, coord. sur ln theta, max comp = 1
V3 = [json.loads(l) for f in sorted(glob.glob("camp_v3_*.jsonl")) for l in open(f)]
V4 = {(r["i"], r["sigma_P"]): r for f in sorted(glob.glob("camp_v4_[0-9].jsonl")) for l in open(f) for r in [json.loads(l)]}
def cp(th): M_O, A_Om, A_Pm, M_R, eta = th; return M_O * (M_R * A_Om / (M_R * A_Om + M_O)) * eta
def grad_lncp(th):
    th = np.asarray(th, float); g = np.zeros(5); h = 1e-6
    for k in range(5):
        tp = th.copy(); tm = th.copy(); tp[k] *= 1 + h; tm[k] *= 1 - h
        g[k] = (np.log(cp(tp)) - np.log(cp(tm))) / (2 * h)
    return g
out = {}
for m in ["LM", "DE", "PINN", "PINN_v4"]:
    cf, cg, nd = [], [], []
    for r in V3:
        if r["sigma_P"] != 5.0: continue
        est = np.array(V4[(r["i"], 5.0)]["est"]) if m == "PINN_v4" else np.array(r["est"][m])
        d = np.log(est / np.array(r["theta_true"])); v = soft[r["i"]]; v = v / np.linalg.norm(v)
        g = grad_lncp(r["theta_true"]); g = g / np.linalg.norm(g)
        cf.append(abs(d @ v) / np.linalg.norm(d)); cg.append(abs(d @ g) / np.linalg.norm(d)); nd.append(np.linalg.norm(d))
    out[m] = dict(cos_flat=float(np.median(cf)), cos_gradCP=float(np.median(cg)), norm=float(np.median(nd)), n=len(cf))
    print("%-8s n=%d  |cos| flat %.3f   |cos| grad lnCP %.3f   |delta| %.3f" % (m, len(cf), out[m]["cos_flat"], out[m]["cos_gradCP"], out[m]["norm"]))
# reference isotrope en dimension 5 : mediane de |cos| (Monte Carlo)
rng = np.random.default_rng(0); X = rng.normal(size=(200000, 5)); print("isotrope 5D : mediane |cos| = %.3f" % np.median(np.abs(X[:, 0]) / np.linalg.norm(X, axis=1)))
json.dump(out, open("cos_align_v5.json", "w"), indent=1)
