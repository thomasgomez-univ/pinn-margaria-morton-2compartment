#!/usr/bin/env python3
"""Campagne v4 : PINN seul, variante H2, sur les 150 runs de la campagne v3.

Reutilise pipeline_v3 pour le modele, la population et la generation des
donnees ; les jeux de donnees sont regeneres a l'identique (meme graine,
sigma_AP lu dans camp_v3_*.jsonl pour eviter la recalibration Monte-Carlo).

Variante H2 (voir DIAGNOSTIC_PINN.md) :
  - protocoles A + B (comme LM et DE, et non B seul) ;
  - entrees du reseau (tau, sin 2*pi*t/60, cos 2*pi*t/60) : le reseau peut
    representer les commutations du protocole intermittent ;
  - condition initiale dure a(tau) = 1 + tau * N(tau), pas de sigmoide de sortie ;
  - 300 points de collocation par protocole ;
  - z gele pendant les n_warm premieres epoques (reseau seul), puis conjoint.

Usage : python3 pipeline_v4.py --shard K --nshards N [--seeds 1] [--sigma 2,5,10]
                               [--athletes 0-49] [--out camp_v4_K.jsonl]
"""
import argparse, glob, json, os, sys, time
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p3", os.path.join(HERE, "pipeline_v3.py"))
p3 = importlib.util.module_from_spec(_spec); sys.modules["p3"] = p3; _spec.loader.exec_module(p3)

H2 = dict(protos=("A", "B"), n_colloc=300, hard_ic=True, feat=True, period=60.0,
          w_r=0.3, n_adam=8000, n_lbfgs=80, n_warm=2000, lr=5e-3, lr_z=5e-3, width=48)


def fit_PINN_v4(data, seed=0, cfg=H2):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(1)
    lb = torch.tensor(p3.PARAM_LB, dtype=torch.float32); ub = torch.tensor(p3.PARAM_UB, dtype=torch.float32)
    z = nn.Parameter(torch.zeros(5))
    theta_of = lambda z: lb + (ub - lb) * torch.sigmoid(z)
    W = cfg["width"]; nin = 3 if cfg["feat"] else 1
    nets, D = {}, {}
    for kind in cfg["protos"]:
        d = data["protocols"][kind]
        t = np.asarray(d["t"]); obs = np.asarray(d["A_P_obs"]); T = float(d["t_end"])
        layers = [nn.Linear(nin, W), nn.Tanh(), nn.Linear(W, W), nn.Tanh(), nn.Linear(W, W), nn.Tanh(), nn.Linear(W, 2)]
        if not cfg["hard_ic"]: layers.append(nn.Sigmoid())
        nets[kind] = nn.Sequential(*layers)
        nc = cfg["n_colloc"]
        D[kind] = dict(T=T, scale=float(np.max(obs)) + 1.0,
                       t_d=torch.tensor(t / T, dtype=torch.float32).unsqueeze(1),
                       y_d=torch.tensor(obs / (float(np.max(obs)) + 1.0), dtype=torch.float32).unsqueeze(1),
                       tc=torch.linspace(0, 1, nc).unsqueeze(1),
                       Pc=torch.tensor([d["_P"](x) for x in np.linspace(0, T, nc)], dtype=torch.float32).unsqueeze(1))

    def traj(kind, tau):
        if cfg["feat"]:
            ph = 2 * np.pi * tau * D[kind]["T"] / cfg["period"]
            inp = torch.cat([tau, torch.sin(ph), torch.cos(ph)], dim=1)
        else:
            inp = tau
        out = nets[kind](inp)
        return 1.0 + tau * out if cfg["hard_ic"] else out

    def losses(ep, n_ep, parts=False):
        M_O, A_Om, A_Pm, M_R, eta = theta_of(z)
        ld = 0.0; lr_ = 0.0; li = 0.0
        for kind in cfg["protos"]:
            c = D[kind]; T = c["T"]
            pred = traj(kind, c["t_d"])
            ld = ld + torch.mean((pred[:, 1:2] * A_Pm / c["scale"] - c["y_d"]) ** 2)
            tt = c["tc"].clone().requires_grad_(True); pc = traj(kind, tt)
            aO, aP = pc[:, 0:1], pc[:, 1:2]
            daO = torch.autograd.grad(aO, tt, torch.ones_like(aO), True, True)[0]
            daP = torch.autograd.grad(aP, tt, torch.ones_like(aP), True, True)[0]
            rO = T * (M_R * (1 - aO) - (M_O / A_Om) * aO * (1 - aP))
            rP = T * ((M_O / A_Pm) * aO * (1 - aP) - c["Pc"] / (eta * A_Pm))
            lr_ = lr_ + torch.mean((daO - rO) ** 2 + (daP - rP) ** 2)
            if not cfg["hard_ic"]:
                p0 = traj(kind, torch.tensor([[0.0]])); li = li + (p0[0, 0] - 1) ** 2 + (p0[0, 1] - 1) ** 2
        w = cfg["w_r"] * min(1.0, ep / max(n_ep * 0.15, 1)) if n_ep else cfg["w_r"]
        if parts: return float(ld), float(lr_), float(li)
        return ld + w * lr_ + 5.0 * li

    pnet = [p for n in nets.values() for p in n.parameters()]; params = pnet + [z]
    opt = torch.optim.Adam([{"params": pnet, "lr": cfg["lr"]}, {"params": [z], "lr": cfg["lr_z"]}])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["n_adam"])
    for ep in range(cfg["n_adam"]):
        opt.zero_grad(); l = losses(ep, cfg["n_adam"]); l.backward()
        if ep < cfg["n_warm"]: z.grad = None
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
    if cfg["n_lbfgs"]:
        opt2 = torch.optim.LBFGS(params, lr=0.05, max_iter=cfg["n_lbfgs"], history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt2.zero_grad(); l = losses(cfg["n_adam"], 0); l.backward(); return l
        try: opt2.step(closure)
        except Exception: pass
    ld, lr_, li = losses(cfg["n_adam"], 0, parts=True)
    with torch.no_grad():
        return theta_of(z).numpy().astype(float), dict(loss_data=ld, loss_res=lr_, loss_ic=li)


def rmse_fit(theta, data):
    ss = 0.0; n = 0
    for kind, d in data["protocols"].items():
        tf = float(d["t"][-1]) * (1 + 1e-9)
        pred, _ = p3.simulate(np.asarray(theta, dtype=float), d["_P"], tf, np.asarray(d["t"]), switches=d["_sw"])
        ss += float(np.sum((pred - np.asarray(d["A_P_obs"])) ** 2)); n += len(d["t"])
    return float(np.sqrt(ss / n))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--shard", type=int, default=0); a.add_argument("--nshards", type=int, default=1)
    a.add_argument("--seeds", type=int, default=1); a.add_argument("--sigma", default="2,5,10")
    a.add_argument("--athletes", default="0-49"); a.add_argument("--out", default=None)
    a.add_argument("--v3", default=os.path.join(HERE, "camp_v3_*.jsonl"))
    a = a.parse_args()
    out = a.out or os.path.join(HERE, "camp_v4_%d.jsonl" % a.shard)
    sig = [float(s) for s in a.sigma.split(",")]
    lo, hi = (a.athletes.split("-") + [None])[:2]
    ath = list(range(int(lo), int(hi) + 1)) if hi else [int(x) for x in a.athletes.split(",")]
    V3 = {}
    for f in sorted(glob.glob(a.v3)):
        for l in open(f):
            r = json.loads(l); V3[(r["i"], r["sigma_P"])] = r
    if not V3: sys.exit("camp_v3_*.jsonl introuvables : %s" % a.v3)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["i"], r["sigma_P"], r["seed_k"]))
    jobs = [(i, s, k) for i in ath for s in sig for k in range(a.seeds)]
    jobs = [j for idx, j in enumerate(jobs) if idx % a.nshards == a.shard and j not in done]
    print("shard %d/%d : %d runs a faire -> %s" % (a.shard, a.nshards, len(jobs), out), flush=True)
    cfg = p3.Config()
    for (i, s, k) in jobs:
        r3 = V3.get((i, s))
        if r3 is None: print("  (%d, %g) absent de v3, saute" % (i, s)); continue
        th = np.array(r3["theta_true"]); sg = r3["sigma_AP"]
        p3.calibrate_sigma_AP = lambda t_, kind, s_, n_mc=100, seed=0, _s=sg: _s[kind]
        seed = 1000 * i + int(s * 10)
        data = p3.gen_data(th, s, seed, cfg)          # identique a la campagne v3
        t0 = time.time()
        x, parts = fit_PINN_v4(data, seed=seed + 7919 * k)
        dt = time.time() - t0
        rec = dict(i=i, sigma_P=s, seed_k=k, method="PINN_v4", variant="H2", theta_true=th.tolist(),
                   est=x.tolist(), wall_s=round(dt, 1), rmse_fit=rmse_fit(x, data), rmse_true=rmse_fit(th, data),
                   crlb_pct=r3["crlb_pct"], sigma_AP=sg, **parts)
        open(out, "a").write(json.dumps(rec) + "\n")
        print("  i=%2d s=%4.1f k=%d  eta=%.3f  rmse=%6.0f (vrai %4.0f)  %4.0fs" % (i, s, k, x[4], rec["rmse_fit"], rec["rmse_true"], dt), flush=True)


if __name__ == "__main__":
    main()
