"""Ablation N=4, k=1: switch rate = чистый argmax-флип роутера.
Прогоняем random (A) и margin-PGD (B) на той же сетке eps.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
DATA_DIR = os.environ.get("MOE_DATA", os.path.join(_ROOT, "..", "data"))
CKPT_N4 = os.environ.get("MOE_CKPT_N4", os.path.join(_ROOT, "checkpoints", "checkpoint_n4k1.pt"))
RESULTS_DIR = os.environ.get("MOE_RESULTS", os.path.join(_ROOT, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import json
import torch
import torch.nn.functional as F
import moe_model
from moe_model import MoE, load_mnist, set_seed

EPS_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
M = 20
BS = 2048
N_ATTACK = 2000
PGD_STEPS = 20


def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def topk_set_changed(a, b):
    return (a.sort(-1).values != b.sort(-1).values).any(-1)


def build_model(ckpt):
    cfg = ckpt["config"]
    moe_model.CONFIG.update(cfg)
    m = MoE(cfg); m.load_state_dict(ckpt["model"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def router_stats(model, X):
    ps, tis, prs = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), BS):
            y, p, ti = model(X[i:i+BS], return_router=True)
            ps.append(p); tis.append(ti); prs.append(y.argmax(-1))
    return torch.cat(ps), torch.cat(tis), torch.cat(prs)


def pgd_margin(model, x, i_star, eps, steps=PGD_STEPS):
    alpha = eps / 10 * 2.5
    lo = torch.maximum(-x, torch.full_like(x, -eps))
    hi = torch.minimum(1.0 - x, torch.full_like(x, eps))
    delta = torch.zeros_like(x); B = x.size(0); ar = torch.arange(B)
    for _ in range(steps):
        delta.requires_grad_(True)
        z = model.router(x + delta)
        zs = z[ar, i_star]
        zo = z.clone(); zo[ar, i_star] = -1e9
        loss = (zo.max(-1).values - zs).sum()
        (g,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta + alpha * g.sign()).clamp(lo, hi)
    return (x + delta).detach()


def run_random(model, Xte, yte, p0, topi0, pred0):
    top1_0 = p0.argmax(-1)
    res = []
    for eps in EPS_GRID:
        sw1 = swk = acc = pc = 0.0
        for m in range(M):
            gen = torch.Generator().manual_seed(1000 + m)
            for i in range(0, len(Xte), BS):
                x = Xte[i:i+BS]; sl = slice(i, i + len(x))
                d = (torch.rand(x.shape, generator=gen) * 2 - 1) * eps
                xp = (x + d).clamp(0, 1)
                with torch.no_grad():
                    y, p, ti = model(xp, return_router=True)
                sw1 += (p.argmax(-1) != top1_0[sl]).float().sum().item()
                swk += topk_set_changed(ti, topi0[sl]).float().sum().item()
                acc += (y.argmax(-1) == yte[sl]).float().sum().item()
                pc  += (y.argmax(-1) != pred0[sl]).float().sum().item()
        n = len(Xte) * M
        res.append(dict(eps=eps, sw1=sw1/n, swk=swk/n, acc=acc/n, pred_change=pc/n))
        print(f"[rand] eps={eps:5.2f} sw1={sw1/n:.4f} swk={swk/n:.4f} acc={acc/n:.4f} pc={pc/n:.4f}")
    return res


def run_pgd(model, Xa, ya, p0, topi0, pred0):
    top1_0 = p0.argmax(-1)
    res = []
    for eps in EPS_GRID:
        if eps == 0.0:
            res.append(dict(eps=0.0, sw1=0.0, swk=0.0,
                            acc=(pred0 == ya).float().mean().item(), pred_change=0.0))
            continue
        sw1 = swk = acc = pc = 0.0
        for i in range(0, len(Xa), 1024):
            x = Xa[i:i+1024]; sl = slice(i, i + len(x))
            xp = pgd_margin(model, x, top1_0[sl], eps)
            with torch.no_grad():
                y, p, ti = model(xp, return_router=True)
            sw1 += (p.argmax(-1) != top1_0[sl]).float().sum().item()
            swk += topk_set_changed(ti, topi0[sl]).float().sum().item()
            acc += (y.argmax(-1) == ya[sl]).float().sum().item()
            pc  += (y.argmax(-1) != pred0[sl]).float().sum().item()
        n = len(Xa)
        res.append(dict(eps=eps, sw1=sw1/n, swk=swk/n, acc=acc/n, pred_change=pc/n))
        print(f"[pgd ] eps={eps:5.2f} sw1={sw1/n:.4f} swk={swk/n:.4f} acc={acc/n:.4f} pc={pc/n:.4f}")
    return res


def main():
    set_seed(42)
    ckpt = torch.load(CKPT_N4, weights_only=False)
    model = build_model(ckpt)
    _, _, Xte, yte = load_mnist(DATA_DIR, None)
    p0, topi0, pred0 = router_stats(model, Xte)
    print(f"N={model.N} k={model.k}  clean_acc={(pred0==yte).float().mean():.4f}  "
          f"H={entropy(p0).mean():.4f}")

    rand = run_random(model, Xte, yte, p0, topi0, pred0)
    print()
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(Xte), generator=gen)[:N_ATTACK]
    Xa, ya = Xte[idx], yte[idx]
    pa, tia, pra = router_stats(model, Xa)
    pgd = run_pgd(model, Xa, ya, pa, tia, pra)

    json.dump({"config": {"N": model.N, "k": model.k},
               "clean_acc": (pred0 == yte).float().mean().item(),
               "random": rand, "pgd_margin": pgd},
              open(os.path.join(RESULTS_DIR, "results_ablation_n4k1.json"), "w"), indent=2)
    print("\nСохранено: results_ablation_n4k1.json")


if __name__ == "__main__":
    main()
