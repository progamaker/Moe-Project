"""Эксперимент B: направленная PGD-атака на роутер MoE.

Untargeted: сбить текущего top-1 эксперта i* = argmax p(x).
Основной loss — margin по логитам роутера: max_{j != i*} z_j - z_i*
(максимизируем). Дополнительно — loss из ТЗ: -log(1 - p_{i*}).

Проекция PGD учитывает клиппинг: delta ограничена одновременно
[-eps, eps] и [ -x, 1-x ] покоординатно, т.е. x+delta в [0,1] всегда.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# --- пути репозитория (переопределяются переменными окружения) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.environ.get("MOE_DATA", os.path.join(_ROOT, "..", "data"))
CKPT_PATH = os.environ.get("MOE_CKPT", os.path.join(_ROOT, "checkpoints", "checkpoint.pt"))
RESULTS_DIR = os.environ.get("MOE_RESULTS", os.path.join(_ROOT, "results"))
FIGURES_DIR = os.environ.get("MOE_FIGURES", os.path.join(_ROOT, "figures"))
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

import json
import torch
import torch.nn.functional as F
from moe_model import MoE, load_mnist, set_seed, CONFIG
from exp_a_random import router_stats, entropy, topk_set_changed, EPS_GRID

PGD_STEPS = 20
BS = 1024
N_ATTACK = 2000  # подвыборка test для атаки (PGD дорог: 20 шагов x forward+backward)


def pgd_router(model, x, i_star, eps, loss_type="margin", steps=PGD_STEPS):
    """Untargeted PGD на роутер. Возвращает x' = clip(x + delta)."""
    alpha = eps / 10 * 2.5  # чуть агрессивнее eps/10 из ТЗ, шагов хватает дойти до границы
    lo, hi = -x, 1.0 - x                      # допустимая delta от клиппинга
    lo = torch.maximum(lo, torch.full_like(x, -eps))
    hi = torch.minimum(hi, torch.full_like(x, eps))
    delta = torch.zeros_like(x)
    B = x.size(0)
    ar = torch.arange(B)
    for _ in range(steps):
        delta.requires_grad_(True)
        z = model.router(x + delta)
        if loss_type == "margin":
            z_star = z[ar, i_star]
            z_other = z.clone()
            z_other[ar, i_star] = -1e9
            loss = (z_other.max(-1).values - z_star).sum()
        else:  # "tz" (испр. знак): ascend по log(1 - p_i*), понижаем лидера
            p = F.softmax(z, dim=-1)
            loss = torch.log((1 - p[ar, i_star]).clamp_min(1e-12)).sum()
        (g,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta + alpha * g.sign()).clamp(lo, hi)  # проекция: eps-шар ∩ [0,1]
    return (x + delta).detach()


def run(model, Xte, yte, p0, topi0, pred0, loss_type):
    top1_0 = p0.argmax(-1)
    results = []
    for eps in EPS_GRID:
        if eps == 0.0:
            results.append(dict(eps=0.0, sw1=0.0, swk=0.0,
                                acc=(pred0 == yte).float().mean().item(),
                                pred_change=0.0, H=entropy(p0).mean().item(),
                                kl=0.0))
            continue
        sw1 = swk = acc = pc = Hs = kl = 0.0
        n = 0
        for i in range(0, len(Xte), BS):
            x = Xte[i:i+BS]
            sl = slice(i, i + len(x))
            xp = pgd_router(model, x, top1_0[sl], eps, loss_type)
            with torch.no_grad():
                y, p, topi = model(xp, return_router=True)
            pred = y.argmax(-1)
            sw1 += (p.argmax(-1) != top1_0[sl]).float().sum().item()
            swk += topk_set_changed(topi, topi0[sl]).float().sum().item()
            acc += (pred == yte[sl]).float().sum().item()
            pc  += (pred != pred0[sl]).float().sum().item()
            Hs  += entropy(p).sum().item()
            kl  += F.kl_div(p.clamp_min(1e-12).log(), p0[sl],
                            reduction="none").sum(-1).sum().item()
            n += len(x)
        row = dict(eps=eps, sw1=sw1/n, swk=swk/n, acc=acc/n,
                   pred_change=pc/n, H=Hs/n, kl=kl/n)
        results.append(row)
        print(f"[{loss_type}] eps={eps:5.2f}  switch_top1={row['sw1']:.4f}  "
              f"switch_topk_set={row['swk']:.4f}  acc={row['acc']:.4f}  "
              f"pred_change={row['pred_change']:.4f}  H={row['H']:.4f}")
    return results


def main():
    set_seed(CONFIG["seed"])
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    model = MoE(); model.load_state_dict(ckpt["model"]); model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    _, _, Xte, yte = load_mnist(DATA_DIR, None)

    gen = torch.Generator().manual_seed(CONFIG["seed"])
    idx = torch.randperm(len(Xte), generator=gen)[:N_ATTACK]
    Xa, ya = Xte[idx], yte[idx]
    p0, topi0, pred0 = router_stats(model, Xa)

    out = {}
    out["margin"] = run(model, Xa, ya, p0, topi0, pred0, "margin")
    print()
    out["tz"] = run(model, Xa, ya, p0, topi0, pred0, "tz")

    with open(os.path.join(RESULTS_DIR, "results_B_pgd.json"), "w") as f:
        json.dump({"n_attack": N_ATTACK, "pgd_steps": PGD_STEPS,
                   "results": out}, f, indent=2)
    print("\nСохранено: results_B_pgd.json")


if __name__ == "__main__":
    main()
