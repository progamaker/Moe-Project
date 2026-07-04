"""Эксперимент B2: targeted-атака на роутер + матрица переходов.

Targeted: заставить роутер выбрать конкретного i_target != i*,
максимизируя log p_{i_target}(x+delta). Для каждой точки цель
выбирается детерминированно: i_target = (i* + N//2) mod N.

Матрица переходов T[i,j]: сколько раз при untargeted margin-атаке
исходный лидер i был заменён на нового лидера j (при фикс. eps).
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
from exp_b_pgd import PGD_STEPS, BS, N_ATTACK

TRANSITION_EPS = 0.1  # eps, при котором строим матрицу переходов


def pgd_targeted(model, x, i_target, eps, steps=PGD_STEPS):
    """Targeted PGD: максимизируем log p_{i_target}. Проекция с клиппингом."""
    alpha = eps / 10 * 2.5
    lo = torch.maximum(-x, torch.full_like(x, -eps))
    hi = torch.minimum(1.0 - x, torch.full_like(x, eps))
    delta = torch.zeros_like(x)
    B = x.size(0); ar = torch.arange(B)
    for _ in range(steps):
        delta.requires_grad_(True)
        z = model.router(x + delta)
        logp = F.log_softmax(z, dim=-1)
        loss = logp[ar, i_target].sum()
        (g,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta + alpha * g.sign()).clamp(lo, hi)
    return (x + delta).detach()


def run_targeted(model, Xa, ya, p0, pred0):
    N = model.N
    top1_0 = p0.argmax(-1)
    i_target = (top1_0 + N // 2) % N
    results = []
    for eps in EPS_GRID:
        if eps == 0.0:
            results.append(dict(eps=0.0, hit=0.0, acc=(pred0 == ya).float().mean().item(),
                                pred_change=0.0))
            continue
        hit = acc = pc = 0.0; n = 0
        for i in range(0, len(Xa), BS):
            x = Xa[i:i+BS]; sl = slice(i, i + len(x))
            xp = pgd_targeted(model, x, i_target[sl], eps)
            with torch.no_grad():
                y, p, topi = model(xp, return_router=True)
            new_top1 = p.argmax(-1)
            hit += (new_top1 == i_target[sl]).float().sum().item()
            acc += (y.argmax(-1) == ya[sl]).float().sum().item()
            pc  += (y.argmax(-1) != pred0[sl]).float().sum().item()
            n += len(x)
        row = dict(eps=eps, hit=hit/n, acc=acc/n, pred_change=pc/n)
        results.append(row)
        print(f"[targeted] eps={eps:5.2f}  hit_rate={row['hit']:.4f}  "
              f"acc={row['acc']:.4f}  pred_change={row['pred_change']:.4f}")
    return results


def transition_matrix(model, Xa, p0, eps=TRANSITION_EPS):
    """T[i,j] = #(лидер i -> лидер j) при untargeted margin-атаке."""
    from exp_b_pgd import pgd_router
    N = model.N
    top1_0 = p0.argmax(-1)
    T = torch.zeros(N, N)
    for i in range(0, len(Xa), BS):
        x = Xa[i:i+BS]; sl = slice(i, i + len(x))
        xp = pgd_router(model, x, top1_0[sl], eps, "margin")
        with torch.no_grad():
            _, p, _ = model(xp, return_router=True)
        new = p.argmax(-1)
        for a, b in zip(top1_0[sl].tolist(), new.tolist()):
            T[a, b] += 1
    return T


def main():
    set_seed(CONFIG["seed"])
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    model = MoE(); model.load_state_dict(ckpt["model"]); model.eval()
    for prm in model.parameters(): prm.requires_grad_(False)
    _, _, Xte, yte = load_mnist(DATA_DIR, None)
    gen = torch.Generator().manual_seed(CONFIG["seed"])
    idx = torch.randperm(len(Xte), generator=gen)[:N_ATTACK]
    Xa, ya = Xte[idx], yte[idx]
    p0, topi0, pred0 = router_stats(model, Xa)

    tgt = run_targeted(model, Xa, ya, p0, pred0)
    print()
    T = transition_matrix(model, Xa, p0)
    print(f"Матрица переходов при eps={TRANSITION_EPS} (строка i = исходный лидер):")
    print(T.int().numpy())
    # "магнитные" эксперты: доля притянутых к j среди всех переключившихся
    off = T.clone(); off.fill_diagonal_(0)
    incoming = off.sum(0)  # сколько точек стало вести к j
    print("\nВходящие переходы (магнитность) по экспертам:")
    for j in range(model.N):
        print(f"  expert {j}: {int(incoming[j])}")

    with open(os.path.join(RESULTS_DIR, "results_B2_targeted.json"), "w") as f:
        json.dump({"targeted": tgt, "transition_eps": TRANSITION_EPS,
                   "transition_matrix": T.int().tolist(),
                   "incoming_magnet": incoming.tolist()}, f, indent=2)
    print("\nСохранено: results_B2_targeted.json")


if __name__ == "__main__":
    main()
