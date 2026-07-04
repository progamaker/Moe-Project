"""Эксперимент A: устойчивость роутера к случайному шуму.

Для каждого x из test и каждого eps из сетки: M равномерных возмущений,
метрики switch rate (top-1 / top-k set), accuracy, энтропия, KL.
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

EPS_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
M = 20
BS = 2048


def router_stats(model, X, bs=BS):
    """p, top-k индексы, предсказание класса — по всей выборке."""
    ps, topis, preds = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            y, p, topi = model(X[i:i+bs], return_router=True)
            ps.append(p); topis.append(topi); preds.append(y.argmax(-1))
    return torch.cat(ps), torch.cat(topis), torch.cat(preds)


def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def topk_set_changed(topi0, topi1):
    """1, если состав top-k набора отличается (как множество)."""
    s0, _ = topi0.sort(-1)
    s1, _ = topi1.sort(-1)
    return (s0 != s1).any(-1)


def main():
    set_seed(CONFIG["seed"])
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    model = MoE(); model.load_state_dict(ckpt["model"]); model.eval()
    _, _, Xte, yte = load_mnist(DATA_DIR, None)

    p0, topi0, pred0 = router_stats(model, Xte)
    top1_0 = p0.argmax(-1)
    H0 = entropy(p0).mean().item()
    clean_acc = (pred0 == yte).float().mean().item()
    print(f"clean: acc={clean_acc:.4f}, H(p)={H0:.4f}")

    results = []
    for eps in EPS_GRID:
        agg = {k: 0.0 for k in
               ["sw1", "swk", "acc", "H", "kl", "pred_change"]}
        for m in range(M):
            gen = torch.Generator().manual_seed(CONFIG["seed"] * 1000 + m)
            sw1 = swk = acc = Hs = kl = pc = 0.0
            n = 0
            for i in range(0, len(Xte), BS):
                x = Xte[i:i+BS]
                delta = (torch.rand(x.shape, generator=gen) * 2 - 1) * eps
                xp = (x + delta).clamp(0, 1)
                with torch.no_grad():
                    y, p, topi = model(xp, return_router=True)
                pred = y.argmax(-1)
                sl = slice(i, i + len(x))
                sw1 += (p.argmax(-1) != top1_0[sl]).float().sum().item()
                swk += topk_set_changed(topi, topi0[sl]).float().sum().item()
                acc += (pred == yte[sl]).float().sum().item()
                pc  += (pred != pred0[sl]).float().sum().item()
                Hs  += entropy(p).sum().item()
                kl  += F.kl_div(p.clamp_min(1e-12).log(), p0[sl],
                                reduction="none").sum(-1).sum().item()
                n += len(x)
            agg["sw1"] += sw1 / n; agg["swk"] += swk / n
            agg["acc"] += acc / n; agg["pred_change"] += pc / n
            agg["H"] += Hs / n;   agg["kl"] += kl / n
        row = {k: v / M for k, v in agg.items()}
        row["eps"] = eps
        results.append(row)
        print(f"eps={eps:5.2f}  switch_top1={row['sw1']:.4f}  "
              f"switch_topk_set={row['swk']:.4f}  acc={row['acc']:.4f}  "
              f"pred_change={row['pred_change']:.4f}  H={row['H']:.4f}  "
              f"KL={row['kl']:.5f}")

    with open(os.path.join(RESULTS_DIR, "results_A_random.json"), "w") as f:
        json.dump({"clean_acc": clean_acc, "H0": H0, "M": M,
                   "results": results}, f, indent=2)
    print("\nСохранено: results_A_random.json")


if __name__ == "__main__":
    main()
