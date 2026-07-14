"""Аудит редундантности экспертов ResNetMoE (CIFAR-100).

Прямой перенос логики exp_c_redundancy.py / exp_f_grid.py с MNIST на
conv-архитектуру. Отличие только в том, что experts работают на картах
признаков [B,C,H,W], а не на плоских векторах, и что классификатор (GAP+fc)
применяется ПОСЛЕ эксперта — поэтому "solo"-прогон эксперта i на всех входах
считается как fc(GAP(expert_i(features))), а не forward всей модели.

Метрики (как в MNIST-версии):
  R        = mean_i acc(E_i) / acc(MoE)                — индекс редундантности
  agree    = среднее попарное согласие argmax(E_i) == argmax(E_j)
  A[i][j]  = acc(E_i | top-1 роутера = j)               — матрица специализации
  swap_acc = acc при случайной перестановке top-k экспертов (веса g сохранены)
  rank curve = acc эксперта, стоящего на ранге r роутера
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet_cifar_moe import ResNetMoE

BS = 500  # батч для инференса на всём test-сете (память conv-активаций)


@torch.no_grad()
def backbone_features(model, X, device):
    """stem + stage1 + stage2, применяется батчами. Возвращает [n,32,H,W] на CPU."""
    outs = []
    for i in range(0, len(X), BS):
        x = X[i:i+BS].to(device)
        out = F.relu(model.bn1(model.conv1(x)))
        out = model.stage1(out)
        out = model.stage2(out)
        outs.append(out.cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def expert_logits(model, feats, expert_idx, device):
    """fc(GAP(expert_i(feats))) для одного эксперта на всех точках, батчами."""
    outs = []
    e = model.moe_stage.experts[expert_idx]
    for i in range(0, len(feats), BS):
        f = feats[i:i+BS].to(device)
        y = e(f)
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        logits = model.fc(y)
        outs.append(logits.cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def route_all(model, feats, device):
    """p, topi, g роутера по всей выборке, батчами."""
    ps, topis, gs = [], [], []
    for i in range(0, len(feats), BS):
        f = feats[i:i+BS].to(device)
        p, topi, g = model.moe_stage.route(f)
        ps.append(p.cpu()); topis.append(topi.cpu()); gs.append(g.cpu())
    return torch.cat(ps), torch.cat(topis), torch.cat(gs)


@torch.no_grad()
def base_logits(model, feats, topi, g, device):
    """Полный MoE-выход (использует уже посчитанный маршрут topi/g)."""
    outs = []
    N = model.N
    for i in range(0, len(feats), BS):
        f = feats[i:i+BS].to(device)
        ti = topi[i:i+BS].to(device)
        gi = g[i:i+BS].to(device)
        all_out = torch.stack([e(f) for e in model.moe_stage.experts], dim=1)  # [b,N,C,H,W]
        idx = ti.view(*ti.shape, 1, 1, 1).expand(-1, -1, *all_out.shape[2:])
        gathered = torch.gather(all_out, 1, idx)
        y = (gi.view(*gi.shape, 1, 1, 1) * gathered).sum(1)
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        outs.append(model.fc(y).cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    N, k = ck["n_experts"], ck["top_k"]
    model = ResNetMoE(n_classes=100, n_experts=N, top_k=k).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"N={N} k={k}  сохранённая test_acc={ck.get('test_acc')}")

    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_ds = torchvision.datasets.CIFAR100(args.data_dir, train=False, download=True, transform=tf)
    Xte = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    yte = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    n = len(Xte)
    print(f"test set: {n} примеров")

    feats = backbone_features(model, Xte, device)   # [n,32,H,W] на CPU
    p, topi, g = route_all(model, feats, device)
    top1 = topi[:, 0]

    base_pred = base_logits(model, feats, topi, g, device).argmax(-1)
    base_acc = (base_pred == yte).float().mean().item()
    print(f"\nbase_acc (полный MoE) = {base_acc:.4f}")

    # --- эксперты по отдельности на ВСЕХ точках
    print("\n=== solo: forced routing на одного эксперта для всех входов ===")
    Epred = torch.zeros(N, n, dtype=torch.long)
    solo = []
    for i in range(N):
        logits = expert_logits(model, feats, i, device)
        pred = logits.argmax(-1)
        Epred[i] = pred
        acc = (pred == yte).float().mean().item()
        solo.append(acc)
        print(f"  expert {i}: acc={acc:.4f}  (Δ к базе {acc-base_acc:+.4f})")
    R = float(np.mean(solo) / base_acc)
    print(f"  min={min(solo):.4f} max={max(solo):.4f} mean={np.mean(solo):.4f}")
    print(f"  ИНДЕКС РЕДУНДАНТНОСТИ R = {R:.4f}")

    # --- матрица специализации A[i][j] = acc(E_i | top1 роутера = j)
    print("\n=== матрица специализации A[i][j] = acc(E_i | top-1 = j) ===")
    A = np.full((N, N), np.nan)
    share = []
    for j in range(N):
        m = (top1 == j)
        share.append(m.float().mean().item())
        if m.sum() == 0:
            continue
        for i in range(N):
            A[i, j] = (Epred[i][m] == yte[m]).float().mean().item()
    for i in range(N):
        row = "".join(("   —   " if np.isnan(A[i, j]) else f" {A[i,j]:.3f} ") for j in range(N))
        print(f"  i={i} |{row}")
    diag = np.array([A[j, j] for j in range(N) if not np.isnan(A[j, j])])
    off = np.array([A[i, j] for i in range(N) for j in range(N) if i != j and not np.isnan(A[i, j])])
    print(f"  доля точек с top-1=j: " + " ".join(f"{s:.3f}" for s in share))
    print(f"  диагональ (свой эксперт)  = {diag.mean():.4f}")
    print(f"  вне диагонали (чужой)     = {off.mean():.4f}")
    print(f"  разрыв специализации      = {diag.mean()-off.mean():+.4f}")

    # --- попарное согласие
    agree = np.array([[(Epred[i] == Epred[j]).float().mean().item() for j in range(N)] for i in range(N)])
    tri = agree[np.triu_indices(N, k=1)]
    print(f"\nпопарное согласие экспертов: mean={tri.mean():.4f} min={tri.min():.4f}")

    # --- ранговая кривая
    print("\n=== ранговая кривая: acc эксперта, стоящего на ранге r роутера ===")
    order = p.argsort(dim=-1, descending=True)
    ar = torch.arange(n)
    rank_acc, rank_p = [], []
    for r in range(N):
        idx = order[:, r]                                  # [n]
        sel = Epred.gather(0, idx.unsqueeze(0)).squeeze(0)  # [n], векторно вместо цикла
        ok = (sel == yte)
        rank_acc.append(ok.float().mean().item())
        rank_p.append(p[ar, idx].mean().item())
        print(f"  ранг {r+1}: acc={rank_acc[-1]:.4f}  ср.p={rank_p[-1]:.4f}")

    # --- случайный свап top-k (веса g сохранены, эксперты под ними — случайные)
    print("\n=== случайный свап top-k экспертов (g сохранены) ===")
    swap_accs = []
    with torch.no_grad():
        for t in range(5):
            gen = torch.Generator().manual_seed(t)
            preds = []
            for i in range(0, n, BS):
                f = feats[i:i+BS].to(device)
                bN = feats[i:i+BS].size(0)
                ri = torch.stack([torch.randperm(N, generator=gen)[:k] for _ in range(bN)]).to(device)
                gi = g[i:i+BS].to(device)
                all_out = torch.stack([e(f) for e in model.moe_stage.experts], dim=1)
                idx = ri.view(*ri.shape, 1, 1, 1).expand(-1, -1, *all_out.shape[2:])
                gathered = torch.gather(all_out, 1, idx)
                y = (gi.view(*gi.shape, 1, 1, 1) * gathered).sum(1)
                y = F.adaptive_avg_pool2d(y, 1).flatten(1)
                logits = model.fc(y)
                preds.append(logits.argmax(-1).cpu())
            pred = torch.cat(preds)
            swap_accs.append((pred == yte).float().mean().item())
    print(f"  acc = {np.mean(swap_accs):.4f} ± {np.std(swap_accs):.4f}  (5 сидов, база {base_acc:.4f})")

    res = dict(N=N, k=k, base_acc=base_acc, solo_acc=solo, R=R,
               diag_mean=float(diag.mean()), offdiag_mean=float(off.mean()),
               agree_mean=float(tri.mean()), agree_min=float(tri.min()),
               rank_acc=rank_acc, rank_p=rank_p,
               swap_acc_mean=float(np.mean(swap_accs)), swap_acc_std=float(np.std(swap_accs)))
    out_path = os.path.join(os.path.dirname(args.ckpt), f"audit_{os.path.basename(args.ckpt)}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nСохранено: {out_path}")
    print("=" * 60)
    print(f"R = {R:.4f}   разрыв специализации = {diag.mean()-off.mean():+.4f}   "
          f"swap_acc = {np.mean(swap_accs):.4f} (база {base_acc:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()