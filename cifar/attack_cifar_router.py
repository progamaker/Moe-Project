"""Атака на роутер ResNetMoE (CIFAR-100): "добавление шума" и анализ последствий.

Всё то же самое, что exp_d/exp_e/exp_g на MNIST, перенесённое на conv-архитектуру.
Атака ищется градиентным спуском через ВЕСЬ backbone (пиксели -> stem -> stage1 ->
stage2 -> роутер), в физическом пространстве пикселей [0,1], eps задаёт максимальную
амплитуду возмущения на пиксель (как в стандартных adversarial-бенчмарках).

ЧАСТЬ A — разложение "переключений" роутера (аналог exp_d + exp_e):
  Margin-PGD подталкивает top-1/top-2 эксперта к перестановке местами.
  Для каждого eps считаем отдельно:
    R = top-1 сменился, но МНОЖЕСТВО top-k то же   -> ожидаем малый урон
    S = множество top-k изменилось                  -> ожидаем больший урон
  Плюс контрфактуальный прогон: маршрут от x', эксперты видят ЧИСТЫЙ x
  (изолирует чистый эффект смены маршрута от эффекта "шум портит вход экспертам").

ЧАСТЬ B — targeted-атака к эксперту заданного ранга (аналог exp_g):
  Проверяем Закон 2: цена (eps для успеха) и урон должны расти с рангом r,
  если ранговое упорядочивание роутера согласовано с компетентностью экспертов.
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

MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)

EPS_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]   # амплитуда в пикселях [0,1]
RANKS = [2, 3, 4, 6, 8]
STEPS = 40
BS = 250


def normalize(x, device):
    return (x - MEAN.to(device)) / STD.to(device)


def backbone(model, x_raw, device):
    """stem+stage1+stage2 с градиентом (для PGD). x_raw в [0,1]."""
    x = normalize(x_raw, device)
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.stage1(out)
    out = model.stage2(out)
    return out


def mix_from_features(model, feats, topi, g):
    """y = sum_{i in topi} g_i * fc(GAP(E_i(feats)))."""
    all_out = torch.stack([e(feats) for e in model.moe_stage.experts], dim=1)  # [b,N,C,H,W]
    idx = topi.view(*topi.shape, 1, 1, 1).expand(-1, -1, *all_out.shape[2:])
    gathered = torch.gather(all_out, 1, idx)
    y = (g.view(*g.shape, 1, 1, 1) * gathered).sum(dim=1)
    y = F.adaptive_avg_pool2d(y, 1).flatten(1)
    return model.fc(y)


def pgd_margin(model, x_raw, top1_target, eps, device, steps=STEPS):
    """Толкаем логиты роутера так, чтобы margin(top1_target, второй по величине) -> 0.
    Это и есть "ближайшая граница роутера" — самое дешёвое место для флипа top-1."""
    alpha = eps / steps * 2.5
    delta = torch.zeros_like(x_raw, requires_grad=True)
    ar = torch.arange(x_raw.size(0))
    for _ in range(steps):
        x_adv = (x_raw + delta).clamp(0, 1)
        feats = backbone(model, x_adv, device)
        z = model.moe_stage.router(F.adaptive_avg_pool2d(feats, 1).flatten(1))
        z_t = z[ar, top1_target]
        z_other = z.clone()
        z_other[ar, top1_target] = -1e9
        z_second = z_other.max(dim=-1).values
        loss = (z_t - z_second).sum()  # минимизируем margin -> ascend on -loss
        (grad,) = torch.autograd.grad(-loss, delta)
        with torch.no_grad():
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
            delta = ((x_raw + delta).clamp(0, 1) - x_raw)
        delta.requires_grad_(True)
    return (x_raw + delta).clamp(0, 1).detach()


def pgd_targeted(model, x_raw, target_expert, eps, device, steps=STEPS):
    """Толкаем target_expert на позицию top-1 роутера (targeted-атака)."""
    alpha = eps / steps * 2.5
    delta = torch.zeros_like(x_raw, requires_grad=True)
    ar = torch.arange(x_raw.size(0))
    for _ in range(steps):
        x_adv = (x_raw + delta).clamp(0, 1)
        feats = backbone(model, x_adv, device)
        z = model.moe_stage.router(F.adaptive_avg_pool2d(feats, 1).flatten(1))
        z_t = z[ar, target_expert]
        z_other = z.clone()
        z_other[ar, target_expert] = -1e9
        loss = (z_t - z_other.max(dim=-1).values).sum()
        (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
            delta = ((x_raw + delta).clamp(0, 1) - x_raw)
        delta.requires_grad_(True)
    return (x_raw + delta).clamp(0, 1).detach()


def part_a(model, Xa, ya, device):
    print("\n" + "=" * 70)
    print("ЧАСТЬ A: разложение переключений роутера (R = переупорядочивание, S = смена множества)")
    print("=" * 70)

    with torch.no_grad():
        feats0 = backbone(model, Xa, device)
        p0, topi0, g0 = model.moe_stage.route(feats0)
        pred0 = mix_from_features(model, feats0, topi0, g0).argmax(-1)
    base_acc = (pred0 == ya).float().mean().item()
    top1_0 = p0.argmax(-1)
    set0 = [set(t.tolist()) for t in topi0]
    print(f"база: acc={base_acc:.4f}  N={model.N} k={model.k}\n")

    hdr = (f"{'eps':>5} | {'sw1':>6} {'swk':>6} | {'P(R)':>6} {'accB|R':>7} {'chgB|R':>7} | "
           f"{'P(S)':>6} {'accB|S':>7} {'chgB|S':>7} | {'accD':>6}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for eps in EPS_GRID:
        preds_B, top1s, sets, preds_D = [], [], [], []
        for i in range(0, len(Xa), BS):
            x = Xa[i:i+BS]
            sl = slice(i, i + len(x))
            xp = pgd_margin(model, x, top1_0[sl], eps, device)
            with torch.no_grad():
                feats_p = backbone(model, xp, device)
                p, topi, g = model.moe_stage.route(feats_p)
                yB = mix_from_features(model, feats0[sl], topi, g)   # эксперты видят ЧИСТЫЙ вход
                yD = mix_from_features(model, feats_p, topi, g)      # всё возмущено
            preds_B.append(yB.argmax(-1)); preds_D.append(yD.argmax(-1))
            top1s.append(p.argmax(-1))
            sets += [set(t.tolist()) for t in topi]
        predB, predD, top1 = torch.cat(preds_B), torch.cat(preds_D), torch.cat(top1s)

        set_chg = torch.tensor([s != s0 for s, s0 in zip(sets, set0)], device=device)
        t1_chg = (top1 != top1_0)
        R = t1_chg & (~set_chg)
        S = set_chg
        okB = (predB == ya).float()
        chgB = (predB != pred0).float()
        f = lambda m: (okB[m].mean().item(), chgB[m].mean().item()) if m.sum() > 0 else (float("nan"),) * 2
        accR, chgR = f(R)
        accS, chgS = f(S)
        accD = (predD == ya).float().mean().item()
        row = dict(eps=eps, sw1=t1_chg.float().mean().item(), swk=S.float().mean().item(),
                   pR=R.float().mean().item(), accB_R=accR, chgB_R=chgR,
                   pS=S.float().mean().item(), accB_S=accS, chgB_S=chgS, accD=accD)
        rows.append(row)
        print(f"{eps:5.2f} | {row['sw1']:6.3f} {row['swk']:6.3f} | "
              f"{row['pR']:6.3f} {accR:7.3f} {chgR:7.3f} | "
              f"{row['pS']:6.3f} {accS:7.3f} {chgS:7.3f} | {accD:6.3f}")
    print("\nsw1/swk = доля смены top-1 / доля смены множества top-k")
    print("accB|R,S = точность (эксперты видят ЧИСТЫЙ вход) на подгруппе R или S")
    print("chgB|R,S = доля точек, где ответ поменялся относительно чистого предсказания")
    print("accD = точность при полном возмущении (маршрут + вход эксперта)")
    return rows


def part_b(model, Xa, ya, device):
    print("\n" + "=" * 70)
    print("ЧАСТЬ B: targeted-атака к эксперту заданного ранга")
    print("=" * 70)

    with torch.no_grad():
        feats0 = backbone(model, Xa, device)
        p0, topi0, g0 = model.moe_stage.route(feats0)
        pred0 = mix_from_features(model, feats0, topi0, g0).argmax(-1)
    base_acc = (pred0 == ya).float().mean().item()
    order = p0.argsort(dim=-1, descending=True)
    print(f"база: acc={base_acc:.4f}\n")

    hdr = f"{'ранг':>5} | " + " ".join(f"{'e='+format(e,'.2f'):>13}" for e in EPS_GRID)
    print("формат ячейки: succ (доля успешных переводов) / acc (точность на успешных, чистый вход экспертам)\n")
    print(hdr); print("-" * len(hdr))
    rows = []
    for r in RANKS:
        tgt_all = order[:, r - 1]
        cells = []
        for eps in EPS_GRID:
            succ_l, ok_l = [], []
            for i in range(0, len(Xa), BS):
                x, sl = Xa[i:i+BS], slice(i, i + BS)
                t = tgt_all[sl]
                xp = pgd_targeted(model, x, t, eps, device)
                with torch.no_grad():
                    feats_p = backbone(model, xp, device)
                    p, topi, g = model.moe_stage.route(feats_p)
                    yB = mix_from_features(model, feats0[sl], topi, g)
                s = (p.argmax(-1) == t)
                succ_l.append(s)
                ok_l.append((yB.argmax(-1) == ya[sl]).float())
            s = torch.cat(succ_l); ok = torch.cat(ok_l)
            sr = s.float().mean().item()
            ac = ok[s].mean().item() if s.sum() > 0 else float("nan")
            cells.append((sr, ac))
            rows.append(dict(rank=r, eps=eps, succ=sr, acc=ac))
        print(f"{r:>5} | " + " ".join(
            f"{sr:5.2f}/{ac:5.3f}" if not np.isnan(ac) else "  —  /  —  " for sr, ac in cells))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--n_attack", type=int, default=1000)
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
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"N={N} k={k}  сохранённая test_acc={ck.get('test_acc')}")

    # СЫРЫЕ пиксели [0,1], БЕЗ нормализации — она встроена в backbone()
    tf = T.Compose([T.ToTensor()])
    test_ds = torchvision.datasets.CIFAR100(args.data_dir, train=False, download=True, transform=tf)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(test_ds), generator=gen)[:args.n_attack]
    Xa = torch.stack([test_ds[i][0] for i in idx]).to(device)
    ya = torch.tensor([test_ds[i][1] for i in idx]).to(device)
    print(f"атакуем {len(Xa)} примеров")

    rows_a = part_a(model, Xa, ya, device)
    rows_b = part_b(model, Xa, ya, device)

    out = dict(N=N, k=k, part_a=rows_a, part_b=rows_b)
    out_path = os.path.join(os.path.dirname(args.ckpt), f"attack_{os.path.basename(args.ckpt)}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()