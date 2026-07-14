"""Обучение ResNetMoE на CIFAR-100 с нуля.

Запуск (Colab/Kaggle с GPU):
  pip install torch torchvision
  python train_cifar100.py --n_experts 8 --top_k 2 --epochs 100

CIFAR-100 скачивается автоматически через torchvision (нужен открытый интернет —
в отличие от песочницы, где готовился этот код, у Colab/Kaggle сеть не ограничена).
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from resnet_cifar_moe import ResNetMoE, load_balance_loss


def get_loaders(data_dir, batch_size, workers):
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def expert_usage(model, loader, device):
    model.eval()
    counts = torch.zeros(model.N, device=device)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        out = F.relu(model.bn1(model.conv1(x)))
        out = model.stage1(out)
        out = model.stage2(out)
        _, topi, _ = model.moe_stage.route(out)
        counts += torch.bincount(topi.flatten(), minlength=model.N).float()
    return (counts / counts.sum()).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--n_experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--lambda_balance", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_out", default="./checkpoints/checkpoint_cifar100.pt")
    ap.add_argument("--usage_every", type=int, default=5,
                    help="печатать загрузку экспертов каждые N эпох (0 = только в конце)")
    ap.add_argument("--resume_from", default=None,
                    help="путь к существующему чекпоинту — дообучение вместо обучения с нуля. "
                         "n_experts/top_k берутся из чекпоинта (переопределяют --n_experts/--top_k)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"          # Apple Silicon GPU (Metal)
    else:
        device = "cpu"
    if device == "cpu":
        print("ВНИМАНИЕ: без GPU это будет очень медленно (часы на эпоху). "
              "Проверь Colab/Kaggle с включённым GPU-рантаймом.")
    elif device == "mps":
        print("Используется Apple GPU через MPS. Если возникнет "
              "NotImplementedError на какой-то операции — это баг покрытия "
              "MPS-бэкенда в твоей версии torch; тогда добавь переменную "
              "окружения PYTORCH_ENABLE_MPS_FALLBACK=1 и перезапусти.")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.workers)

    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if resume_ckpt["n_experts"] != args.n_experts or resume_ckpt["top_k"] != args.top_k:
            print(f"ВНИМАНИЕ: n_experts/top_k из чекпоинта ({resume_ckpt['n_experts']}/"
                  f"{resume_ckpt['top_k']}) переопределяют переданные аргументы "
                  f"({args.n_experts}/{args.top_k})")
        args.n_experts = resume_ckpt["n_experts"]
        args.top_k = resume_ckpt["top_k"]
        print(f"Дообучение с {args.resume_from} (сохранённый test_acc="
              f"{resume_ckpt.get('test_acc')}), ещё {args.epochs} эпох")

    print(f"device={device}  N={args.n_experts}  k={args.top_k}  "
          f"lambda={args.lambda_balance}  epochs={args.epochs}")

    model = ResNetMoE(n_classes=100, n_experts=args.n_experts, top_k=args.top_k).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_ce = tot_bal = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, p, topi = model(x, return_router=True)
            ce = F.cross_entropy(logits, y)
            bal = load_balance_loss(p, topi, args.n_experts)
            loss = ce + args.lambda_balance * bal
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = y.size(0)
            tot_ce += ce.item() * bs
            tot_bal += bal.item() * bs
            correct += (logits.argmax(-1) == y).sum().item()
            total += bs
        sched.step()

        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"CE={tot_ce/total:.4f}  bal={tot_bal/total:.4f}  "
              f"train_acc={correct/total:.4f}  test_acc={test_acc:.4f}  "
              f"best={best_acc:.4f}  {dt:.1f}s")

        if args.usage_every and epoch % args.usage_every == 0:
            u = expert_usage(model, test_loader, device)
            usage_str = " ".join(f"{v:.2f}" for v in u.tolist())
            print(f"    usage[{','.join(str(i) for i in range(args.n_experts))}] "
                  f"= [{usage_str}]  min={u.min():.3f} max={u.max():.3f}")

    usage = expert_usage(model, test_loader, device)
    print("\nЗагрузка экспертов на test (доля попаданий в top-k):")
    for i, u in enumerate(usage):
        print(f"  expert {i}: {u.item():.4f}")
    print(f"min/max = {usage.min():.4f}/{usage.max():.4f}  "
          f"(равномерная = {1/args.n_experts:.4f})")

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "lambda_balance": args.lambda_balance,
        "test_acc": test_acc,
        "best_acc": best_acc,
    }, args.ckpt_out)
    print(f"\nЧекпоинт сохранён: {args.ckpt_out}")


if __name__ == "__main__":
    main()