"""Обучение MoE на MNIST + проверка отсутствия коллапса роутера."""
import torch
import torch.nn.functional as F
from moe_model import CONFIG, MoE, load_mnist, set_seed, load_balance_loss

def expert_usage(model, X, bs=1024):
    """Доля попаданий каждого эксперта в top-k по выборке."""
    counts = torch.zeros(model.N)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            _, topi, _ = model.route(X[i:i+bs])
            counts += torch.bincount(topi.flatten(), minlength=model.N).float()
    return counts / counts.sum()

def accuracy(model, X, y, bs=1024):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            correct += (model(X[i:i+bs]).argmax(-1) == y[i:i+bs]).sum().item()
    return correct / len(X)

def main():
    set_seed(CONFIG["seed"])
    import os
    data_dir = os.environ.get("MOE_DATA", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
    Xtr, ytr, Xte, yte = load_mnist(data_dir, CONFIG["train_subset"])
    model = MoE()
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    bs, N = CONFIG["batch_size"], CONFIG["n_experts"]

    for epoch in range(1, CONFIG["epochs"] + 1):
        perm = torch.randperm(len(Xtr))
        tot_ce = tot_bal = 0.0
        model.train()
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr[idx], ytr[idx]
            logits, p, topi = model(xb, return_router=True)
            ce = F.cross_entropy(logits, yb)
            bal = load_balance_loss(p, topi, N)
            loss = ce + CONFIG["lambda_balance"] * bal
            opt.zero_grad(); loss.backward(); opt.step()
            tot_ce += ce.item() * len(xb); tot_bal += bal.item() * len(xb)
        model.eval()
        acc = accuracy(model, Xte, yte)
        print(f"epoch {epoch:2d}  CE={tot_ce/len(Xtr):.4f}  bal={tot_bal/len(Xtr):.4f}  test_acc={acc:.4f}")

    usage = expert_usage(model, Xte)
    print("\nЗагрузка экспертов на test (доля попаданий в top-k):")
    for i, u in enumerate(usage):
        print(f"  expert {i}: {u.item():.4f}")
    print(f"min/max = {usage.min():.4f}/{usage.max():.4f}  (равномерная = {1/N:.4f})")
    import os
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.environ.get("MOE_CKPT", os.path.join(ckpt_dir, "checkpoint.pt"))
    torch.save({"model": model.state_dict(), "config": CONFIG, "test_acc": acc},
               ckpt_path)
    print(f"\nЧекпоинт сохранён: {ckpt_path}")

if __name__ == "__main__":
    main()
