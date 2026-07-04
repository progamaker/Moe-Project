"""MoE на MNIST: модель, данные, конфиг. P22 — устойчивость роутера."""
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "seed": 42,
    "n_experts": 8,          # N
    "top_k": 2,              # k
    "expert_hidden": 128,
    "input_dim": 784,
    "n_classes": 10,
    "train_subset": 10000,
    "batch_size": 256,
    "epochs": 15,
    "lr": 1e-3,
    "lambda_balance": 0.01,  # вес load-balancing loss
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_mnist(data_dir: str, subset: int | None):
    Xtr = load_idx(f"{data_dir}/train-images-idx3-ubyte.gz").reshape(-1, 784) / 255.0
    ytr = load_idx(f"{data_dir}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{data_dir}/t10k-images-idx3-ubyte.gz").reshape(-1, 784) / 255.0
    yte = load_idx(f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    if subset is not None:
        rng = np.random.default_rng(CONFIG["seed"])
        idx = rng.choice(len(Xtr), size=subset, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    to_t = lambda a, dt: torch.tensor(np.ascontiguousarray(a), dtype=dt)
    return (to_t(Xtr, torch.float32), to_t(ytr, torch.long),
            to_t(Xte, torch.float32), to_t(yte, torch.long))


class Expert(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    """y = sum_{i in TopK} g_i(x) * E_i(x); роутер линейный."""

    def __init__(self, cfg=CONFIG):
        super().__init__()
        self.N, self.k = cfg["n_experts"], cfg["top_k"]
        self.router = nn.Linear(cfg["input_dim"], self.N)
        self.experts = nn.ModuleList(
            Expert(cfg["input_dim"], cfg["expert_hidden"], cfg["n_classes"])
            for _ in range(self.N)
        )

    def route(self, x):
        """Возвращает p(x) [B,N], индексы top-k [B,k], перенормированные веса g [B,k]."""
        z = self.router(x)
        p = F.softmax(z, dim=-1)
        topv, topi = p.topk(self.k, dim=-1)
        g = topv / topv.sum(dim=-1, keepdim=True)
        return p, topi, g

    def forward(self, x, return_router=False):
        p, topi, g = self.route(x)
        # Плотная реализация: считаем всех экспертов (N=8 мало, MNIST дёшев),
        # маскируем выбором top-k. Проще для градиентов атаки, чем dispatch.
        all_out = torch.stack([e(x) for e in self.experts], dim=1)  # [B,N,C]
        gather = torch.gather(
            all_out, 1, topi.unsqueeze(-1).expand(-1, -1, all_out.size(-1))
        )  # [B,k,C]
        y = (g.unsqueeze(-1) * gather).sum(dim=1)  # [B,C]
        if return_router:
            return y, p, topi
        return y


def load_balance_loss(p: torch.Tensor, topi: torch.Tensor, N: int):
    """Switch-style aux loss: N * sum_i f_i * P_i,
    f_i — доля токенов, где эксперт i в top-k; P_i — средняя вероятность."""
    B = p.size(0)
    onehot = torch.zeros(B, N, device=p.device)
    onehot.scatter_(1, topi, 1.0)
    f = onehot.mean(dim=0)          # [N]
    P = p.mean(dim=0)               # [N]
    return N * (f * P).sum()
