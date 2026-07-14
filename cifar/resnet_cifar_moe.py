"""CIFAR-style ResNet (He et al., 3 стадии 16/32/64) с ОДНИМ MoE-слоем
вместо последней стадии. Эксперты = residual BasicBlock'и, роутер = Linear
поверх GAP входной карты признаков. Один узел маршрутизации — прямой аналог
MNIST-эксперимента (moe_model.py), что позволяет переиспользовать логику
аудита (exp_c/f/g) почти без изменений.

Обоснование места MoE: последняя стадия (64 канала, 8x8) — самые абстрактные
признаки backbone'а. Роутер на сырых пикселях был бы линейным пробником, не
связанным с backbone; MoE-голова поверх готовых h(x) вырождает экспертов
тривиально (признаки почти линейно разделимы). Маршрутизация ВНУТРИ residual-
стадии — единственный вариант, соответствующий реальным MoE-архитектурам
(Switch/Mixtral маршрутизируют по скрытым состояниям, не по входу и не по
финальным признакам).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Стандартный residual-блок: conv-bn-relu-conv-bn + skip, затем relu."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


def make_stage(c_in, c_out, n_blocks, stride):
    layers = [BasicBlock(c_in, c_out, stride)]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlock(c_out, c_out, 1))
    return nn.Sequential(*layers)


class MoEStage(nn.Module):
    """Один узел маршрутизации: y = sum_{i in TopK} g_i(x) * E_i(x).

    Эксперты — полноценные BasicBlock (с понижением разрешения и сменой
    числа каналов на первом эксперте наравне со всеми, т.к. каждый эксперт
    должен уметь сыграть роль всей стадии, а не только части).
    Роутер смотрит на GAP входной карты признаков (до понижения разрешения).
    Плотная реализация (считаем всех N экспертов) — как в MNIST-версии,
    удобно для градиентов атаки на роутер.
    """

    def __init__(self, c_in, c_out, n_experts, top_k, stride=2):
        super().__init__()
        self.N, self.k = n_experts, top_k
        self.router = nn.Linear(c_in, n_experts)
        self.experts = nn.ModuleList(
            BasicBlock(c_in, c_out, stride) for _ in range(n_experts)
        )

    def route(self, x):
        z = self.router(F.adaptive_avg_pool2d(x, 1).flatten(1))
        p = F.softmax(z, dim=-1)
        topv, topi = p.topk(self.k, dim=-1)
        g = topv / topv.sum(dim=-1, keepdim=True)
        return p, topi, g

    def forward(self, x, return_router=False):
        p, topi, g = self.route(x)
        outs = torch.stack([e(x) for e in self.experts], dim=1)          # [B,N,C,H,W]
        idx = topi.view(*topi.shape, 1, 1, 1).expand(-1, -1, *outs.shape[2:])
        gathered = torch.gather(outs, 1, idx)                            # [B,k,C,H,W]
        y = (g.view(*g.shape, 1, 1, 1) * gathered).sum(dim=1)
        if return_router:
            return y, p, topi
        return y


class ResNetMoE(nn.Module):
    """conv stem -> stage1(16) -> stage2(32) -> MoEStage(32->64) -> GAP -> fc."""

    def __init__(self, n_classes=100, n_blocks=(3, 3), n_experts=8, top_k=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = make_stage(16, 16, n_blocks[0], stride=1)
        self.stage2 = make_stage(16, 32, n_blocks[1], stride=2)
        self.moe_stage = MoEStage(32, 64, n_experts, top_k, stride=2)
        self.fc = nn.Linear(64, n_classes)
        self.N, self.k = n_experts, top_k

    def forward(self, x, return_router=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        if return_router:
            out, p, topi = self.moe_stage(out, return_router=True)
        else:
            out = self.moe_stage(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.fc(out)
        if return_router:
            return logits, p, topi
        return logits


def load_balance_loss(p: torch.Tensor, topi: torch.Tensor, N: int):
    """Switch-style aux loss, идентична MNIST-версии (moe_model.py)."""
    B = p.size(0)
    onehot = torch.zeros(B, N, device=p.device)
    onehot.scatter_(1, topi, 1.0)
    f = onehot.mean(dim=0)
    P = p.mean(dim=0)
    return N * (f * P).sum()


if __name__ == "__main__":
    # быстрый smoke-test формы тензоров и градиентов на синтетике
    torch.manual_seed(0)
    m = ResNetMoE(n_classes=100, n_experts=8, top_k=2)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 100, (4,))
    logits, p, topi = m(x, return_router=True)
    assert logits.shape == (4, 100)
    assert p.shape == (4, 8)
    assert topi.shape == (4, 2)
    loss = F.cross_entropy(logits, y) + 0.01 * load_balance_loss(p, topi, 8)
    loss.backward()
    n_grad = sum(p_.grad is not None for p_ in m.parameters())
    n_all = sum(1 for _ in m.parameters())
    print(f"OK: logits {logits.shape}, loss={loss.item():.4f}, "
          f"градиент получили {n_grad}/{n_all} тензоров параметров")