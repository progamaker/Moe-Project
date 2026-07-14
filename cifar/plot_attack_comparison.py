"""Визуализация результатов attack_cifar_router.py.

Строит три графика из сохранённого JSON (checkpoints/attack_<ckpt>.json):
  1. accuracy_vs_eps_adversarial.png — падение точности под направленной атакой
  2. accuracy_vs_eps_random.png      — падение точности под случайным шумом
  3. accuracy_vs_eps_comparison.png  — оба графика вместе, для наглядного контраста

Запуск:
  python3 plot_attack_comparison.py --json checkpoints/attack_ck_n8_k4_v2.pt.json
"""
import argparse
import json
import os

import matplotlib.pyplot as plt


def load_curves(data):
    a = data["part_a"]
    c = data["part_c"]
    eps_a = [r["eps"] for r in a]
    accD_a = [r["accD"] for r in a]
    swk_a = [r["swk"] for r in a]

    eps_c = [r["eps"] for r in c]
    accD_c = [r["accD_mean"] for r in c]
    accD_c_std = [r["accD_std"] for r in c]
    swk_c = [r["swk_mean"] for r in c]
    swk_c_std = [r["swk_std"] for r in c]

    return dict(eps_a=eps_a, accD_a=accD_a, swk_a=swk_a,
                eps_c=eps_c, accD_c=accD_c, accD_c_std=accD_c_std,
                swk_c=swk_c, swk_c_std=swk_c_std)


def plot_swk_comparison(curves, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(curves["eps_a"], curves["swk_a"], marker="o", color="crimson",
            linewidth=2.5, label="направленная атака")
    ax.errorbar(curves["eps_c"], curves["swk_c"], yerr=curves["swk_c_std"],
               marker="s", color="steelblue", linewidth=2.5, capsize=4,
               label="случайный шум")
    ax.set_xlabel("ε (амплитуда возмущения, доля пикселя)")
    ax.set_ylabel("доля смены множества top-k (swk)")
    ax.set_ylim(0, 1)
    ax.set_title("Доля смены множества top-k: направленная атака vs случайный шум")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"сохранено: {out_path}")


def plot_comparison_normalized(curves, base_acc, out_path):
    """То же самое, что plot_comparison, но в относительных величинах:
    accD / base_acc, обе кривые стартуют от 1.0. Убирает визуальную путаницу
    "модель вообще слабая", т.к. базовая точность не участвует как потолок шкалы —
    показывает именно ДОЛЮ УТРАЧЕННОГО качества, а не абсолютные цифры."""
    rel_a = [a / base_acc for a in curves["accD_a"]]
    rel_c = [a / base_acc for a in curves["accD_c"]]
    rel_c_err = [e / base_acc for e in curves["accD_c_std"]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(curves["eps_a"], rel_a, marker="o", color="crimson",
            linewidth=2.5, label="направленная атака (adversarial)")
    ax.errorbar(curves["eps_c"], rel_c, yerr=rel_c_err,
               marker="s", color="steelblue", linewidth=2.5, capsize=4,
               label="случайный шум (та же амплитуда)")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="без атаки (100%)")
    ax.set_xlabel("ε (амплитуда возмущения, доля пикселя)")
    ax.set_ylabel("доля сохранённой точности (accD / чистая точность)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Относительная деградация точности: направленная атака vs случайный шум")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"сохранено: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="путь к attack_<ckpt>.json")
    ap.add_argument("--out_dir", default=None, help="куда сохранять графики "
                    "(по умолчанию — рядом с json, в подпапку figures/)")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    curves = load_curves(data)
    base_acc = data["base_acc"]

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.json) or ".", "figures")
    os.makedirs(out_dir, exist_ok=True)

    plot_comparison_normalized(curves, base_acc,
                                os.path.join(out_dir, "accuracy_retention_vs_eps_comparison.png"))
    plot_swk_comparison(curves, os.path.join(out_dir, "swk_vs_eps_comparison.png"))

    print(f"\nВсе графики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()