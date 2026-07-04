# P22 — Устойчивость маршрутизатора в Mixture-of-Experts

Экспериментальная проверка гипотезы: **граница решения роутера MoE проходит ближе к типичным точкам данных, чем граница основной задачи классификации.** Малое $\ell_\infty$-возмущение входа переключает выбор экспертов значительно раньше, чем ломает ответ модели.

Полное описание, результаты и графики — в [`PREPRINT.md`](PREPRINT.md).

## Главный результат

Направленная PGD-атака на роутер переключает ведущего эксперта у половины точек при возмущении $\varepsilon \approx 0.031$ — в ~15 раз меньшем, чем требуется случайному шуму ($\approx 0.46$). При $\varepsilon = 0.05$ роутер переключается у **73%** точек, тогда как точность модели держится на **93.9%**, а предсказание меняется лишь у **5.4%**. Гипотеза подтверждена количественно.

## Структура

```
.
├── config.yaml              # гиперпараметры (дублируют CONFIG в src/moe_model.py)
├── requirements.txt
├── download_data.sh         # скачивание MNIST в data/
├── PREPRINT.md              # препринт с результатами и графиками
├── src/
│   ├── moe_model.py         # MoE-слой, роутер, загрузка MNIST, load-balancing loss
│   └── train.py             # обучение + проверка отсутствия коллапса роутера
├── experiments/
│   ├── exp_a_random.py      # эксперимент A: случайный шум по сетке eps
│   ├── exp_b_pgd.py         # эксперимент B: untargeted PGD (margin / tz-loss)
│   ├── exp_b2_targeted.py   # targeted-атака + матрица переходов эксперт→эксперт
│   ├── exp_ablation_n4k1.py # ablation N=4,k=1: резкость границы
│   ├── make_plots.py        # генерация графиков fig1..fig5
│   └── make_ablation_plot.py # график сравнения конфигураций (fig6)
├── figures/                 # fig1..fig5 (png)
├── results/                 # результаты экспериментов (json)
└── checkpoints/
    └── checkpoint.pt        # обученная модель (test acc 97.46%)
```

## Установка

```bash
pip install -r requirements.txt
bash download_data.sh          # MNIST в ./data (idx-формат, зеркало на GitHub)
```

## Запуск

Скрипты находят данные, чекпоинт, результаты и графики по путям относительно репозитория, поэтому запускать можно из любого каталога. Пути переопределяются переменными окружения: `MOE_DATA`, `MOE_CKPT`, `MOE_RESULTS`, `MOE_FIGURES`.

**Обучение с нуля** (~2–3 мин на CPU):

```bash
cd src
python train.py               # обучает MoE, печатает загрузку экспертов, сохраняет checkpoint.pt
```

> Готовый чекпоинт (test acc 97.46%) уже лежит в `checkpoints/checkpoint.pt` — эксперименты запускаются без переобучения.

**Эксперименты** (данные ожидаются в `../data` относительно корня репозитория, либо задайте `MOE_DATA`):

```bash
cd experiments
python exp_a_random.py        # эксп. A: случайный шум      -> results/results_A_random.json
python exp_b_pgd.py           # эксп. B: untargeted PGD      -> results/results_B_pgd.json
python exp_b2_targeted.py     # targeted + матрица переходов -> results/results_B2_targeted.json
python make_plots.py          # графики раздела 6            -> figures/fig1..fig5.png
```

Результаты (json) и графики (png) пишутся в `results/` и `figures/` автоматически.

**Ablation $N=4, k=1$** (отдельная модель в `checkpoints/checkpoint_n4k1.pt`, test acc 96.45%):

```bash
cd experiments
python exp_ablation_n4k1.py    # random + margin-PGD на N=4,k=1 -> results/results_ablation_n4k1.json
python make_ablation_plot.py   # сравнение конфигураций          -> figures/fig6_ablation_n4k1.png
```

## Конфигурация

Основные параметры в `config.yaml` (и как `CONFIG`/константы в коде):

| Параметр | Значение | Смысл |
|---|---|---|
| `n_experts` (N) | 8 | число экспертов |
| `top_k` (k) | 2 | top-k routing |
| `lambda_balance` | 0.01 | вес load-balancing loss |
| `eps_grid` | 0…0.3 | сетка бюджета возмущения |
| `pgd_steps` | 20 | шагов PGD |
| `n_attack` | 2000 | размер подвыборки test для атаки |
| `random_M` | 20 | случайных возмущений на точку (эксп. A) |

Смена $N$ и $k$ — одна строка в `CONFIG` (`src/moe_model.py`). Ablation $N=4, k=1$ для более резкого эффекта — установить эти значения и переобучить.

## Ключевые детали реализации

- **Проекция PGD с клиппингом.** На MNIST большинство пикселей = 0, поэтому $\ell_\infty$-шар проецируется на пересечение с $[0,1]^{784}$ покоординатно. Иначе реальная норма возмущения меньше заявленной и сравнение random vs adversarial нечестно.
- **Margin-loss вместо $-\log(1-p_{i^*})$.** Градиент loss из ТЗ затухает вдали от $p_{i^*}=1$; margin даёт стабильный сигнал. Обе версии реализованы (`loss_type="margin"` / `"tz"`).
- **Load-balancing обязателен.** Без него линейный роутер коллапсирует в 2–3 эксперта и эксперимент по переключению вырождается.

## Воспроизводимость

Seed 42 для torch/numpy. Чекпоинт отделён от кода атак — эксперименты перезапускаются без переобучения.

## Автор

Никита — [github.com/progamaker](https://github.com/progamaker)
