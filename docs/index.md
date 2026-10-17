# ranklens

Модель поменяли — стало ли лучше, **где** и **почему**? ranklens отвечает на это из
таблиц pandas одной строкой: разница с интервалом, p- и q-value, минимальный заметный
эффект (MDE), разбивка по сегментам с поправкой на множественные сравнения. Метрики
совпадают с trec_eval до 1e-9.

```python
import ranklens as rl

rl.compare(current, new, qrels, ["ndcg@10"], by="device")
```

## Что пригодилось на рабочих задачах

Перед релизом библиотеку прогнали как пользователь на двух задачах из продакшна.
Три вещи оказались самыми полезными — с них и стоит начинать.

### Эффект, невидимый в среднем

В каталожную модель добавили признак «товар в избранном у пользователя». Обучили по два
сида старой и новой модели и сравнили по сегментам:

```python
rl.compare([base_s0, base_s1], [flag_s0, flag_s1], qrels, ["ndcg@10"], by="has_flag")
```

| сегмент | запросов | Δ NDCG@10 | 95% интервал | MDE |
|---|---|---|---|---|
| все запросы | 100% | +0.0004 | [+0.0002; +0.0006] | 0.0003 |
| есть избранное | 2.8% | **+0.0147** | [+0.0117; +0.0177] | 0.004 |

В среднем эффект на пределе чувствительности — его легко списать на шум. Разбивка
показала, где он: там, где признак вообще есть, он большой, а на остальных запросах
потерь нет. Сиды усредняются по запросу, поэтому разница между ними не выдаётся
за эффект.

### Две выдачи размечены разными партиями асессоров

Текущую выдачу витрины сравнивали с новой: другой кандидатогенератор, а значит и другое
ранжирование. Каждую выдачу размечала своя партия краудсорсинга.

```python
rl.agreement(batch_cur, batch_new)  # 75% совпадений, вторая партия мягче на 0.05
qrels = rl.merge_qrels(batch_cur, batch_new)  # одна разметка для обеих сторон
rl.compare(current, new, qrels, ["ndcg@5", "ndcg@10", "err@10"])
```

На «своих» оценках разница партий смешалась бы с разницей систем. На общей разметке
NDCG@10 значимо выше (q = 0.034), а NDCG@5 и ERR@10 — нет: новые кандидаты делают
выдачу полнее и улучшают позиции 5–10, а первые карточки не изменились. Какая метрика
главная — решает то, сколько карточек реально смотрят.

### Почему: вклад признаков в метрику

```python
ex = rl.explainer(model, features, query="qid", doc="sku", relevance="relevance")
ex.importance("ndcg@10")  # насколько падает метрика, если перемешать признак
ex.worst(20)  # худшие запросы и признаки, которые их подвели
```

Новый признак оказался первым по вкладу в NDCG@10 на запросах, где он есть. Адаптеры:
CatBoost, LightGBM, XGBoost, scikit-learn, PyTorch, ONNX или любая функция скоринга.

## Попробовать

```python exec="on" source="block" result="text"
import numpy as np
import pandas as pd

import ranklens as rl

rng = np.random.default_rng(1)
rows = []
for q in range(300):
    device = "mobile" if q % 2 else "desktop"
    for d in range(10):
        grade = int(rng.integers(0, 4))
        noise_new = 1.0 if device == "mobile" else 1.6  # новая модель лучше на mobile
        scores = grade + rng.normal(0, 1.6), grade + rng.normal(0, noise_new)
        rows.append((q, d, grade, device, *scores))
df = pd.DataFrame(rows, columns=["qid", "sku", "grade", "device", "current", "new"])

qrels = rl.qrels(df, query="qid", doc="sku", relevance="grade")
current = rl.run(df, query="qid", doc="sku", score="current", segments="device")
new = rl.run(df, query="qid", doc="sku", score="new", segments="device")

result = rl.compare(current, new, qrels, ["ndcg@10"], by="device")
print(result.table[["n_queries", "delta", "ci_low", "ci_high", "mde", "q_value"]].round(4))
```

Дальше — [«Первые шаги»](tutorial.md): файлы и CLI, отчёт с графиками, объяснение модели.

## Что ещё умеет

- **Метрики:** `ndcg`, `map`, `mrr`, `err`, `rbp`, `judged` — по умолчанию как в trec_eval,
  плюс `rbo` для сравнения двух порядков без разметки; своя метрика — плагином.
- **Данные:** таблицы pandas, polars, pyarrow; файлы csv, tsv, jsonl, TREC (в том числе
  сжатые), parquet, Feather — потоком, в памяти одна выдача.
- **Неточная разметка:** покрытие judged@k для обеих сторон с предупреждением о разрыве;
  PPI++ — дешёвая разметка всего плюс небольшая gold-выборка.
- **Клики:** IPS и SNIPS с поправкой на позицию, диагностика весов (ESS), симулятор
  с известной истиной — [рецепт](howto/offpolicy.md).
- **Отчёты:** самодостаточный HTML или Markdown с таблицей и графиками.
- **CLI:** `ranklens eval`, `compare`, `explain` — для файлов и гейтов качества в CI.

## Чего ranklens не делает

- не обучает модели — только оценивает;
- не отвечает, различаются ли **порядки** двух выдач: одинаковый NDCG при разном
  порядке вполне возможен, для этого есть `rbo`;
- не заменяет онлайн-эксперимент: оффлайн-метрика считается на фиксированной разметке.

## Установка

```bash
pip install "ranklens[viz] @ git+https://github.com/AlexandrBaranov7/ranklens"
```

Версия 0.1.0 опубликована на TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ranklens
```

Extras: `viz` (графики и отчёты), `arrow` (parquet, Feather), `models` (CatBoost,
LightGBM, XGBoost, scikit-learn), `neural` (ONNX).
