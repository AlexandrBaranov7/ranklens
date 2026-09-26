# Первые шаги

За десять минут: от файлов с выдачей и разметкой до честного сравнения двух моделей.

## Что нужно

Два файла. **Прогон** — что модель выдала по каждому запросу, **разметка** —
насколько документы релевантны. Формат TREC понятен без схемы:

```text title="run.trec"
q1 Q0 d3 1 3.0 baseline
q1 Q0 d1 2 2.0 baseline
q2 Q0 d4 1 1.0 baseline
```

```text title="qrels.trec"
q1 0 d1 2
q1 0 d3 1
q2 0 d4 1
```

Столбцы прогона: запрос, служебное поле, документ, ранг, score, метка прогона.
Разметка: запрос, служебное поле, документ, релевантность.

!!! note "csv, jsonl, parquet и DataFrame"

    Формат определяется по расширению. Для csv и jsonl имена колонок задаются
    схемой `RunSchema`, а parquet, Feather и таблицы pandas и polars читаются
    напрямую: `iter_run(df)`.

## Шаг 1. Посчитать метрики

```python exec="on" source="block" result="text"
from ranklens.io import iter_run, read_qrels
from ranklens.metrics import evaluate

qrels = read_qrels("docs/examples/qrels.trec")
result = evaluate(iter_run("docs/examples/run.trec"), qrels, ["ndcg@10", "mrr@10"])

print(f"NDCG@10 = {result['ndcg@10'].mean:.4f}")
print(f"запросов оценено: {result.n_queries}")
print(f"без релевантных документов: {result.n_without_relevant}")
print(f"нет в разметке (пропущены): {result.n_unjudged}")
```

Прогон читается потоково: в памяти держится одна выдача, а не весь файл.
Значения по запросам лежат в `result["ndcg@10"].per_query` — они понадобятся дальше.

То же самое из терминала:

```bash
ranklens eval --run run.trec --qrels qrels.trec --metrics ndcg@10 mrr@10
```

Коды возврата различают ситуации: `0` — успех, `1` — ошибка конфигурации
(нет файла, неизвестная метрика), `2` — неверные аргументы, `3` — битые данные.
Так `ranklens eval` можно ставить в CI как гейт качества.

## Шаг 2. Сравнить две модели

Вопрос «стало лучше?» — это вопрос о том, отличим ли прирост от шума выборки запросов.

```python exec="on" source="block" result="text"
import numpy as np

from ranklens.core import MetricResult, QueryId
from ranklens.stats import compare

rng = np.random.default_rng(0)
queries = [QueryId(f"q{i}") for i in range(300)]
baseline = MetricResult(
    "ndcg@10", {q: float(v) for q, v in zip(queries, rng.normal(0.40, 0.10, 300))}
)
candidate = MetricResult(
    "ndcg@10", {q: float(v) for q, v in zip(queries, rng.normal(0.42, 0.10, 300))}
)

verdict = compare(baseline, candidate, seed=1)
print(f"было {verdict.mean_a:.4f} -> стало {verdict.mean_b:.4f}")
print(f"прирост {verdict.delta:+.4f}, интервал [{verdict.ci_low:+.4f}, {verdict.ci_high:+.4f}]")
print(f"p-value {verdict.p_value:.4f}, значимо: {verdict.significant}")
```

Как это читать:

- **прирост** — среднее по запросам разниц «кандидат минус базовая модель»;
- **интервал** — bootstrap по запросам: где правдоподобно лежит прирост,
  если бы запросы были другой выборкой из той же совокупности;
- **p-value** — перестановочный тест: вероятность увидеть такую же разницу
  или ещё более экстремальную, если на самом деле разницы нет.

!!! warning "Чего это не значит"

    Значимость не равна пользе: прирост в 0.001 может быть значимым на 100 000 запросов
    и не значить ничего на практике. И наоборот, на 30 запросах отсутствие значимости
    не означает, что улучшения нет — просто выборка мала, о чём библиотека предупредит.

Из терминала — два прогона и одна разметка:

```bash
ranklens compare --baseline a.csv --candidate b.csv --qrels qrels.csv \
    --metrics ndcg@10 mrr@10 --segments device locale
```

С `--segments` сравнение идёт отдельно в каждом сегменте, и среднее по всем запросам
больше не прячет, где модель выиграла, а где проиграла:

```text
metric   device   locale  queries  mean A  mean B  delta    95% CI              p       q
ndcg@10  desktop  ru      60       0.5352  0.5352  +0.0000  [+0.0000, +0.0000]  1.0000  1.0000
ndcg@10  mobile   en      60       0.6490  0.6329  -0.0161  [-0.0302, -0.0038]  0.0655  0.0982
ndcg@10  mobile   ru      60       0.6137  0.6926  +0.0789  [+0.0409, +0.1234]  0.0005  0.0015  *
ndcg@10  tablet   ru      1        -       -       -        too few queries     -       -
```

- **q** — p-value с поправкой Бенджамини–Хохберга: при сорока сегментах пара
  «значимых» появляется случайно, q-value это учитывает. Звёздочка — `q < alpha`.
- Сегмент меньше 20 запросов показан, но не сравнивается и в поправку не входит.
- Интервал считается по сегменту без поправки. В строке `mobile/en` он не содержит ноль,
  а q = 0.098: решает q-value. Без `--segments` поправка идёт по метрикам.
- Колонки сегментов берутся из прогона, поэтому нужен csv, jsonl или parquet — в TREC
  таких колонок нет. `--format json` отдаёт то же для скриптов.

## Шаг 3. Выбрать метрику под задачу

```python exec="on" source="block" result="text"
from ranklens.metrics import registry

print(", ".join(registry.names()))
```

Параметры задаются прямо в спецификации:

- `ndcg@10` — как в trec_eval (линейный gain);
- `ndcg(gain=exp)@10` — как в LightGBM и CatBoost;
- `map(rel=2)@100` — релевантными считаются оценки от 2;
- `rbp(p=0.95)` — «терпеливый» пользователь;
- `err(max_rel=3)@20` — шкала релевантности 0…3.

## Что дальше

- [Своя метрика](howto/custom-metric.md) — как добавить свою и подключить её плагином.
- [Справочник](api/io.md) — все функции и их параметры.
