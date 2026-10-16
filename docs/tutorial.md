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
- Последняя строка — `judged@K`, доля документов в top-K, у которых есть оценка.
  Неразмеченный документ считается нерелевантным, поэтому прогон, который приносит
  много нового, проигрывает из-за разметки, а не из-за качества. Если покрытие
  у прогонов расходится на 10 п.п. и больше, `compare` предупреждает.

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

## Шаг 4. Понять, где модель ошибается

Метрика упала — где и почему? Нужны модель и таблица признаков: `query_id, doc_id,
relevance` и по колонке на признак (как в датасетах learning-to-rank).

```bash
ranklens explain --model model.cbm --features features.csv --top-worst 8
```

```text
ndcg@10 over 120 queries: mean 0.9857
groups shuffled together: bm25+bm25_title

feature                        drop of ndcg@10 when shuffled
bm25+bm25_title                +0.3072
freshness                      +0.0558
price                          +0.0144

worst 8 queries (negative: the feature misleads the model):
q000             0.8388  freshness +0.272, bm25+bm25_title +0.162, price +0.099
q112             0.8953  bm25+bm25_title +0.268, freshness +0.187, price +0.055
...
```

- **Вклад — в метрику, а не в скор:** признак перемешивается среди документов запроса,
  документы сортируются заново, и меряется, насколько упал NDCG.
- **Коррелированные признаки перемешиваются вместе** (два варианта BM25 выше): по одному
  каждый выглядел бы неважным, потому что второй подменяет его.
- **Худшие запросы** с их причинами. Здесь все восемь — «новостные», где по построению
  решает свежесть: модель, обученная на всех запросах сразу, недооценивает её именно там.
  Отрицательный вклад значит, что перемешивание признака **улучшает** выдачу — модель
  доверяет признаку, который на этом запросе её обманывает.

Модели: CatBoost (`.cbm`), LightGBM (`.txt`), XGBoost (`.ubj`, `.json`), ONNX (`.onnx`,
в том числе экспорт из PyTorch, TensorFlow, Keras) и scikit-learn (`.pkl` — только
с `--allow-pickle`, см. SECURITY.md). Из Python то же: `ranklens.explain.listwise`,
`diagnose`, а для одной пары документов — `pairwise`.

## Шаг 5. Отчёт

У `compare`, `explain` и `offpolicy` есть `--report`: HTML — один файл со всеми графиками,
который открывается без интернета (plotly.js встроен один раз), Markdown — те же таблицы
без графиков, для пулл-реквеста или вики.

```bash
ranklens compare --baseline a.csv --candidate b.csv --qrels qrels.csv \
    --metrics ndcg@5 ndcg@10 mrr@10 --segments device locale --report compare.html
```

В отчёте сравнения: таблица, покрытие разметкой, разница по сегментам с интервалами
(значимые после BH — цветом), распределение поразапросных разниц (хвосты важнее
среднего), метрика по cutoff и куда переместились документы из топ-10. Файл пишется
атомарно: при ошибке остаётся прежний отчёт, а не половина нового.

## Что дальше

- [Своя метрика](howto/custom-metric.md) — как добавить свою и подключить её плагином.
- [Справочник](api/io.md) — все функции и их параметры.
