# ranklens

Библиотека и CLI для оффлайн-оценки качества ранжирования: посчитать метрики,
корректно сравнить два прогона и понять, почему они отличаются.

!!! warning "Статус"

    Ранняя разработка. Готовы метрики, чтение данных и статистика сравнения;
    объяснимость и отчёты — в работе.

## Пример

Считаем метрики прогона по разметке — из Python:

```python exec="on" source="block" result="text"
from ranklens.io import iter_run, read_qrels
from ranklens.metrics import evaluate

qrels = read_qrels("docs/examples/qrels.trec")
result = evaluate(iter_run("docs/examples/run.trec"), qrels, ["ndcg@10", "map", "mrr@10"])

for metric in result.metrics:
    print(f"{metric.metric:8} {metric.mean:.4f}  запросов: {metric.n_queries}")
print(f"без релевантных документов: {result.n_without_relevant}")
```

То же из командной строки:

```bash
ranklens eval --run run.trec --qrels qrels.trec --metrics ndcg@10 map mrr@10
```

## Что умеет

- **Метрики:** `ndcg`, `map`, `mrr`, `err`, `rbp` — по умолчанию совпадают с trec_eval
  до 1e-9, плюс `rbo` для сравнения двух порядков без разметки.
- **Данные:** csv, tsv, jsonl, TREC (в том числе сжатые), parquet, Feather,
  а также таблицы pandas и polars напрямую.
- **Потоковость:** в памяти одна выдача, а не весь прогон.
- **Статистика сравнения:** paired bootstrap по запросам и перестановочный тест —
  прирост либо отличим от шума выборки запросов, либо нет.
- **Расширяемость:** своя метрика подключается через entry points, без форка.

## Чего ranklens не делает

- не обучает модели — только оценивает;
- не отвечает, различаются ли **порядки** двух выдач: одинаковый NDCG при разном
  порядке вполне возможен, для этого есть `rbo`;
- не заменяет онлайн-эксперимент: оффлайн-метрика считается на фиксированной разметке.

## Установка

```bash
pip install ranklens            # после первого релиза
pip install 'ranklens[arrow]'   # + parquet, Feather, DataFrame
```
