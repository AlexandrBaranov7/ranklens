# Своя метрика

Метрика — любой вызываемый объект с сигнатурой
`(ranked, judgements, k) -> float` (протокол `ranklens.core.Metric`). Наследоваться
ни от чего не нужно.

- `ranked` — документы выдачи по порядку (позиция 0 — первая);
- `judgements` — разметка запроса `doc_id -> relevance`; неразмеченный документ
  отсутствует в словаре и считается нерелевантным;
- `k` — отсечка из спецификации (`hits@10` → 10) или `None`, если её нет.

## В своём коде: реестр

```python
from dataclasses import dataclass

from ranklens.core.registry import Registry
from ranklens.metrics import NDCG, evaluate


@dataclass(frozen=True)
class Hits:
    """Число релевантных документов в top-k."""

    rel: float = 1  # параметр: hits(rel=2)@10

    def __call__(self, ranked, judgements, k):
        return float(sum(judgements.get(doc, 0) >= self.rel for doc in ranked[:k]))


registry = Registry()
registry.register("hits", Hits)  # класс — это фабрика: Hits(**параметры из спецификации)
registry.register("ndcg", NDCG)

result = evaluate(runs, qrels, ["hits(rel=2)@10", "ndcg@10"], registry=registry)
```

Метрику без параметров можно зарегистрировать декоратором:

```python
@registry.metric("first_relevant")
def first_relevant(ranked, judgements, k): ...
```

Параметры спецификации приходят как `int`, `float` или строка. Проверяйте их
в `__post_init__` и бросайте `ValueError` — реестр превратит его в понятную
`MetricSpecError`.

## Для всех: плагин через entry points

Чтобы метрика появилась в `ranklens eval` и во встроенном реестре без правок ranklens,
опубликуйте её фабрику в группе entry points `ranklens.metrics` своего пакета:

```toml
# pyproject.toml вашего пакета
[project.entry-points."ranklens.metrics"]
hits = "my_package.metrics:Hits"
```

После `pip install my-package`:

```bash
ranklens metrics                     # в списке появится hits
ranklens eval --run run.trec --qrels qrels.trec --metrics 'hits(rel=2)@10' ndcg@10
```

Правила:

- имя — строчные латинские буквы, цифры и `_`, начинается с буквы;
- значение entry point — **фабрика**: класс метрики или функция без обязательных
  аргументов, возвращающая метрику. Функция-метрика `(ranked, judgements, k)` сама
  фабрикой не является — оберните её классом или функцией `lambda: metric`;
- плагины загружаются при первом обращении к реестру, а не при `import ranklens`.

Если плагин не импортируется, падает при загрузке или занимает уже существующее имя
(например, `ndcg`), он **пропускается** с предупреждением `PluginWarning`, где указаны
имя, путь и пакет. Остальные метрики продолжают работать: один сломанный пакет
не ломает все команды.

## Быстрее: пакетный расчёт

`evaluate` считает запросы пачками. Если у метрики есть метод
`batch(batch, k) -> numpy.ndarray` (протокол `ranklens.metrics.vectorized.BatchMetric`),
она получает сразу матрицы релевантностей пачки:

- `batch.relevance[i, r]` — релевантность документа на позиции `r` запроса `i`;
- `batch.ideal[i, j]` — все оценки запроса `i` по убыванию;
- дополнение — нулями, то есть «нерелевантен».

Результат обязан совпадать с поштучным расчётом — проверьте это тестом, как
`tests/metrics/test_vectorized.py`. Метрика без `batch` тоже работает, просто медленнее.

## Как проверить

Минимум — точные значения на примере, посчитанном вручную, и свойства из
`tests/metrics/test_invariants.py`: значение в [0, 1], ноль без релевантных документов,
подъём более релевантного документа выше не ухудшает метрику.
