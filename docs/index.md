# ranklens

Модель поменяли — стало ли лучше, **где** и **почему**? ranklens отвечает на это из
таблиц pandas одной строкой: разница с интервалом, p- и q-value, минимальный заметный
эффект (MDE), разбивка по сегментам с поправкой на множественные сравнения. Метрики
совпадают с trec_eval до 1e-9.

```python
import ranklens as rl

rl.compare(current, new, qrels, ["ndcg@10"], by="device")
```

## Типичные задачи

Примеры ниже выполняются при сборке этой страницы на синтетическом каталоге: запросы,
товары и оценки выдуманы, но устроены как в настоящем поиске
(генератор — `docs/examples/catalog.py`).

```python exec="on" session="cases"
import sys

import pandas as pd

sys.path.insert(0, "docs/examples")
pd.set_option("display.width", 120)
```

Данные — обычная таблица с одной строкой на пару «запрос — товар»: оценка асессора,
признаки и скоры моделей.

```python exec="on" session="cases" source="block" result="text"
import ranklens as rl
from catalog import searches  # синтетический каталог

df = searches()
print(df[["query", "product", "grade", "history", "base_s0", "new_s0"]].head(5).round(2))
```

### Эффект виден только в сегменте

Новая модель знает, покупал ли пользователь товар раньше. Такой признак есть только
у пользователей с историей покупок — это около 9% запросов. У каждой модели два сида
обучения: они усредняются по запросу, чтобы шум обучения не выдавался за эффект.

```python exec="on" session="cases" source="block" result="text"
qrels = rl.qrels(df, query="query", doc="product", relevance="grade")


def run(score):
    return rl.run(df, query="query", doc="product", score=score, segments="history")


base, new = [run("base_s0"), run("base_s1")], [run("new_s0"), run("new_s1")]

columns = ["n_queries", "delta", "ci_low", "ci_high", "mde", "q_value"]
print(rl.compare(base, new, qrels, ["ndcg@10"]).table[columns].round(4), "\n")
print(rl.compare(base, new, qrels, ["ndcg@10"], by="history").table[columns].round(4))
```

В среднем прирост не отличим от нуля и меньше MDE — минимального эффекта, который эти
запросы вообще способны показать. Разбивка находит, где он: там, где признак есть,
прирост значимый и после поправки на число сегментов (q-value), а на остальных
запросах потерь нет.

### Две версии выдачи размечены разными партиями асессоров

Новая версия поиска находит больше подходящих товаров. Каждую выдачу оценивала своя
партия асессоров — сначала проверяем, согласны ли они между собой.

```python exec="on" session="cases" source="block" result="text"
from catalog import two_versions

current_df, new_df = two_versions()
batch_cur = rl.qrels(current_df, query="query", doc="product", relevance="grade")
batch_new = rl.qrels(new_df, query="query", doc="product", relevance="grade")
print(rl.agreement(batch_cur, batch_new), "\n")

both = rl.merge_qrels(batch_cur, batch_new)  # одна разметка для обеих сторон
current = rl.run(current_df, query="query", doc="product", rank="pos")
new_version = rl.run(new_df, query="query", doc="product", rank="pos")
result = rl.compare(current, new_version, both, ["ndcg@5", "ndcg@10", "err@10"])
print(result.table[columns].round(4))
```

Вторая партия ставит оценки мягче, и на «своих» оценках эта разница смешалась бы
с разницей версий — поэтому сравниваем на общей разметке. NDCG@10 значимо выше, а
NDCG@5 и ERR@10, которые смотрят в основном на верх выдачи, — нет: новые товары
улучшают позиции 6–10, первые карточки почти не изменились. Какая метрика главная, решает
то, сколько карточек пользователи реально просматривают.

### Почему: вклад признаков в метрику

Модель — любая: CatBoost, LightGBM, XGBoost, scikit-learn, PyTorch, ONNX или функция
скоринга, как здесь.

```python exec="on" session="cases" source="block" result="text"
from catalog import FEATURES, new_model

from ranklens.explain import CallableAdapter

model = CallableAdapter(new_model, FEATURES)
ex = rl.explainer(model, df, query="query", doc="product", relevance="grade")
print(ex.importance("ndcg@10", n_repeats=3).round(4), "\n")  # если перемешать признак
print(ex.worst(3)[["ndcg@10", "cause 1"]].round(3))  # худшие запросы и что их подвело
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
