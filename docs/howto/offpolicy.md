# Лог показов из событий

Off-policy оценке нужен лог в простой форме: для каждого показа выдачи — какие
документы на каких позициях были показаны и какую награду получил каждый.
В продакшене так данные почти не лежат. Обычно есть два потока событий:

- **показы**: `request_id`, запрос, выдача (массив `items`) и время;
- **реакции**: `request_id`, документ, событие (`click`, `cart`, `purchase`, …) и время.

`impressions_from_events` склеивает их и не решает ничего молча.

## Из pandas

```python exec="on" source="block" result="text"
import pandas as pd

from ranklens.core import FeedbackScale
from ranklens.offpolicy import impressions_from_events

t = pd.Timestamp("2026-09-26 10:00")
shows = pd.DataFrame(
    {
        "request_id": ["r1", "r2", "r1"],  # r1 доставлен дважды
        "query_id": ["q42", "q7", "q42"],
        "items": [["d7", "d3", "d9"], ["d1", "d2"], ["d7", "d3", "d9"]],
        "ts": [t, t, t + pd.Timedelta("1s")],
    }
)
feedback = pd.DataFrame(
    {
        "request_id": ["r1", "r1", "r1", "r1", "r1", "r2"],
        "doc_id": ["d7", "d7", "d9", "d9", "d5", "d2"],
        "event": ["click", "click", "click", "purchase", "click", "scroll"],
        "ts": [t + pd.Timedelta(s) for s in ["5s", "40s", "70s", "7min", "10s", "3s"]],
    }
)
scale = FeedbackScale.of({"click": 1, "cart": 3, "purchase": 10})

log = impressions_from_events(shows, feedback, scale, window="5min")
print(log)
for impression in log.impressions:
    print(impression.impression_id, impression.docs, impression.rewards)
```

Что здесь произошло:

- **d7 кликнули дважды** — засчитан один клик: несколько событий на документ дают
  одну награду, сильнейшую по шкале.
- **d9: клик через 70 с и покупка через 7 минут.** Окно считается от показа, 5 минут —
  покупка не засчитана, остался клик. Без `window` время не смотрится вовсе.
- **d5 не показывался в r1** — событие отброшено и посчитано: к этой выдаче оно
  не относится (покупка через корзину, переход с другого экрана).
- **scroll** нет в шкале — проигнорирован, счётчик по типу.
- **r1 доставлен дважды** — оставлен самый ранний показ.
- **r2 без реакций** тоже в логе: пустые показы нужны оценщикам.

Выдача может быть и строками — `doc_id` и `position` вместо массива `items`.
Имена колонок меняются через `EventColumns`.

## Большие логи: тот же join в SQL

Если событий миллионы, склеивать удобнее в хранилище, а в Python забирать уже плоскую
таблицу — одна строка на показанный документ. Запрос для Trino повторяет правила
выше; шкала и окно — в `CASE` и `INTERVAL`:

```sql
WITH first_show AS (           -- повторная доставка: самый ранний показ
    SELECT request_id, query_id, items, ts
    FROM (
        SELECT *, row_number() OVER (PARTITION BY request_id ORDER BY ts) AS n
        FROM shows
    )
    WHERE n = 1
),
shown AS (                     -- позиция = индекс в массиве; повтор документа — первая
    SELECT request_id, query_id, doc_id, position, shown_at
    FROM (
        SELECT s.request_id, s.query_id, t.doc_id, t.position, s.ts AS shown_at,
               row_number() OVER (
                   PARTITION BY s.request_id, t.doc_id ORDER BY t.position
               ) AS n
        FROM first_show AS s
        CROSS JOIN UNNEST(s.items) WITH ORDINALITY AS t (doc_id, position)
    )
    WHERE n = 1
),
rewarded AS (                  -- сильнейшее событие в окне от показа
    SELECT f.request_id, f.doc_id,
           max(CASE f.event WHEN 'click' THEN 1 WHEN 'cart' THEN 3
                            WHEN 'purchase' THEN 10 END) AS reward
    FROM feedback AS f
    JOIN shown AS s ON s.request_id = f.request_id AND s.doc_id = f.doc_id
    WHERE f.event IN ('click', 'cart', 'purchase')
      AND f.ts BETWEEN s.shown_at AND s.shown_at + INTERVAL '5' MINUTE
    GROUP BY f.request_id, f.doc_id
)
SELECT s.request_id AS impression_id, s.query_id, s.doc_id, s.position,
       coalesce(r.reward, 0) AS reward
FROM shown AS s
LEFT JOIN rewarded AS r ON r.request_id = s.request_id AND r.doc_id = s.doc_id
ORDER BY s.request_id, s.position
```

Результат читается потоково — из файла или DataFrame (для DataFrame нужен extra `arrow`):

```python
from ranklens.io import iter_clicklog

for impression in iter_clicklog("impressions.parquet"):
    ...
```

!!! warning "Счётчики в SQL не считаются сами"

    `impressions_from_events` сообщает, сколько событий отброшено и почему.
    В SQL это нужно посчитать отдельно (события без показа, вне окна) — большая доля
    отброшенного значит, что ключи или время в потоках не согласованы, и оценке
    по такому логу верить нельзя.
