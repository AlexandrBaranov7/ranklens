# ruff: noqa: RUF001  # Cyrillic goods names, not confusable Latin
"""Synthetic catalog search for the docs: the shape of real data, made-up goods and brands.

``searches()`` — one row per (query, product) with a grade, features and the scores of two
models trained with two seeds each; the new model also uses ``bought_before``, which is
set only for users with a purchase history. ``two_versions()`` — top-10 of two search
versions, each judged by its own batch of assessors.
"""

import numpy as np
import pandas as pd

GOODS = {
    "молоко": [
        "Молоко Луговое 3,2% 1 л",
        "Молоко Северная ферма 2,5% 930 мл",
        "Молоко безлактозное Утро 1,5% 1 л",
        "Молоко топлёное Луговое 4% 950 мл",
        "Молоко овсяное Зерно 1 л",
        "Коктейль молочный Утро клубника",
        "Кефир Луговое 2,5% 1 л",
        "Сливки Северная ферма 10% 500 мл",
        "Сгущёнка Утро 380 г",
        "Йогурт питьевой Луговое 270 г",
        "Ряженка Северная ферма 4% 500 мл",
        "Печенье овсяное к молоку 300 г",
    ],
    "кофе в зёрнах": [
        "Кофе в зёрнах Арабика Бразилия 1 кг",
        "Кофе в зёрнах Эспрессо-бленд 500 г",
        "Кофе в зёрнах Колумбия Супремо 250 г",
        "Кофе в зёрнах без кофеина 500 г",
        "Кофе в зёрнах Робуста 1 кг",
        "Кофе молотый Арабика 250 г",
        "Кофе растворимый Утро 95 г",
        "Кофемолка ручная керамическая",
        "Кофе в капсулах Лунго 10 шт",
        "Сливки для кофе 10% 200 мл",
        "Сироп ванильный для кофе",
        "Чай чёрный листовой 100 г",
    ],
    "корм для кошек": [
        "Корм сухой для кошек с курицей 1,5 кг",
        "Корм влажный для кошек лосось 85 г",
        "Корм для стерилизованных кошек 400 г",
        "Корм сухой для котят 800 г",
        "Корм влажный для кошек паштет 12 шт",
        "Лакомство для кошек подушечки",
        "Наполнитель древесный 10 л",
        "Корм сухой для собак 2 кг",
        "Миска керамическая для кошек",
        "Витамины для кошек 60 шт",
        "Когтеточка картонная",
        "Игрушка для кошек мышь",
    ],
    "шампунь": [
        "Шампунь для всех типов волос 400 мл",
        "Шампунь против перхоти 250 мл",
        "Шампунь для окрашенных волос 300 мл",
        "Шампунь детский без слёз 200 мл",
        "Шампунь-кондиционер 2 в 1 400 мл",
        "Бальзам-ополаскиватель 300 мл",
        "Маска для волос восстанавливающая",
        "Гель для душа морской 250 мл",
        "Шампунь сухой 200 мл",
        "Мыло жидкое 500 мл",
        "Расчёска массажная",
        "Шапочка для душа",
    ],
    "детское пюре": [
        "Пюре яблоко с 4 месяцев 100 г",
        "Пюре груша с 4 месяцев 100 г",
        "Пюре тыква с 5 месяцев 80 г",
        "Пюре мясное индейка 80 г",
        "Пюре фруктовое ассорти 90 г",
        "Каша гречневая безмолочная 200 г",
        "Сок яблочный детский 200 мл",
        "Печенье детское 150 г",
        "Пюре овощное брокколи 80 г",
        "Вода детская 1,5 л",
        "Ложка детская силиконовая",
        "Нагрудник непромокаемый",
    ],
    "оливковое масло": [
        "Масло оливковое Extra Virgin 500 мл",
        "Масло оливковое Extra Virgin 1 л",
        "Масло оливковое рафинированное 1 л",
        "Масло оливковое для жарки 750 мл",
        "Масло оливковое нефильтрованное 250 мл",
        "Масло подсолнечное 1 л",
        "Оливки без косточки 300 г",
        "Уксус бальзамический 250 мл",
        "Масло кунжутное 250 мл",
        "Спрей масло для жарки",
        "Маслины чёрные 300 г",
        "Соус песто 190 г",
    ],
    "яблоки": [
        "Яблоки Гала 1 кг",
        "Яблоки Голден 1 кг",
        "Яблоки Гренни Смит 1 кг",
        "Яблоки сезонные 2 кг",
        "Яблоки Семеренко 1 кг",
        "Груши Конференция 1 кг",
        "Сок яблочный прямого отжима 1 л",
        "Чипсы яблочные 50 г",
        "Пюре яблочное 100 г",
        "Бананы 1 кг",
        "Повидло яблочное 350 г",
        "Сидр безалкогольный 0,5 л",
    ],
    "сыр моцарелла": [
        "Сыр моцарелла в рассоле 125 г",
        "Сыр моцарелла для пиццы 200 г",
        "Сыр моцарелла мини 150 г",
        "Сыр моцарелла тёртый 150 г",
        "Сыр моцарелла буррата 150 г",
        "Сыр пармезан 200 г",
        "Сыр рикотта 250 г",
        "Сыр сулугуни 300 г",
        "Сыр плавленый 90 г",
        "Тесто для пиццы 400 г",
        "Соус томатный для пиццы",
        "Сыр брынза 200 г",
    ],
}
REFINEMENTS = [
    "", " купить", " недорого", " акция", " доставка", " большая упаковка", " отзывы",
    " цена", " скидка", " выгодно", " быстро", " рядом", " сегодня", " набор", " новинка",
    " хит", " вкусно", " натуральный", " для семьи", " оптом", " онлайн", " каталог",
    " популярное", " лучшее", " выбор", " по акции", " на неделю", " в наличии", " 2 шт",
    " 3 шт", " маленький", " большой", " premium", " эконом", " бюджетно", " срочно",
    " на завтра", " хороший", " свежий", " рекомендуем",
]  # fmt: skip
FEATURES = ["text_match", "popularity", "price_rank", "bought_before"]


def searches(seed: int = 7, history_share: float = 0.1) -> pd.DataFrame:
    """Grades, features and scores of ``base_s0/1`` and ``new_s0/1`` for 320 queries."""
    rng = np.random.default_rng(seed)
    rows = []
    for base, goods in GOODS.items():
        for refinement in REFINEMENTS:
            query = base + refinement
            history = rng.random() < history_share
            for i, product in enumerate(goods):
                exact = i < 5
                grade = int(np.clip(rng.normal(2.4 if exact else 0.8, 0.7), 0, 3).round())
                bought = int(history and grade >= 2 and rng.random() < 0.6)
                rows.append(
                    {
                        "query": query,
                        "product": product,
                        "grade": grade,
                        "history": "есть" if history else "нет",
                        "text_match": grade / 3 + rng.normal(0, 0.45),
                        "popularity": rng.lognormal(0, 1) + 0.15 * grade,
                        "price_rank": rng.random(),
                        "bought_before": bought,
                    }
                )
    df = pd.DataFrame(rows)
    base = 1.0 * df.text_match + 0.08 * df.popularity - 0.1 * df.price_rank
    for s in (0, 1):
        noise = np.random.default_rng([seed, s])
        df[f"base_s{s}"] = base + noise.normal(0, 0.08, len(df))
        df[f"new_s{s}"] = base + 0.9 * df.bought_before + noise.normal(0, 0.08, len(df))
    return df


def new_model(features: np.ndarray) -> np.ndarray:
    """The scoring function of the new model, as `rl.explainer` sees it."""
    text_match, popularity, price_rank, bought_before = features.T
    return 1.0 * text_match + 0.08 * popularity - 0.1 * price_rank + 0.9 * bought_before


def two_versions(seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-10 of the current and the new search version, each with its batch's grades.

    The new version finds more good products, which land in positions 6-10; the top-5
    stays as it was up to a swap of neighbours. The second batch of assessors is milder.
    """
    rng = np.random.default_rng(seed)
    current, new = [], []
    for base, goods in GOODS.items():
        for refinement in REFINEMENTS:
            query = base + refinement
            truth = {
                p: int(np.clip(rng.normal(2.3 if i < 5 else 0.7, 0.8), 0, 3).round())
                for i, p in enumerate(goods)
            }
            order = sorted(goods, key=lambda p: -truth[p] - rng.normal(0, 1.1))
            top, new_top = order[:5], order[:5]
            if rng.random() < 0.5:  # the new version swaps two neighbours in the top-5
                j = int(rng.integers(0, 4))
                new_top = [*top[:j], top[j + 1], top[j], *top[j + 2 :]]
            tail = order[5:]
            better_tail = sorted(tail, key=lambda p: -truth[p] - rng.normal(0, 0.9))
            n_cur = int(rng.integers(7, 11))
            for pos, product in enumerate(top + tail[: n_cur - 5], start=1):
                current.append((query, product, pos, truth[product]))
            for pos, product in enumerate(new_top + better_tail[:5], start=1):
                milder = min(3, truth[product] + int(rng.random() < 0.15))
                new.append((query, product, pos, milder))
    columns = ["query", "product", "pos", "grade"]
    return pd.DataFrame(current, columns=columns), pd.DataFrame(new, columns=columns)
