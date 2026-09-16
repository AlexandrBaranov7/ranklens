# ranklens

[![CI](https://github.com/AlexandrBaranov7/ranklens/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandrBaranov7/ranklens/actions/workflows/ci.yml)

Библиотека и CLI для оффлайн-оценки, корректного сравнения и объяснения
качества ранжирования: посчитал → сравнил статистически честно → понял почему.

> Статус: ранняя разработка (pre-alpha). Публичного API пока нет.

## Установка

```bash
pip install ranklens   # после первого релиза
```

## Разработка

```bash
git clone https://github.com/AlexandrBaranov7/ranklens.git
cd ranklens
poetry install          # Poetry >= 2.4
poetry run ranklens --version
```

Правила разработки — в [CONTRIBUTING.md](CONTRIBUTING.md),
принятые решения — в [docs/decisions.md](docs/decisions.md).

## Лицензия

MIT
