# ranklens

[![CI](https://github.com/AlexandrBaranov7/ranklens/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandrBaranov7/ranklens/actions/workflows/ci.yml)

Библиотека и CLI для оффлайн-оценки, корректного сравнения и объяснения
качества ранжирования: посчитал → сравнил статистически честно → понял почему.

> Статус: ранняя разработка (pre-alpha). Публичного API пока нет.

## Установка для разработки

```bash
git clone https://github.com/AlexandrBaranov7/ranklens.git
cd ranklens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ranklens --version
```

Правила разработки — в [CONTRIBUTING.md](CONTRIBUTING.md),
принятые решения — в [docs/decisions.md](docs/decisions.md).

## Лицензия

MIT
