# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование — [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

### Added

- Векторизованный расчёт метрик пачками (`BatchMetric`, `evaluate(batch_size=...)`):
  в 7.5 раза быстрее поштучного на 2 000 запросах; бенчмарки и неблокирующая задача `bench` в CI.
- `iter_run(order="score")`: сортировка по score с разрешением равенств как в trec_eval;
  для TREC-файлов — по умолчанию.
- Golden-тесты: `ndcg`, `map`, `mrr` совпадают с trec_eval (pytrec_eval) по каждому запросу
  до 1e-9; генератор эталона `scripts/make_golden.py`.
- `ranklens.metrics.rbo`: rank-biased overlap двух ранжирований без разметки.
- Метрики `err` (шкала `max_rel`, по умолчанию 4, как в gdeval) и `rbp` (параметр `p`,
  бинарная или градуированная релевантность через `max_rel`).
- `ranklens eval --run RUN --qrels QRELS --metrics SPEC...`: таблица или JSON,
  коды выхода 0/1/2/3; `ranklens.metrics.evaluate` и результаты `Evaluation`, `MetricResult`.
- Метрики `ndcg`, `map`, `mrr` со значениями по умолчанию как в trec_eval;
  `ndcg(gain=exp)` — вариант LTR-библиотек, `map(rel=2)` — порог релевантности.
- `ranklens.core.Metric` — протокол метрики; `ranklens.core.registry`: реестр метрик
  и спецификации вида `ndcg(gain=exp)@10`; исключение `MetricSpecError`.
- Чтение parquet и Feather, а также таблиц pandas, polars и pyarrow напрямую
  (`iter_run(df)`, `read_qrels(df)`) через Arrow; extras `arrow`.
  Исключение `MissingDependencyError` для отсутствующих необязательных зависимостей.
- `ranklens.io.read_qrels`: загрузка разметки в любом порядке строк с проверкой
  повторных оценок.
- `ranklens.io.iter_run`: потоковое чтение прогона по одному `RankedList` на запрос;
  порядок документов по файлу или по `rank`, проверка группировки или сортировки
  по query_id, strict / non-strict режимы.
- `ranklens.io.formats`: чтение csv, tsv, jsonl и TREC (в том числе `.gz`, `.bz2`, `.xz`),
  определение формата по расширению.
- `ranklens.io.schema`: `RunSchema`, `QrelsSchema` — имена колонок и типизированный
  разбор строк с понятными ошибками.
- `ranklens.io.ErrorCollector`: накопитель ошибок данных для non-strict чтения
  (`record`, `snapshot`) с ограниченным числом примеров на тип ошибки;
  `ranklens.core.ErrorSummary` — неизменяемый снимок для CLI и отчётов.
- Исключения `UngroupedInputError`, `MissingColumnError`; предупреждения
  `RankLensWarning`, `SkippedRowsWarning`.
- `ranklens.core`: `RankedList`, `QueryId`/`DocId`, `Qrels`, `SegmentKey`
  и иерархия исключений `RankLensError`.
- Архитектурный тест: границы слоёв из CONTRIBUTING проверяются по AST пакета.
- Каркас проекта: src-layout, Poetry с lock-файлом, CLI `ranklens --version`.
- Инструменты: ruff, mypy strict, pytest с гейтом покрытия 85%, pre-commit.
- CI: lint, types, тесты на Python 3.11–3.14 × Linux/macOS/Windows, установка wheel.
- CONTRIBUTING, Code of Conduct, SECURITY, шаблоны issue и PR.
