# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование — [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

### Added

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
