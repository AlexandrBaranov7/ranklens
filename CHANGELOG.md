# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование — [SemVer](https://semver.org/lang/ru/).

## [Unreleased]

### Added

- `ranklens.io.ErrorCollector`: счётчик ошибок данных для non-strict чтения
  с ограниченным числом примеров на тип ошибки.
- Исключения `UngroupedInputError`, `MissingColumnError`; предупреждения
  `RankLensWarning`, `SkippedRowsWarning`.
- `ranklens.core`: `RankedList`, `QueryId`/`DocId`, `Qrels`, `SegmentKey`
  и иерархия исключений `RankLensError`.
- Архитектурный тест: границы слоёв из CONTRIBUTING проверяются по AST пакета.
- Каркас проекта: src-layout, Poetry с lock-файлом, CLI `ranklens --version`.
- Инструменты: ruff, mypy strict, pytest с гейтом покрытия 85%, pre-commit.
- CI: lint, types, тесты на Python 3.11–3.14 × Linux/macOS/Windows, установка wheel.
- CONTRIBUTING, Code of Conduct, SECURITY, шаблоны issue и PR.
