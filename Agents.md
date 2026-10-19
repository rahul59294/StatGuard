# StatGuard

Local-first CLI: takes a CSV, profiles it, checks assumptions, picks the right
statistical test, runs it, exports a reproducible Markdown report.

## Stack
- Python 3.12 (pinned in `.python-version`), managed with `uv` (`uv.lock` committed)
- pandas, scipy, typer, pytest. No network calls, no API keys.

## Layout
src/statguard/{profile.py, assumptions.py, select.py, run.py, report.py, cli.py}
tests/ (small sample CSVs in tests/data/)
samples/ (demo CSVs for README)

## Commands
- Run: `uv run statguard analyze samples/demo.csv --group <col> --value <col>`
- Test: `uv run pytest`

## Rules
- Relative paths only. Settings via `.env.example`.
- Every report includes the exact code/call used and the reason a test was chosen.
- Keep scope small: two-group numeric comparison, multi-group comparison,
  and categorical association. Nothing else.
- Test selection: normality (Shapiro) + equal variance (Levene) ->
  t-test / Welch / Mann-Whitney (2 groups), ANOVA / Kruskal-Wallis (3+),
  chi-square / Fisher (categorical).
- Short, readable code. Update README when behavior changes.
