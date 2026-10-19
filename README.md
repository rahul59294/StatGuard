# StatGuard

[![CI](https://github.com/rahul59294/StatGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/rahul59294/StatGuard/actions/workflows/ci.yml)

A local-first CLI that profiles a CSV, checks statistical assumptions, selects and runs an appropriate test, and explains the result in a reproducible Markdown report. StatGuard is intentionally small and runs locally with no API keys or network calls.

## Setup

StatGuard requires Python 3.12 and [uv](https://docs.astral.sh/uv/). From the project directory:

```bash
uv sync
```

## Run a numeric comparison

Compare the numeric `score` column across the two `group` values:

```bash
uv run statguard analyze samples/demo.csv --group group --value score
```

Write the same analysis to a report file with `--out`:

```bash
uv run statguard analyze samples/demo.csv \
  --group group --value score --out reports/demo.md
```

## Run a categorical association

Test the association between two categorical columns with `--cat1` and `--cat2`:

```bash
uv run statguard analyze samples/demo_cat.csv --cat1 region --cat2 preference
```

## Test

Run the test suite with:

```bash
uv run pytest
```

## How tests are chosen

StatGuard first profiles the data and checks assumptions. For numeric comparisons it uses Shapiro-Wilk normality checks and a Levene equal-variance check:

- Two groups: choose a t-test when normality and equal variance are supported, Welch's t-test when normality is supported but variances differ, or Mann-Whitney when normality is not supported.
- Three or more groups: choose ANOVA when the data meet the parametric assumptions, otherwise Kruskal-Wallis.
- Categorical association: choose chi-square when expected counts support it, otherwise Fisher's exact test.

Each report includes the reason for the choice and the exact code or SciPy call needed to reproduce it. See the included [demo report](samples/report_demo.md).

## Docker

Build the image from the project root:

```bash
docker build -t statguard .
```

Run the default numeric demo analysis:

```bash
docker run --rm statguard
```

To run another command in the image, keep the `uv run` entry point. For example:

```bash
docker run --rm statguard uv run statguard analyze samples/demo_cat.csv --cat1 region --cat2 preference
```

## What StatGuard does

1. Profiles column types, missing values, outliers, and duplicate rows.
2. Checks the assumptions relevant to the selected analysis.
3. Chooses and runs the appropriate statistical test.
4. Writes a Markdown report containing the result, selection reason, and reproduction call.

## Configuration

Copy `.env.example` to `.env` if you want a local configuration file:

```bash
cp .env.example .env
```

StatGuard currently needs no environment variables or secrets. The `.env` file is ignored by Git and is reserved for future local settings.

## Built with OmniRush

Built solo during Frontier Buildathon 2026. Sessions: <fill in>
