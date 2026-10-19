# StatGuard

A local-first CLI that picks the right statistical test for your CSV, explains why, and exports a reproducible report.

## Setup
    uv sync

## Run
    uv run statguard analyze samples/demo.csv --group group --value score

## Test
    uv run pytest

## Docker
    docker build -t statguard .
    docker run --rm statguard

## What it does
1. Profiles the data (types, missing values, outliers, duplicates)
2. Checks assumptions (normality, equal variance)
3. Chooses and runs the appropriate test
4. Writes a Markdown report with the exact code used

## Configuration
Copy `.env.example` to `.env` (no secrets required).

## Built with OmniRush
Built solo during Frontier Buildathon 2026. Sessions: <fill in>
