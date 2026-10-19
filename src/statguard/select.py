"""Selection rules for StatGuard's supported statistical tests."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


def select_numeric_test(assumptions: dict[str, Any]) -> dict[str, str]:
    """Choose a numeric test from :func:`check_numeric_assumptions` output.

    The returned dictionary has a canonical ``test`` name and a one-sentence
    ``reason`` suitable for inclusion in a report.
    """
    group_sizes = assumptions.get("group_sizes", {})
    n_groups = len(group_sizes)
    shapiro = assumptions.get("shapiro", assumptions.get("normality", {}))
    levene = assumptions.get("levene", {})

    all_normal = bool(shapiro) and all(
        details.get("normal") is True and details.get("p_value") is not None
        for details in shapiro.values()
    )
    equal_variance = levene.get("equal_variance") is True and levene.get("p_value") is not None

    if n_groups == 2:
        if all_normal and equal_variance:
            return {
                "test": "t-test",
                "reason": "Both groups appear normally distributed and have equal variance, so a pooled two-sample t-test is appropriate.",
            }
        if all_normal:
            return {
                "test": "Welch",
                "reason": "Both groups appear normally distributed but their variances are unequal, so Welch's t-test is appropriate.",
            }
        return {
            "test": "Mann-Whitney",
            "reason": "At least one group does not meet the normality check, so the non-parametric Mann-Whitney test is appropriate.",
        }

    if n_groups >= 3:
        if all_normal and equal_variance:
            return {
                "test": "ANOVA",
                "reason": "All groups appear normally distributed with equal variance, so a one-way ANOVA is appropriate.",
            }
        return {
            "test": "Kruskal-Wallis",
            "reason": "The normality or equal-variance assumptions are not met for all groups, so the non-parametric Kruskal-Wallis test is appropriate.",
        }

    return {
        "test": "Mann-Whitney",
        "reason": "There are fewer than two groups, so no valid between-group comparison can be made; Mann-Whitney is the safest numeric fallback.",
    }


def select_categorical_test(frame: pd.DataFrame, cat1: str, cat2: str) -> dict[str, str]:
    """Choose chi-square or Fisher's exact test for two categorical columns."""
    _validate_columns(frame, cat1, cat2)
    clean = frame[[cat1, cat2]].dropna()
    table = _contingency_table(clean, cat1, cat2)

    expected_low = False
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        try:
            expected = stats.chi2_contingency(table, correction=False)[3]
            expected_low = bool((expected < 5).any())
        except ValueError:
            expected_low = True

    if table.shape == (2, 2) and expected_low:
        return {
            "test": "Fisher",
            "reason": "The contingency table is 2x2 and at least one expected count is below 5, so Fisher's exact test is appropriate.",
        }
    return {
        "test": "chi-square",
        "reason": "A chi-square test of independence is appropriate for this categorical contingency table.",
    }


def _validate_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Column(s) not found: {', '.join(missing)}")


def _contingency_table(frame: pd.DataFrame, cat1: str, cat2: str) -> pd.DataFrame:
    rows = list(pd.unique(frame[cat1]))
    columns = list(pd.unique(frame[cat2]))
    return pd.crosstab(frame[cat1], frame[cat2]).reindex(index=rows, columns=columns, fill_value=0)
