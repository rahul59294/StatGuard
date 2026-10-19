"""Execute the statistical tests selected by StatGuard."""

from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd
from scipy import stats


def run_numeric_test(
    frame: pd.DataFrame, group_col: str, value_col: str, test_name: str
) -> dict[str, Any]:
    """Run a supported numeric test and return reproducible result metadata."""
    _validate_columns(frame, group_col, value_col)
    clean = frame[[group_col, value_col]].dropna()
    groups = _group_values(clean, group_col)
    values = [clean.loc[clean[group_col] == group, value_col].astype(float).to_numpy() for group in groups]
    group_sizes = {group: int(len(values[index])) for index, group in enumerate(groups)}
    canonical = _numeric_test_name(test_name)

    statistic: float | None = None
    p_value: float | None = None
    if canonical == "t-test":
        if len(values) != 2:
            raise ValueError("A t-test requires exactly two groups.")
        result = stats.ttest_ind(values[0], values[1], equal_var=True)
        statistic, p_value = _scipy_values(result)
        call = "scipy.stats.ttest_ind(group_1, group_2, equal_var=True)"
        effect_size = _cohens_d(values[0], values[1])
        effect_name = "Cohen's d"
    elif canonical == "Welch":
        if len(values) != 2:
            raise ValueError("Welch's t-test requires exactly two groups.")
        result = stats.ttest_ind(values[0], values[1], equal_var=False)
        statistic, p_value = _scipy_values(result)
        call = "scipy.stats.ttest_ind(group_1, group_2, equal_var=False)"
        effect_size = _cohens_d(values[0], values[1])
        effect_name = "Cohen's d"
    elif canonical == "Mann-Whitney":
        if len(values) != 2:
            raise ValueError("Mann-Whitney requires exactly two groups.")
        result = stats.mannwhitneyu(values[0], values[1], alternative="two-sided")
        statistic, p_value = _scipy_values(result)
        call = "scipy.stats.mannwhitneyu(group_1, group_2, alternative='two-sided')"
        effect_size = _rank_biserial(values[0], values[1], statistic)
        effect_name = "rank-biserial correlation"
    elif canonical == "ANOVA":
        if len(values) < 2:
            raise ValueError("ANOVA requires at least two groups.")
        result = stats.f_oneway(*values)
        statistic, p_value = _scipy_values(result)
        call = "scipy.stats.f_oneway(*groups)"
        effect_size = _eta_squared(values)
        effect_name = "eta-squared"
    else:  # Kruskal-Wallis
        if len(values) < 2:
            raise ValueError("Kruskal-Wallis requires at least two groups.")
        result = stats.kruskal(*values)
        statistic, p_value = _scipy_values(result)
        call = "scipy.stats.kruskal(*groups)"
        effect_size = _epsilon_squared(statistic, len(values), sum(len(group) for group in values))
        effect_name = "epsilon-squared"

    return {
        "test": canonical,
        "statistic": statistic,
        "p_value": p_value,
        "pvalue": p_value,
        "group_sizes": group_sizes,
        "effect_size": effect_size,
        "effect_size_name": effect_name,
        "scipy_call": call,
        "call": call,
    }


def run_categorical_test(
    frame: pd.DataFrame, cat1: str, cat2: str, test_name: str
) -> dict[str, Any]:
    """Run chi-square or Fisher's exact test on two categorical columns."""
    _validate_columns(frame, cat1, cat2)
    clean = frame[[cat1, cat2]].dropna()
    table = _contingency_table(clean, cat1, cat2)
    canonical = _categorical_test_name(test_name)
    group_sizes = {
        cat1: {category: int(table.loc[category].sum()) for category in table.index},
        cat2: {category: int(table[category].sum()) for category in table.columns},
    }

    chi2_statistic: float | None = None
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        try:
            chi2_statistic = _number(stats.chi2_contingency(table, correction=False)[0])
        except ValueError:
            chi2_statistic = None

    if canonical == "Fisher":
        if table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 contingency table.")
        result = stats.fisher_exact(table.to_numpy())
        statistic, p_value = _scipy_values(result)
        call = "scipy.stats.fisher_exact(contingency_table)"
    else:
        result = stats.chi2_contingency(table, correction=False)
        statistic, p_value = _number(result[0]), _number(result[1])
        call = "scipy.stats.chi2_contingency(contingency_table, correction=False)"

    return {
        "test": canonical,
        "statistic": statistic,
        "p_value": p_value,
        "pvalue": p_value,
        "group_sizes": group_sizes,
        "effect_size": _cramers_v(chi2_statistic, table.shape, int(table.to_numpy().sum())),
        "effect_size_name": "Cramér's V",
        "scipy_call": call,
        "call": call,
    }


def _validate_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Column(s) not found: {', '.join(missing)}")


def _group_values(frame: pd.DataFrame, group_col: str) -> list[Any]:
    return list(pd.unique(frame[group_col]))


def _contingency_table(frame: pd.DataFrame, cat1: str, cat2: str) -> pd.DataFrame:
    rows = list(pd.unique(frame[cat1]))
    columns = list(pd.unique(frame[cat2]))
    return pd.crosstab(frame[cat1], frame[cat2]).reindex(index=rows, columns=columns, fill_value=0)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _scipy_values(result: Any) -> tuple[float | None, float | None]:
    return _number(result.statistic), _number(result.pvalue)


def _numeric_test_name(name: str) -> str:
    normalized = name.strip().lower().replace("’", "'")
    aliases = {
        "t-test": "t-test", "t test": "t-test", "ttest": "t-test", "student's t-test": "t-test",
        "welch": "Welch", "welch's t-test": "Welch", "welch t-test": "Welch", "welch's": "Welch",
        "mann-whitney": "Mann-Whitney", "mann-whitney u": "Mann-Whitney", "mann whitney": "Mann-Whitney", "mann whitney u": "Mann-Whitney",
        "anova": "ANOVA", "one-way anova": "ANOVA",
        "kruskal-wallis": "Kruskal-Wallis", "kruskal wallis": "Kruskal-Wallis", "kruskal": "Kruskal-Wallis",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported numeric test: {name}")
    return aliases[normalized]


def _categorical_test_name(name: str) -> str:
    normalized = name.strip().lower().replace("’", "'")
    if normalized in {"fisher", "fisher's exact", "fisher exact", "fisher's exact test", "fisher exact test"}:
        return "Fisher"
    if normalized in {"chi-square", "chi square", "chi2", "chi-squared"}:
        return "chi-square"
    raise ValueError(f"Unsupported categorical test: {name}")


def _cohens_d(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        return 0.0
    pooled_variance = ((len(first) - 1) * np.var(first, ddof=1) + (len(second) - 1) * np.var(second, ddof=1)) / (len(first) + len(second) - 2)
    if pooled_variance <= 0 or not math.isfinite(float(pooled_variance)):
        return 0.0
    return float((np.mean(first) - np.mean(second)) / math.sqrt(float(pooled_variance)))


def _rank_biserial(first: np.ndarray, second: np.ndarray, u_statistic: float | None) -> float:
    denominator = len(first) * len(second)
    if not denominator or u_statistic is None:
        return 0.0
    return float(2 * u_statistic / denominator - 1)


def _eta_squared(groups: list[np.ndarray]) -> float:
    total = np.concatenate(groups)
    grand_mean = float(np.mean(total))
    between = sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
    total_sum = float(np.sum((total - grand_mean) ** 2))
    return float(between / total_sum) if total_sum > 0 else 0.0


def _epsilon_squared(statistic: float | None, n_groups: int, n_total: int) -> float:
    denominator = n_total - n_groups
    if statistic is None or denominator <= 0:
        return 0.0
    return float(max(0.0, min(1.0, (statistic - n_groups + 1) / denominator)))


def _cramers_v(statistic: float | None, shape: tuple[int, int], n_total: int) -> float:
    denominator = n_total * min(shape[0] - 1, shape[1] - 1)
    if statistic is None or denominator <= 0:
        return 0.0
    return float(math.sqrt(max(0.0, statistic / denominator)))
