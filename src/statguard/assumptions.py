"""Assumption checks used to choose numeric statistical tests."""

from __future__ import annotations

from typing import Any
import warnings

import pandas as pd
from scipy import stats


def _validate_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Column(s) not found: {', '.join(missing)}")


def _number(value: Any) -> float | None:
    """Convert a SciPy scalar to a normal float, preserving unusable values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def check_numeric_assumptions(
    frame: pd.DataFrame, group_col: str, value_col: str
) -> dict[str, Any]:
    """Check normality within groups and equal variance between groups.

    Rows missing either the grouping or value column are excluded.  Group order
    follows the first appearance in the input frame.  Shapiro-Wilk is skipped
    for groups with fewer than three observations and the returned ``notes``
    explain that decision in plain English.
    """
    _validate_columns(frame, group_col, value_col)
    clean = frame[[group_col, value_col]].dropna()
    groups = _group_values(clean, group_col)
    values = [clean.loc[clean[group_col] == group, value_col].astype(float) for group in groups]
    group_sizes = {group: int(len(group_values)) for group, group_values in zip(groups, values)}

    notes: list[str] = []
    shapiro: dict[Any, dict[str, Any]] = {}
    for group, group_values in zip(groups, values):
        n = len(group_values)
        if n < 3:
            note = f"Normality was not checked for group {group!r} because it has fewer than 3 observations."
            shapiro[group] = {
                "n": n,
                "statistic": None,
                "p_value": None,
                "normal": None,
                "note": note,
            }
            notes.append(note)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stats.shapiro(group_values)
        statistic = _number(result.statistic)
        p_value = _number(result.pvalue)
        normal = p_value is not None and p_value >= 0.05
        if p_value is not None:
            status = "consistent with normality" if normal else "not consistent with normality"
            note = f"Shapiro-Wilk for group {group!r}: the values are {status} at the 0.05 level."
        else:
            note = f"Shapiro-Wilk could not determine normality for group {group!r}."
        if p_value is None:
            notes.append(note)
        else:
            notes.append(note)
        shapiro[group] = {
            "n": n,
            "statistic": statistic,
            "p_value": p_value,
            "normal": normal,
            "note": note,
        }

    if len(values) < 2:
        levene = {
            "statistic": None,
            "p_value": None,
            "equal_variance": None,
            "note": "Equal variance was not checked because at least two groups are required.",
        }
        notes.append(levene["note"])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stats.levene(*values, center="median")
        statistic = _number(result.statistic)
        p_value = _number(result.pvalue)
        equal_variance = None if p_value is None else p_value >= 0.05
        if p_value is None:
            note = "Levene's test could not determine whether the group variances are equal."
        elif equal_variance:
            note = "Levene's test found no evidence that the group variances differ at the 0.05 level."
        else:
            note = "Levene's test found evidence that the group variances differ at the 0.05 level."
        notes.append(note)
        levene = {
            "statistic": statistic,
            "p_value": p_value,
            "equal_variance": equal_variance,
            "note": note,
        }

    # ``normality`` is the descriptive name; ``shapiro`` is retained as a
    # convenient, explicit alias for callers that want the test name.
    return {
        "group_sizes": group_sizes,
        "shapiro": shapiro,
        "normality": shapiro,
        "levene": levene,
        "notes": notes,
    }


def _group_values(frame: pd.DataFrame, group_col: str) -> list[Any]:
    """Return non-null group labels in stable first-seen order."""
    return list(pd.unique(frame[group_col]))
