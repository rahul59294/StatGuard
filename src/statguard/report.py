"""Render reproducible Markdown reports for StatGuard analyses."""

from __future__ import annotations

import math
import os
import shlex
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from .assumptions import check_numeric_assumptions
from .profile import profile
from .run import run_categorical_test, run_numeric_test
from .select import select_categorical_test, select_numeric_test


def build_report(csv_path: str | Path, mode: str, params: dict[str, str]) -> str:
    """Build a Markdown report for one supported StatGuard analysis.

    ``params`` uses ``{"group": ..., "value": ...}`` in numeric mode and
    ``{"cat1": ..., "cat2": ...}`` in categorical mode.  The CSV is loaded
    here so the report and its reproduction code use the same source data.
    """
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"numeric", "categorical"}:
        raise ValueError("mode must be 'numeric' or 'categorical'")

    frame = pd.read_csv(csv_path)
    summary = profile(csv_path)
    lines = ["## Dataset summary", "", _summary_table(summary), ""]

    if normalized_mode == "numeric":
        group_col = _required_param(params, "group")
        value_col = _required_param(params, "value")
        assumptions = check_numeric_assumptions(frame, group_col, value_col)
        selection = select_numeric_test(assumptions)
        result = run_numeric_test(frame, group_col, value_col, selection["test"])
        lines.extend(["## Assumption checks", "", _numeric_assumptions_table(assumptions), ""])
        reproduce = _numeric_reproduction(csv_path, group_col, value_col, result["scipy_call"])
        command = _cli_command(csv_path, "numeric", group=group_col, value=value_col)
        interpretation_columns = (group_col, value_col)
        subject = "difference between the groups"
    else:
        cat1 = _required_param(params, "cat1")
        cat2 = _required_param(params, "cat2")
        selection = select_categorical_test(frame, cat1, cat2)
        result = run_categorical_test(frame, cat1, cat2, selection["test"])
        lines.extend(["## Assumption checks", "", _categorical_assumptions_table(frame, cat1, cat2), ""])
        reproduce = _categorical_reproduction(csv_path, cat1, cat2, result["scipy_call"])
        command = _cli_command(csv_path, "categorical", cat1=cat1, cat2=cat2)
        interpretation_columns = (cat1, cat2)
        subject = "association between the categorical variables"

    lines.extend(
        [
            "## Chosen test and reason",
            "",
            f"- **Name:** {selection['test']}",
            f"- **Reason:** {selection['reason']}",
            "",
            "## Result",
            "",
            f"- **Statistic:** {_format_value(result.get('statistic'))}",
            f"- **p-value:** {_format_value(result.get('p_value'), p_value=True)}",
            f"- **Effect size:** {result.get('effect_size_name', 'effect size')} = "
            f"{_format_value(result.get('effect_size'))}",
            f"- **Group sizes:** {_format_mapping(result.get('group_sizes', {}))}",
            "",
            "## Interpretation",
            "",
            _interpretation(
                result.get("p_value"),
                selection["test"],
                subject,
                result=result,
                frame=frame,
                columns=interpretation_columns,
            ),
            "",
            "## Reproduce",
            "",
            "```python",
            reproduce,
            "```",
            "",
            f"CLI command: `{command}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _required_param(params: dict[str, str], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name!r} is required in params")
    return value


def _summary_table(summary: dict[str, Any]) -> str:
    rows = ["| Column | Type | Missing | Outliers |", "| --- | --- | ---: | ---: |"]
    for name, details in summary["columns"].items():
        rows.append(
            f"| {_escape(name)} | {_escape(details['type'])} | "
            f"{_format_value(details['missing'])} | {_format_value(details['outliers'])} |"
        )
    rows.append("")
    rows.append(f"**Duplicate rows:** {_format_value(summary['duplicate_rows'])}")
    return "\n".join(rows)


def _numeric_assumptions_table(assumptions: dict[str, Any]) -> str:
    rows = ["| Check | Statistic | p-value | Verdict |", "| --- | ---: | ---: | --- |"]
    for group, details in assumptions["shapiro"].items():
        verdict = _verdict(details.get("normal"), "consistent with normality", "not consistent with normality")
        rows.append(
            f"| Shapiro-Wilk ({_escape(group)}) | {_format_value(details.get('statistic'))} | "
            f"{_format_value(details.get('p_value'), p_value=True)} | {verdict} |"
        )
    levene = assumptions["levene"]
    verdict = _verdict(levene.get("equal_variance"), "equal variance", "unequal variance")
    rows.append(
        f"| Levene (equal variance) | {_format_value(levene.get('statistic'))} | "
        f"{_format_value(levene.get('p_value'), p_value=True)} | {verdict} |"
    )
    return "\n".join(rows)


def _categorical_assumptions_table(frame: pd.DataFrame, cat1: str, cat2: str) -> str:
    clean = frame[[cat1, cat2]].dropna()
    table = pd.crosstab(clean[cat1], clean[cat2])
    rows = ["| Check | Statistic | p-value | Verdict |", "| --- | ---: | ---: | --- |"]
    rows.append(
        f"| Contingency table | {_format_value(table.shape[0])} x {_format_value(table.shape[1])} "
        f"({_format_value(len(clean))} complete rows) | "
        f"n/a | {'usable' if table.shape[0] >= 2 and table.shape[1] >= 2 else 'fewer than two levels'} |"
    )

    expected: Any = None
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        try:
            expected = stats.chi2_contingency(table, correction=False)[3]
        except ValueError:
            expected = None
    if expected is None:
        statistic = "n/a"
        verdict = "expected counts could not be computed"
    else:
        minimum = float(expected.min())
        low_count = int((expected < 5).sum())
        statistic = f"minimum expected count = {_format_value(minimum)}"
        verdict = (
            "all expected counts >= 5"
            if low_count == 0
            else f"{_format_value(low_count)} expected count(s) < 5"
        )
    rows.append(f"| Expected counts | {statistic} | n/a | {verdict} |")
    return "\n".join(rows)


def _verdict(value: bool | None, passed: str, failed: str) -> str:
    if value is None:
        return "not checked"
    return passed if value else failed


def _interpretation(
    p_value: Any,
    test_name: str,
    subject: str,
    *,
    result: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    columns: tuple[str, str] | None = None,
) -> str:
    """Describe the observed direction and practical size in exactly two sentences."""
    result = result or {}
    formatted_p = _format_value(p_value, p_value=True)
    effect_name = str(result.get("effect_size_name", "effect size"))
    effect = result.get("effect_size")
    effect_magnitude = _effect_magnitude(effect_name, effect)

    if frame is not None and columns is not None:
        if test_name in {"t-test", "Welch", "Mann-Whitney", "ANOVA", "Kruskal-Wallis"}:
            direction = _numeric_direction(frame, columns[0], columns[1], test_name)
        else:
            direction = _categorical_direction(frame, columns[0], columns[1])
    else:
        direction = subject

    first = f"The observed {subject} was {direction}, with a p-value of {formatted_p}."
    second = (
        f"The {effect_name} was {_format_value(effect)}, indicating a {effect_magnitude} effect, "
        "and a small p-value alone does not show practical importance."
    )
    return f"{first} {second}"


def _numeric_direction(frame: pd.DataFrame, group_col: str, value_col: str, test_name: str) -> str:
    clean = frame[[group_col, value_col]].dropna()
    measure_name = "median" if test_name in {"Mann-Whitney", "Kruskal-Wallis"} else "mean"
    summaries: list[tuple[Any, float]] = []
    for label in pd.unique(clean[group_col]):
        values = clean.loc[clean[group_col] == label, value_col].astype(float)
        if values.empty:
            continue
        measure = values.median() if measure_name == "median" else values.mean()
        summaries.append((label, float(measure)))

    if not summaries:
        return f"no {measure_name} values were available for the groups"
    highest = max(summaries, key=lambda item: item[1])
    lowest = min(summaries, key=lambda item: item[1])
    if math.isclose(highest[1], lowest[1]):
        return f"equal {measure_name}s across groups ({_format_value(highest[1])})"
    if len(summaries) == 2:
        return (
            f"a higher {measure_name} in group {_label(highest[0])} "
            f"({_format_value(highest[1])}) than group {_label(lowest[0])} "
            f"({_format_value(lowest[1])})"
        )
    return (
        f"the highest {measure_name} in group {_label(highest[0])} "
        f"({_format_value(highest[1])}) and the lowest in group {_label(lowest[0])} "
        f"({_format_value(lowest[1])})"
    )


def _categorical_direction(frame: pd.DataFrame, cat1: str, cat2: str) -> str:
    clean = frame[[cat1, cat2]].dropna()
    table = pd.crosstab(clean[cat1], clean[cat2])
    if table.empty:
        return "no categorical pairing was available"
    row, column = table.stack().idxmax()
    count = table.loc[row, column]
    return (
        f"a pattern with {_escape(cat1)}={_label(row)} and "
        f"{_escape(cat2)}={_label(column)} as the most common pairing "
        f"({_format_value(count)} observations)"
    )


def _effect_magnitude(effect_name: str, effect: Any) -> str:
    try:
        absolute_effect = abs(float(effect))
    except (TypeError, ValueError):
        return "unavailable"
    if not math.isfinite(absolute_effect):
        return "unavailable"

    normalized_name = effect_name.lower().replace("‑", "-")
    if "cohen" in normalized_name:
        thresholds = (0.2, 0.5, 0.8)
    elif "cram" in normalized_name or "rank-biserial" in normalized_name or "rank biserial" in normalized_name:
        thresholds = (0.1, 0.3, 0.5)
    else:
        thresholds = (0.01, 0.06, 0.14)
    if absolute_effect < thresholds[0]:
        return "negligible"
    if absolute_effect < thresholds[1]:
        return "small"
    if absolute_effect < thresholds[2]:
        return "medium"
    return "large"


def _numeric_reproduction(csv_path: str | Path, group_col: str, value_col: str, call: str) -> str:
    test_code = {
        "scipy.stats.ttest_ind(group_1, group_2, equal_var=True)": "result = scipy.stats.ttest_ind(group_1, group_2, equal_var=True)",
        "scipy.stats.ttest_ind(group_1, group_2, equal_var=False)": "result = scipy.stats.ttest_ind(group_1, group_2, equal_var=False)",
        "scipy.stats.mannwhitneyu(group_1, group_2, alternative='two-sided')": "result = scipy.stats.mannwhitneyu(group_1, group_2, alternative='two-sided')",
        "scipy.stats.f_oneway(*groups)": "result = scipy.stats.f_oneway(*groups)",
        "scipy.stats.kruskal(*groups)": "result = scipy.stats.kruskal(*groups)",
    }.get(call)
    if test_code is None:
        raise ValueError(f"Unsupported SciPy call for reproduction: {call}")

    lines = [
        "import pandas as pd",
        "import scipy",
        "",
        f"frame = pd.read_csv({_relative_csv_path(csv_path)!r})",
        f"clean = frame[[{group_col!r}, {value_col!r}]].dropna()",
        f"groups = [clean.loc[clean[{group_col!r}] == label, {value_col!r}].astype(float).to_numpy() "
        "for label in pd.unique(clean[" + repr(group_col) + "])]",
    ]
    if "group_1" in test_code:
        lines.extend(["group_1, group_2 = groups", test_code])
    else:
        lines.append(test_code)
    lines.append("print(result.pvalue)")
    return "\n".join(lines)


def _categorical_reproduction(csv_path: str | Path, cat1: str, cat2: str, call: str) -> str:
    if call == "scipy.stats.fisher_exact(contingency_table)":
        test_code = "result = scipy.stats.fisher_exact(contingency_table)"
    elif call == "scipy.stats.chi2_contingency(contingency_table, correction=False)":
        test_code = "result = scipy.stats.chi2_contingency(contingency_table, correction=False)"
    else:
        raise ValueError(f"Unsupported SciPy call for reproduction: {call}")
    return "\n".join(
        [
            "import pandas as pd",
            "import scipy",
            "",
            f"frame = pd.read_csv({_relative_csv_path(csv_path)!r})",
            f"clean = frame[[{cat1!r}, {cat2!r}]].dropna()",
            f"contingency_table = pd.crosstab(clean[{cat1!r}], clean[{cat2!r}])",
            test_code,
            "print(result.pvalue)",
        ]
    )


def _cli_command(csv_path: str | Path, mode: str, **params: str) -> str:
    command = ["statguard", "analyze", _relative_csv_path(csv_path)]
    if mode == "numeric":
        command.extend(["--group", params["group"], "--value", params["value"]])
    else:
        command.extend(["--cat1", params["cat1"], "--cat2", params["cat2"]])
    return " ".join(shlex.quote(part) for part in command)


def _format_mapping(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{_escape(key)}={_format_mapping(item)}" for key, item in value.items())
    return _format_value(value)


def _format_value(value: Any, *, p_value: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    if not math.isfinite(numeric):
        return "n/a"
    if p_value and abs(numeric) < 0.0001:
        return f"{numeric:.3e}"
    return f"{numeric:.4g}"


def _relative_csv_path(csv_path: str | Path) -> str:
    """Return a path usable from the current workspace without an absolute prefix."""
    return os.path.relpath(str(csv_path), start=str(Path.cwd()))


def _label(value: Any) -> str:
    """Render a group/category label as text rather than as a numeric measure."""
    return _escape(value)


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
