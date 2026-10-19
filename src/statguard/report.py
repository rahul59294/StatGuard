"""Render reproducible Markdown reports for StatGuard analyses."""

from __future__ import annotations

import math
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
        subject = "difference between the groups"
    else:
        cat1 = _required_param(params, "cat1")
        cat2 = _required_param(params, "cat2")
        selection = select_categorical_test(frame, cat1, cat2)
        result = run_categorical_test(frame, cat1, cat2, selection["test"])
        lines.extend(["## Assumption checks", "", _categorical_assumptions_table(frame, cat1, cat2), ""])
        reproduce = _categorical_reproduction(csv_path, cat1, cat2, result["scipy_call"])
        command = _cli_command(csv_path, "categorical", cat1=cat1, cat2=cat2)
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
            f"- **p-value:** {_format_value(result.get('p_value'))}",
            f"- **Effect size:** {result.get('effect_size_name', 'effect size')} = "
            f"{_format_value(result.get('effect_size'))}",
            f"- **Group sizes:** {_format_mapping(result.get('group_sizes', {}))}",
            "",
            "## Interpretation",
            "",
            _interpretation(result.get("p_value"), selection["test"], subject),
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
            f"{details['missing']} | {details['outliers']} |"
        )
    rows.append("")
    rows.append(f"**Duplicate rows:** {summary['duplicate_rows']}")
    return "\n".join(rows)


def _numeric_assumptions_table(assumptions: dict[str, Any]) -> str:
    rows = ["| Check | Statistic | p-value | Verdict |", "| --- | ---: | ---: | --- |"]
    for group, details in assumptions["shapiro"].items():
        verdict = _verdict(details.get("normal"), "consistent with normality", "not consistent with normality")
        rows.append(
            f"| Shapiro-Wilk ({_escape(group)}) | {_format_value(details.get('statistic'))} | "
            f"{_format_value(details.get('p_value'))} | {verdict} |"
        )
    levene = assumptions["levene"]
    verdict = _verdict(levene.get("equal_variance"), "equal variance", "unequal variance")
    rows.append(
        f"| Levene (equal variance) | {_format_value(levene.get('statistic'))} | "
        f"{_format_value(levene.get('p_value'))} | {verdict} |"
    )
    return "\n".join(rows)


def _categorical_assumptions_table(frame: pd.DataFrame, cat1: str, cat2: str) -> str:
    clean = frame[[cat1, cat2]].dropna()
    table = pd.crosstab(clean[cat1], clean[cat2])
    rows = ["| Check | Statistic | p-value | Verdict |", "| --- | ---: | ---: | --- |"]
    rows.append(
        f"| Contingency table | {table.shape[0]} x {table.shape[1]} ({len(clean)} complete rows) | "
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
        verdict = "all expected counts >= 5" if low_count == 0 else f"{low_count} expected count(s) < 5"
    rows.append(f"| Expected counts | {statistic} | n/a | {verdict} |")
    return "\n".join(rows)


def _verdict(value: bool | None, passed: str, failed: str) -> str:
    if value is None:
        return "not checked"
    return passed if value else failed


def _interpretation(p_value: Any, test_name: str, subject: str) -> str:
    formatted = _format_value(p_value)
    if p_value is not None and float(p_value) < 0.05:
        conclusion = f"The result provides evidence of {subject} at the 0.05 level."
    else:
        conclusion = f"The result does not provide sufficient evidence of {subject} at the 0.05 level."
    return f"The {test_name} returned a p-value of {formatted}. {conclusion}"


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
        f"frame = pd.read_csv({str(csv_path)!r})",
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
            f"frame = pd.read_csv({str(csv_path)!r})",
            f"clean = frame[[{cat1!r}, {cat2!r}]].dropna()",
            f"contingency_table = pd.crosstab(clean[{cat1!r}], clean[{cat2!r}])",
            test_code,
            "print(result.pvalue)",
        ]
    )


def _cli_command(csv_path: str | Path, mode: str, **params: str) -> str:
    command = ["statguard", "analyze", str(csv_path)]
    if mode == "numeric":
        command.extend(["--group", params["group"], "--value", params["value"]])
    else:
        command.extend(["--cat1", params["cat1"], "--cat2", params["cat2"]])
    return " ".join(shlex.quote(part) for part in command)


def _format_mapping(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{_escape(key)}={_format_mapping(item)}" for key, item in value.items())
    return _format_value(value)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.15g}"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
