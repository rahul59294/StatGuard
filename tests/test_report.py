import re

import pandas as pd
import pytest

from statguard.assumptions import check_numeric_assumptions
from statguard.report import build_report
from statguard.run import run_categorical_test, run_numeric_test
from statguard.select import select_categorical_test, select_numeric_test


HEADINGS = [
    "## Dataset summary",
    "## Assumption checks",
    "## Chosen test and reason",
    "## Result",
    "## Interpretation",
    "## Reproduce",
]


def _reproduction_code(report: str) -> str:
    return re.search(r"```python\n(.*?)\n```", report, re.DOTALL).group(1)


def _reported_p_value(report: str) -> float:
    match = re.search(r"^- \*\*p-value:\*\* ([^\n]+)$", report, re.MULTILINE)
    assert match
    return float(match.group(1))


def test_numeric_report_has_required_sections_and_reproduces_result(tmp_path):
    csv_path = tmp_path / "numeric.csv"
    frame = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8,
            "score": list(range(1, 9)) + list(range(2, 10)),
            "label": ["x"] * 16,
        }
    )
    frame.to_csv(csv_path, index=False)

    report = build_report(csv_path, "numeric", {"group": "group", "value": "score"})

    for heading in HEADINGS:
        assert heading in report
    assert "| Column | Type | Missing | Outliers |" in report
    assert "| Shapiro-Wilk (A) |" in report
    assert "| Shapiro-Wilk (B) |" in report
    assert "| Levene (equal variance) |" in report
    assert "**Duplicate rows:** 0" in report
    assert len(report.split("## Interpretation", 1)[1].split("## Reproduce", 1)[0].strip().split(". ")) == 2

    assumptions = check_numeric_assumptions(frame, "group", "score")
    selection = select_numeric_test(assumptions)
    expected = run_numeric_test(frame, "group", "score", selection["test"])
    namespace = {"__name__": "__main__"}
    exec(_reproduction_code(report), namespace)

    assert namespace["result"].pvalue == pytest.approx(expected["p_value"])
    assert _reported_p_value(report) == pytest.approx(expected["p_value"])


def test_categorical_report_has_expected_count_check_and_reproduces_result(tmp_path):
    csv_path = tmp_path / "categorical.csv"
    frame = pd.DataFrame(
        {
            "x": ["A"] * 9 + ["B"] * 9,
            "y": ["X"] * 8 + ["Y"] + ["X"] + ["Y"] * 8,
        }
    )
    frame.to_csv(csv_path, index=False)

    report = build_report(csv_path, "categorical", {"cat1": "x", "cat2": "y"})
    assert all(heading in report for heading in HEADINGS)
    assert "| Contingency table | 2 x 2 (18 complete rows) |" in report
    assert "| Expected counts |" in report

    selection = select_categorical_test(frame, "x", "y")
    expected = run_categorical_test(frame, "x", "y", selection["test"])
    namespace = {"__name__": "__main__"}
    exec(_reproduction_code(report), namespace)

    assert namespace["result"].pvalue == pytest.approx(expected["p_value"])
    assert _reported_p_value(report) == pytest.approx(expected["p_value"])
