import pandas as pd
from typer.testing import CliRunner

from statguard.assumptions import check_numeric_assumptions
from statguard.cli import app
from statguard.select import select_categorical_test, select_numeric_test


def _numeric_selection(frame: pd.DataFrame) -> dict[str, str]:
    assumptions = check_numeric_assumptions(frame, "group", "value")
    return select_numeric_test(assumptions)


def test_normal_two_group_data_selects_t_test_or_welch():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8,
            "value": list(range(1, 9)) + list(range(2, 10)),
        }
    )

    assert _numeric_selection(frame)["test"] in {"t-test", "Welch"}


def test_skewed_two_group_data_selects_mann_whitney():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8,
            "value": [1, 1, 1, 1, 1, 1, 1, 20] + [2, 2, 2, 2, 2, 2, 2, 30],
        }
    )

    assert _numeric_selection(frame)["test"] == "Mann-Whitney"


def test_three_group_data_selects_anova_or_kruskal_wallis():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
            "value": list(range(1, 9)) + list(range(2, 10)) + list(range(3, 11)),
        }
    )

    assert _numeric_selection(frame)["test"] in {"ANOVA", "Kruskal-Wallis"}


def test_sparse_two_by_two_table_selects_fisher():
    frame = pd.DataFrame(
        {
            "cat1": ["A"] * 9 + ["B"] * 9,
            "cat2": ["X"] * 8 + ["Y"] + ["X"] + ["Y"] * 8,
        }
    )

    selection = select_categorical_test(frame, "cat1", "cat2")

    assert selection["test"] == "Fisher"


def test_analyze_cli_runs_categorical_mode(tmp_path):
    csv_path = tmp_path / "categorical.csv"
    pd.DataFrame(
        {
            "cat1": ["A"] * 9 + ["B"] * 9,
            "cat2": ["X"] * 8 + ["Y"] + ["X"] + ["Y"] * 8,
        }
    ).to_csv(csv_path, index=False)

    result = CliRunner().invoke(app, ["analyze", str(csv_path), "--cat1", "cat1", "--cat2", "cat2"])

    assert result.exit_code == 0, result.output
    assert "Contingency table: 2 x 2" in result.output
    assert "Chosen test: Fisher" in result.output
    assert "SciPy call: scipy.stats.fisher_exact" in result.output


def test_analyze_cli_prints_assumptions_result_and_scipy_call(tmp_path):
    csv_path = tmp_path / "numeric.csv"
    pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8,
            "value": list(range(1, 9)) + list(range(2, 10)),
        }
    ).to_csv(csv_path, index=False)

    result = CliRunner().invoke(app, ["analyze", str(csv_path), "--group", "group", "--value", "value"])

    assert result.exit_code == 0, result.stdout
    assert "Assumption checks:" in result.stdout
    assert "Chosen test:" in result.stdout
    assert "Reason:" in result.stdout
    assert "Statistic:" in result.stdout
    assert "p-value:" in result.stdout
    assert "Group sizes:" in result.stdout
    assert "Effect size:" in result.stdout
    assert "SciPy call: scipy.stats." in result.stdout


def test_analyze_cli_requires_exactly_one_mode(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"group": ["A", "B"], "value": [1, 2]}).to_csv(csv_path, index=False)

    result = CliRunner().invoke(app, ["analyze", str(csv_path)])

    assert result.exit_code != 0
    assert "exactly one analysis mode" in result.output


def test_analyze_cli_reports_missing_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"group": ["A", "B"], "value": [1, 2]}).to_csv(csv_path, index=False)

    result = CliRunner().invoke(app, ["analyze", str(csv_path), "--group", "missing", "--value", "value"])

    assert result.exit_code != 0
    assert "Column(s) not found in CSV: missing" in result.output
