from pathlib import Path

from statguard.profile import profile


DATA_PATH = Path(__file__).parent / "data" / "tiny.csv"


def test_profile_reports_column_types():
    result = profile(DATA_PATH)

    assert result["columns"]["score"]["type"] == "numeric"
    assert result["columns"]["age"]["type"] == "numeric"
    assert result["columns"]["name"]["type"] == "categorical"


def test_profile_counts_missing_values():
    result = profile(DATA_PATH)

    assert result["columns"]["score"]["missing"] == 2
    assert result["columns"]["name"]["missing"] == 0


def test_profile_counts_duplicate_rows():
    result = profile(DATA_PATH)

    assert result["duplicate_rows"] == 1


def test_profile_counts_numeric_iqr_outliers_excluding_missing_values():
    result = profile(DATA_PATH)

    assert result["columns"]["score"]["outliers"] == 1
    assert result["columns"]["age"]["outliers"] == 0


def test_profile_does_not_report_outliers_for_non_numeric_columns():
    result = profile(DATA_PATH)

    assert result["columns"]["name"]["outliers"] == 0
