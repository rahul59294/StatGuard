"""Basic profiling for CSV data."""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


def profile(csv_path: str | Path) -> dict:
    """Return column metadata and row-level duplicate information for a CSV."""
    frame = pd.read_csv(csv_path)
    columns: dict[str, dict[str, int | str]] = {}

    for name, series in frame.items():
        numeric = is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        outliers = _count_outliers(series) if numeric else 0
        columns[name] = {
            "type": "numeric" if numeric else "categorical",
            "missing": int(series.isna().sum()),
            "outliers": outliers,
        }

    return {
        "columns": columns,
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def _count_outliers(series: pd.Series) -> int:
    """Count non-missing values outside the standard 1.5*IQR fences."""
    values = series.dropna()
    if values.empty:
        return 0

    first_quartile = values.quantile(0.25)
    third_quartile = values.quantile(0.75)
    iqr = third_quartile - first_quartile
    lower_fence = first_quartile - 1.5 * iqr
    upper_fence = third_quartile + 1.5 * iqr

    return int(((values < lower_fence) | (values > upper_fence)).sum())
