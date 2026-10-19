"""Command-line interface for StatGuard."""

from pathlib import Path
from typing import Any

import pandas as pd
import typer

from .assumptions import check_numeric_assumptions
from .run import run_categorical_test, run_numeric_test
from .select import select_categorical_test, select_numeric_test

app = typer.Typer(
    name="statguard",
    help="Profile CSV data and prepare it for statistical analysis.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Profile CSV data and prepare it for statistical analysis."""


def _format_result(result: Any) -> str:
    """Render common profiling result types as a readable table."""
    if hasattr(result, "to_string"):
        return result.to_string(index=False)
    if isinstance(result, dict):
        columns = result.get("columns")
        if isinstance(columns, dict) and all(isinstance(value, dict) for value in columns.values()):
            headers = ("Column", "Type", "Missing", "Outliers")
            rows = [
                (
                    str(name),
                    str(details.get("type", "")),
                    str(details.get("missing", "")),
                    str(details.get("outliers", "")),
                )
                for name, details in columns.items()
            ]
            widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
            lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
            lines.append("  ".join("-" * width for width in widths))
            lines.extend("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
            if "duplicate_rows" in result:
                lines.append(f"\nDuplicate rows: {result['duplicate_rows']}")
            return "\n".join(lines)
        width = max((len(str(key)) for key in result), default=0)
        return "\n".join(f"{str(key).ljust(width)}  {value}" for key, value in result.items())
    if isinstance(result, (list, tuple)):
        return "\n".join(str(row) for row in result)
    return str(result)


@app.command("profile")
def profile_csv(csv: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False)) -> None:
    """Profile CSV and print the resulting summary table."""
    # Import lazily so the scaffold's CLI help remains available before the
    # profile implementation is added.
    from .profile import profile

    result = profile(csv)
    typer.echo(_format_result(result))


def _format_value(value: Any) -> str:
    """Format scalar and mapping values compactly for terminal output."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items())
    return str(value)


def _print_result(selection: dict[str, Any], result: dict[str, Any]) -> None:
    """Print the common, reproducible part of an analysis result."""
    typer.echo(f"Chosen test: {selection['test']}")
    typer.echo(f"Reason: {selection['reason']}")
    typer.echo(f"Statistic: {_format_value(result.get('statistic'))}")
    typer.echo(f"p-value: {_format_value(result.get('p_value'))}")
    typer.echo(f"Group sizes: {_format_value(result.get('group_sizes', {}))}")
    typer.echo(
        "Effect size: "
        f"{result.get('effect_size_name', 'effect size')}="
        f"{_format_value(result.get('effect_size'))}"
    )
    typer.echo(f"SciPy call: {result.get('scipy_call', 'n/a')}")


def _validate_mode(
    group: str | None,
    value: str | None,
    cat1: str | None,
    cat2: str | None,
) -> str:
    """Validate mutually exclusive numeric and categorical option pairs."""
    numeric_options = group is not None or value is not None
    categorical_options = cat1 is not None or cat2 is not None

    if numeric_options and categorical_options:
        raise typer.BadParameter(
            "Choose exactly one analysis mode: --group with --value, or --cat1 with --cat2."
        )
    if not numeric_options and not categorical_options:
        raise typer.BadParameter(
            "Provide exactly one analysis mode: --group with --value, or --cat1 with --cat2."
        )
    if numeric_options and (group is None or value is None):
        raise typer.BadParameter("Numeric mode requires both --group and --value.")
    if categorical_options and (cat1 is None or cat2 is None):
        raise typer.BadParameter("Categorical mode requires both --cat1 and --cat2.")
    return "numeric" if numeric_options else "categorical"


def _validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise a user-facing error for any requested CSV column that is absent."""
    missing = list(dict.fromkeys(column for column in columns if column not in frame.columns))
    if missing:
        raise typer.BadParameter(f"Column(s) not found in CSV: {', '.join(missing)}")


@app.command("analyze")
def analyze_csv(
    csv: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    group: str | None = typer.Option(None, "--group", help="Numeric grouping column."),
    value: str | None = typer.Option(None, "--value", help="Numeric value column."),
    cat1: str | None = typer.Option(None, "--cat1", help="First categorical column."),
    cat2: str | None = typer.Option(None, "--cat2", help="Second categorical column."),
) -> None:
    """Check assumptions, select, and run a statistical test for a CSV."""
    mode = _validate_mode(group, value, cat1, cat2)
    try:
        frame = pd.read_csv(csv)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"Could not read CSV: {exc}") from exc

    if mode == "numeric":
        assert group is not None and value is not None
        _validate_columns(frame, [group, value])
        try:
            assumptions = check_numeric_assumptions(frame, group, value)
            selection = select_numeric_test(assumptions)
            result = run_numeric_test(frame, group, value, selection["test"])
        except (TypeError, ValueError, KeyError) as exc:
            raise typer.BadParameter(f"Unable to analyze numeric data: {exc}") from exc

        typer.echo(f"Analysis: numeric values grouped by '{group}'")
        typer.echo("Assumption checks:")
        for name, details in assumptions["shapiro"].items():
            normal = details.get("normal")
            normal_text = "not checked" if normal is None else ("passed" if normal else "failed")
            typer.echo(
                f"  Shapiro-Wilk ({name}): n={details.get('n')}, "
                f"statistic={_format_value(details.get('statistic'))}, "
                f"p-value={_format_value(details.get('p_value'))}; {normal_text}"
            )
            if details.get("note"):
                typer.echo(f"    Note: {details['note']}")
        levene = assumptions["levene"]
        variance = levene.get("equal_variance")
        variance_text = "not checked" if variance is None else ("passed" if variance else "failed")
        typer.echo(
            "  Levene (equal variance): "
            f"statistic={_format_value(levene.get('statistic'))}, "
            f"p-value={_format_value(levene.get('p_value'))}; {variance_text}"
        )
        if levene.get("note"):
            typer.echo(f"    Note: {levene['note']}")
        _print_result(selection, result)
        return

    assert cat1 is not None and cat2 is not None
    _validate_columns(frame, [cat1, cat2])
    try:
        selection = select_categorical_test(frame, cat1, cat2)
        result = run_categorical_test(frame, cat1, cat2, selection["test"])
    except (TypeError, ValueError, KeyError) as exc:
        raise typer.BadParameter(f"Unable to analyze categorical data: {exc}") from exc

    clean = frame[[cat1, cat2]].dropna()
    table = pd.crosstab(clean[cat1], clean[cat2])
    typer.echo(f"Analysis: categorical association between '{cat1}' and '{cat2}'")
    typer.echo("Assumption checks:")
    typer.echo(
        f"  Contingency table: {table.shape[0]} x {table.shape[1]} "
        f"({len(clean)} complete rows)."
    )
    typer.echo("  Expected counts: checked to choose chi-square or Fisher's exact test.")
    _print_result(selection, result)


if __name__ == "__main__":
    app()
