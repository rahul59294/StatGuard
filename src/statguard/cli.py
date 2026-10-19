"""Command-line interface for StatGuard."""

from pathlib import Path
from typing import Any

import typer

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


if __name__ == "__main__":
    app()
