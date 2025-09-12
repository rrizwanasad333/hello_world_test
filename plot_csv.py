#!/usr/bin/env python3
"""
Plot columns from a CSV file using pandas and matplotlib.

Usage examples:
  python plot_csv.py --file data.csv --x Date --y Sales --chart line
  python plot_csv.py --file data.csv --y ColA ColB --chart bar --rotate-x 45
  python plot_csv.py --file data.csv --chart scatter --limit 500

If --x is omitted, the row index is used. If --y is omitted, all numeric columns are plotted.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def infer_y_columns(df: pd.DataFrame, y_cols):
    if y_cols:
        missing = [c for c in y_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Y columns not found in CSV: {missing}")
        return y_cols
    # default to all numeric columns
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    if not nums:
        raise SystemExit("No numeric columns found to plot. Specify --y explicitly.")
    return nums


def load_csv(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read CSV: {e}")
    if limit:
        df = df.head(limit)
    return df


essential_charts = {"line", "bar", "scatter"}


def make_plot(df: pd.DataFrame, x_col: str | None, y_cols: list[str], chart: str,
              title: str | None, rotate_x: int, style: str | None):
    if style:
        try:
            plt.style.use(style)
        except Exception:
            print(f"Warning: style '{style}' not found. Using default.")

    if x_col and x_col not in df.columns:
        raise SystemExit(f"X column not found in CSV: {x_col}")

    x = df[x_col] if x_col else df.index

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart == "line":
        for col in y_cols:
            ax.plot(x, df[col], label=col)
    elif chart == "bar":
        import numpy as np
        n = len(y_cols)
        idx = np.arange(len(df))
        width = 0.8 / max(n, 1)
        for i, col in enumerate(y_cols):
            ax.bar(idx + i * width, df[col], width=width, label=col)
        ax.set_xticks(idx + (n - 1) * width / 2)
        ax.set_xticklabels(list(x))
    elif chart == "scatter":
        # For multiple y, plot multiple scatter series
        for col in y_cols:
            ax.scatter(x, df[col], label=col, s=20)
    else:
        raise SystemExit(f"Unsupported chart type: {chart}. Choose from {sorted(essential_charts)}")

    ax.set_xlabel(x_col if x_col else "index")
    ax.set_ylabel(", ".join(y_cols))
    if title:
        ax.set_title(title)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha="right")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig, ax


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot columns from a CSV file.")
    p.add_argument("--file", "-f", required=True, help="Path to CSV file")
    p.add_argument("--x", help="Column name to use for X-axis (default: row index)")
    p.add_argument("--y", nargs="*", help="One or more column names for Y-axis (default: all numeric columns)")
    p.add_argument("--chart", choices=sorted(list(essential_charts)), default="line", help="Chart type")
    p.add_argument("--title", help="Plot title")
    p.add_argument("--output", "-o", help="Path to save the figure (e.g., out.png). If omitted, shows a window.")
    p.add_argument("--limit", type=int, help="Limit number of rows from top of CSV")
    p.add_argument("--rotate-x", type=int, default=0, help="Rotate X tick labels by degrees")
    p.add_argument("--style", help="matplotlib style (e.g., 'seaborn-v0_8', 'ggplot')")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.file)

    df = load_csv(csv_path, limit=args.limit)
    y_cols = infer_y_columns(df, args.y)

    fig, _ = make_plot(
        df=df,
        x_col=args.x,
        y_cols=y_cols,
        chart=args.chart,
        title=args.title,
        rotate_x=args.rotate_x,
        style=args.style,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
