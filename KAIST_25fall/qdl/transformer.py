"""
qdl.transformer

Transformer utilities to preprocess and reshape data.

This module provides a generic wide pivot function and a
factors-specific convenience wrapper that defaults to the
observed factor file schema (`date`, `name`, `ret`).

Note: No schema coercion beyond minimal checks and time parsing.
"""

from __future__ import annotations

from typing import Iterable, Literal

import pandas as pd


AggKind = Literal["first", "mean", "sum"]


def _ensure_columns_exist(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def to_wide(
    df: pd.DataFrame,
    *,
    index_cols: list[str],
    column_col: str,
    value_col: str,
    agg: AggKind = "first",
    sort_index: bool = True,
    sort_columns: bool = True,
) -> pd.DataFrame:
    """
    Convert long data to wide by pivoting `column_col` to columns and
    `index_cols` to the row index, filling values from `value_col`.

    When duplicates exist for the same key tuple, aggregate using `agg`.
    """
    _ensure_columns_exist(df, [*index_cols, column_col, value_col])

    # Use pivot_table to handle potential duplicates deterministically.
    aggfunc = {"first": "first", "mean": "mean", "sum": "sum"}[agg]
    wide = pd.pivot_table(
        df,
        index=index_cols,
        columns=column_col,
        values=value_col,
        aggfunc=aggfunc,
        dropna=False,
    )

    # Restore a regular Index for columns (remove the name to keep it clean).
    wide.columns.name = None

    if sort_index:
        wide = wide.sort_index()
    if sort_columns:
        wide = wide.reindex(sorted(wide.columns), axis=1)

    return wide


def to_wide_factors(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    name_col: str = "name",
    value_col: str = "ret",
    agg: AggKind = "first",
) -> pd.DataFrame:
    """
    Convenience wrapper to pivot factor data to wide format.

    - Rows indexed by `date_col`
    - Columns from unique values of `name_col`
    - Values from `value_col`
    """
    _ensure_columns_exist(df, [date_col, name_col, value_col])

    # Parse date column to datetime for stable indexing (no timezone assumption here).
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    wide = to_wide(
        df,
        index_cols=[date_col],
        column_col=name_col,
        value_col=value_col,
        agg=agg,
        sort_index=True,
        sort_columns=True,
    )
    return wide


def to_wide_chars(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "id",
    value_col: str,
    agg: AggKind = "first",
) -> pd.DataFrame:
    """
    Pivot characteristics data to wide format.

    - Rows indexed by `date_col`
    - Columns from unique values of `id_col`
    - Values from `value_col` (must be specified by caller)
    """
    _ensure_columns_exist(df, [date_col, id_col, value_col])

    # Parse date column to datetime for stable indexing (no timezone assumption here).
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    wide = to_wide(
        df,
        index_cols=[date_col],
        column_col=id_col,
        value_col=value_col,
        agg=agg,
        sort_index=True,
        sort_columns=True,
    )
    return wide
