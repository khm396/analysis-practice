"""
qdl.facade

Minimal facade exposing only load and validate_factor per PRD v0.2.

- Transform and preprocessing are internal concerns handled by underlying modules.
- Facade orchestrates dataloader and validator without assuming schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from qdl import dataloader as _dataloader  # absolute import per project policy
from qdl import validator as _validator    # validator API expected to be defined later
from qdl import transformer as _transformer


class QDL:
    """
    Facade for end users. Exposes only load and validate_factor.

    Notes
    -----
    - This class delegates to qdl.dataloader and qdl.validator.
    - No schema assumptions are made here; callers must provide keys explicitly
      or rely on later config-driven resolution when available.
    """

    def __init__(self, *, loader: Any = None, validator: Any = None) -> None:
        # Allow dependency injection for tests/extensibility
        self._loader = loader or _dataloader
        self._validator = validator or _validator

    def load_factor_dataset(
        self,
        *,
        country: Literal["usa", "kor"],
        dataset: Literal["factor", "theme", "mkt"],
        weighting: Literal["ew", "vw", "vw_cap"],
        frequency: Literal["monthly"] = "monthly",
        encoding: str = "utf-8",
        columns: Optional[List[str]] = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Load the raw long-form factor dataset (CSV) via the public API.

        Delegates to dataloader.load_factors.

        Notes
        -----
        - Returns long-form data.
        - Regardless of the requested `columns`, the composite identifier ["date","name"]
          is always included to support downstream operations.
        """
        df = self._loader.load_factors(
            country=country,
            dataset=dataset,
            weighting=weighting,
            frequency=frequency,
            encoding=encoding,
        )
        if columns is None:
            return df

        # Always include composite identifier keys for factors
        required_keys = ["date", "name"]
        requested_with_required = list(dict.fromkeys([*columns, *required_keys]))

        missing = [c for c in requested_with_required if c not in df.columns]
        if missing and strict:
            raise KeyError(f"Requested columns not found: {missing}")

        # Order keys first, then the remaining requested columns preserving order
        keys_first = [c for c in required_keys if c in df.columns]
        rest = [c for c in requested_with_required if c in df.columns and c not in required_keys]
        return df[keys_first + rest]

    def load_factors(
        self,
        *,
        country: Literal["usa", "kor"],
        dataset: Literal["factor", "theme", "mkt"],
        weighting: Literal["ew", "vw", "vw_cap"],
        frequency: Literal["monthly"] = "monthly",
        encoding: str = "utf-8",
        factors: Optional[List[str]] = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Load factors and return a wide DataFrame (date index, factor names as columns).

        Parameters
        ----------
        factors : list[str], optional
            Subset of factor names (wide columns) to return. If provided and `strict=True`,
            raise when any requested factor is missing. If `strict=False`, return the
            intersection silently.
        """
        long_df = self.load_factor_dataset(
            country=country,
            dataset=dataset,
            weighting=weighting,
            frequency=frequency,
            encoding=encoding,
            columns=["date", "name", "ret"],
            strict=True,
        )
        wide = _transformer.to_wide_factors(long_df)
        if factors is None:
            return wide
        missing = [f for f in factors if f not in wide.columns]
        if missing and strict:
            raise KeyError(f"Requested factors not found: {missing}")
        present_in_order = [f for f in factors if f in wide.columns]
        return wide[present_in_order]

    def load_char_dataset(
        self,
        *,
        country: Literal["usa", "kor"],
        vintage: Literal["1972-", "2000-", "2020-"],
        columns: Optional[List[str]] = None,
        engine: str = "pyarrow",
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Load JKP characteristics datasets (Parquet) via the public API.

        Constructs the filename from (vintage, country) and delegates to dataloader.load_chars.

        Notes
        -----
        - Regardless of the requested `columns`, the composite identifier
          ["date", "id"] is always included in the returned frame to support
          downstream operations (e.g., pivoting).
        """
        file_name = f"jkp_{vintage}_{country}.parquet"
        # Always include composite identifier keys for chars
        required_keys = ["date", "id"]
        if columns is None:
            requested_with_required = None
        else:
            requested_with_required = list(dict.fromkeys([*columns, *required_keys]))

        if requested_with_required is None or strict:
            # Strict mode (or no projection): delegate directly; underlying reader will raise on missing columns
            return self._loader.load_chars(
                file_name=file_name,
                columns=requested_with_required,
                engine=engine,
            )

        # Non-strict with projection: try pushdown first; if it fails, load all and filter intersection
        try:
            return self._loader.load_chars(
                file_name=file_name,
                columns=requested_with_required,
                engine=engine,
            )
        except (KeyError, ValueError):
            df_all = self._loader.load_chars(
                file_name=file_name,
                columns=None,
                engine=engine,
            )
            target_cols = requested_with_required or []
            keys_first = [c for c in required_keys if c in df_all.columns]
            rest = [c for c in target_cols if c in df_all.columns and c not in required_keys]
            return df_all[keys_first + rest]

    # Backward-compatible alias to previous API name
    def load_chars(
        self,
        *,
        country: Literal["usa", "kor"],
        vintage: Literal["1972-", "2000-", "2020-"],
        columns: Optional[List[str]] = None,
        engine: str = "pyarrow",
        strict: bool = True,
    ) -> pd.DataFrame:
        return self.load_char_dataset(
            country=country,
            vintage=vintage,
            columns=columns,
            engine=engine,
            strict=strict,
        )

    def load_char(
        self,
        *,
        country: Literal["usa", "kor"],
        vintage: Literal["1972-", "2000-", "2020-"],
        char: str,
        engine: str = "pyarrow",
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Load a single characteristic and return a 2D wide DataFrame with `date` as index
        and `id` as columns, values from the specified `char` column.
        """
        # Ensure required columns are present (strict load to surface errors early)
        df = self.load_char_dataset(
            country=country,
            vintage=vintage,
            columns=["date", "id", char],
            engine=engine,
            strict=True if strict else False,
        )
        # Pivot to wide
        wide = _transformer.to_wide(
            df,
            index_cols=["date"],
            column_col="id",
            value_col=char,
            agg="first",
            sort_index=True,
            sort_columns=True,
        )
        return wide

    def validate_factor(
        self,
        *,
        user_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        on: Optional[List[str]] = None,
        value_col: Optional[str] = None,
        reference_load_params: Optional[Dict[str, Any]] = None,
        names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Validate a user-generated factor series against a reference.

        Parameters
        ----------
        user_df : pd.DataFrame
            User-provided factor data.
        reference_df : pd.DataFrame, optional
            Reference factor data. If not provided, it will be loaded using reference_load_params.
        on : list[str]
            Join keys for alignment (e.g., [time, series-id]). Required (no PRD default).
        value_col : str
            Name of the numeric value column to compare. Required (no PRD default).
        reference_load_params : dict, optional
            Parameters forwarded to self.load_factor_dataset(**params) when reference_df is not provided.
        names : list[str], optional
            Optional list of factor names to include in the validation when
            working with long-form factors (column "name"). Both user and
            reference datasets will be filtered to these names when the column
            exists.
        kwargs : Any
            Forwarded to the underlying validator implementation (e.g., thresholds).
        """
        if on is None or value_col is None:
            raise ValueError("validate_factor requires explicit 'on' and 'value_col' arguments")

        if reference_df is None:
            if not reference_load_params:
                raise ValueError(
                    "reference_df is None and reference_load_params not provided; cannot load reference"
                )
            ref = self.load_factor_dataset(**reference_load_params)
        else:
            ref = reference_df

        # Optional name-level filtering (long format factors)
        if names:
            if "name" in user_df.columns:
                user_df = user_df[user_df["name"].isin(names)]
            if "name" in ref.columns:
                ref = ref[ref["name"].isin(names)]

        # Delegate to validator; assumes a compatible API exists.
        return self._validator.validate_factor(
            user=user_df,
            reference=ref,
            on=on,
            value_col=value_col,
            **kwargs,
        )
