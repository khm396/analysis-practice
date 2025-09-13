"""
qdl.dataloader

PRD v0.2-aligned, schema-agnostic DataLoader core for factors and characteristics.

This module implements `load_factors` (CSV) and a generic `load_chars` (Parquet).
`load_factors` follows the naming convention found under `data/factors/`:

    [<country>]_[<dataset>]_[monthly]_[<weighting>].csv

Where:
- <country> ∈ {usa, kor}
- <dataset> ∈ {all_factors, all_themes, mkt}
- frequency is fixed to `monthly`
- <weighting> ∈ {ew, vw, vw_cap}

Schema is intentionally not assumed here; CSVs are loaded as-is. Any parsing,
renaming, or dtype normalization is the responsibility of `qdl.transformer`.

Design policies (from PRD v0.2):
- Do not invent schemas or column names.
- Use absolute imports and centralized path config from `qdl.config`.
- Raise explicit errors; do not synthesize data on failure.

Factor reader spec (factors CSV)
--------------------------------
Inputs (required unless noted):
- country: {"usa", "kor"}
- dataset: {"factor", "theme", "mkt"} → mapped internally to {"all_factors", "all_themes", "mkt"}
- frequency: {"monthly"} (only) — default "monthly"
- weighting: {"ew", "vw", "vw_cap"}  ← includes "vw" option
- encoding: str (default "utf-8")

File naming pattern (literal brackets/underscores):
    "[<country>]_[<dataset_token>]_[monthly]_[<weighting>].csv"
Examples:
    [usa]_[all_factors]_[monthly]_[ew].csv
    [usa]_[mkt]_[monthly]_[vw].csv
    [kor]_[all_themes]_[monthly]_[vw_cap].csv

Location:
- Files are read from qdl.config.FACTORS_PATH (i.e., <repo>/data/factors)

Behavior:
- Validate inputs; build the exact filename; assemble full path under FACTORS_PATH
- If file exists: load with pandas.read_csv(encoding=encoding) and return DataFrame (no schema enforcement)
- If file missing: raise FileNotFoundError and list available .csv files in the directory
- If inputs invalid: raise ValueError with clear message

Notes:
- Any column parsing/renaming/typing happens in `qdl.transformer` later
- Future: expose optional usecols/dtype hints if needed for performance

Usage examples:
- load_factors(country="usa", dataset="mkt", weighting="vw")
- load_factors(country="kor", dataset="factor", weighting="ew")
"""

from pathlib import Path
from typing import Literal, Optional, List, Set

import pandas as pd

from qdl.config import FACTORS_PATH, CHARS_PATH

Country = Literal["usa", "kor"]
DatasetKind = Literal["factor", "theme", "mkt"]
Weighting = Literal["ew", "vw", "vw_cap"]
Frequency = Literal["monthly"]
Vintage = Literal["1972-", "2000-", "2020-"]


_DATASET_TOKEN_BY_KIND = {
    "factor": "all_factors",
    "theme": "all_themes",
    "mkt": "mkt",
}


def _build_factors_filename(
    *, country: Country, dataset: DatasetKind, frequency: Frequency, weighting: Weighting
) -> str:
    dataset_token = _DATASET_TOKEN_BY_KIND[dataset]
    # File names are literal with square brackets and underscores
    # Example: [usa]_[all_factors]_[monthly]_[ew].csv
    return f"[{country}]_[{dataset_token}]_[{frequency}]_[{weighting}].csv"


def load_factors(
    *,
    country: Country,
    dataset: DatasetKind,
    weighting: Weighting,
    frequency: Frequency = "monthly",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load factors CSV from `data/factors/` based on naming convention.

    Parameters
    ----------
    country : {"usa", "kor"}
        Country code segment in file name.
    dataset : {"factor", "theme", "mkt"}
        Logical dataset kind; mapped internally to {"all_factors","all_themes","mkt"}.
    weighting : {"ew", "vw", "vw_cap"}
        Weighting scheme in file name.
    frequency : {"monthly"}, default "monthly"
        Only "monthly" is supported at the moment.
    encoding : str, default "utf-8"
        CSV file encoding.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame loaded via pandas.read_csv. No schema assumptions are made.

    Raises
    ------
    FileNotFoundError
        If the composed file does not exist under `qdl.config.FACTORS_PATH`.
    ValueError
        If any of the provided parameters are invalid for the naming convention.
    """
    # Validate allowed values explicitly to provide clear error messages.
    if country not in ("usa", "kor"):
        raise ValueError("country must be one of {'usa','kor'}")
    if dataset not in _DATASET_TOKEN_BY_KIND:
        raise ValueError("dataset must be one of {'factor','theme','mkt'}")
    if frequency != "monthly":
        raise ValueError("frequency must be 'monthly'")
    if weighting not in ("ew", "vw", "vw_cap"):
        raise ValueError("weighting must be one of {'ew','vw','vw_cap'}")

    file_name = _build_factors_filename(
        country=country, dataset=dataset, frequency=frequency, weighting=weighting
    )
    file_path: Path = FACTORS_PATH / file_name

    if not file_path.exists():
        # Provide a helpful hint listing the directory contents for debugging.
        available = sorted(p.name for p in FACTORS_PATH.glob("*.csv"))
        raise FileNotFoundError(
            "Factors file not found: "
            f"{file_path} (available: {', '.join(available) if available else 'none'})"
        )

    # Schema-agnostic load. Any parsing/normalization belongs in transformer.
    df = pd.read_csv(file_path, encoding=encoding)
    # Ensure 'date' is datetime for downstream comparisons/joins
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df


# --------------- Characteristics (Parquet) loader -----------------

def _resolve_single_parquet(
    base_dir: Path,
    *,
    file_name: Optional[str] = None,
    patterns: Optional[List[str]] = None,
) -> Path:
    """
    Resolve exactly one parquet file under `base_dir` using either an exact
    `file_name` or a set of glob `patterns`. Raises when zero or multiple match.
    """
    if file_name:
        candidate = base_dir / file_name
        if not candidate.exists():
            available = sorted(p.name for p in base_dir.glob("*.parquet"))
            raise FileNotFoundError(
                f"Parquet file not found: {candidate} (available: {', '.join(available) if available else 'none'})"
            )
        if candidate.suffix.lower() != ".parquet":
            raise ValueError("file_name must point to a .parquet file")
        return candidate

    if not patterns:
        raise ValueError("Provide either 'file_name' or one or more 'patterns' to locate a parquet file")

    matched: Set[Path] = set()
    for pat in patterns:
        for p in base_dir.glob(pat):
            if p.suffix.lower() == ".parquet":
                matched.add(p)
    if not matched:
        available = sorted(p.name for p in base_dir.glob("*.parquet"))
        raise FileNotFoundError(
            f"No parquet files matched patterns {patterns} under {base_dir} (available: {', '.join(available) if available else 'none'})"
        )
    if len(matched) > 1:
        names = ", ".join(sorted(m.name for m in matched))
        raise ValueError(f"Ambiguous patterns; matched multiple files: {names}")

    return next(iter(matched))


def load_chars(
    *,
    file_name: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    engine: str = "pyarrow",
) -> pd.DataFrame:
    """
    Load a characteristics parquet file from `data/chars/`.

    Exactly one of `file_name` or `patterns` must identify a single `.parquet` file
    under `qdl.config.CHARS_PATH`. No schema is assumed.

    Parameters
    ----------
    file_name : str, optional
        Exact parquet file name inside `data/chars/`.
    patterns : list[str], optional
        One or more glob patterns (relative to `data/chars/`) that must match
        exactly one parquet file (e.g., ["jkp_2020-*_kor.parquet"]).
    columns : list[str], optional
        Column projection to speed up reads.
    engine : str, default "pyarrow"
        Parquet engine to use.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame loaded via pandas.read_parquet.
    """
    file_path = _resolve_single_parquet(CHARS_PATH, file_name=file_name, patterns=patterns)
    df = pd.read_parquet(file_path, columns=columns, engine=engine)
    # Ensure 'date' is datetime for downstream comparisons/joins
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df
