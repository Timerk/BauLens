"""Load the UCI Concrete Compressive Strength table."""

from pathlib import Path

import numpy as np
import pandas as pd

from baulens.paths import RAW_DATA_DIR

DEFAULT_CONCRETE_PATH = RAW_DATA_DIR / "concrete" / "Concrete_Data.xls"
CONCRETE_COLUMNS: tuple[str, ...] = (
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age_days",
    "compressive_strength_mpa",
)
_WORKBOOK_COLUMNS = (*CONCRETE_COLUMNS[:-2], "age", "concrete_compressive_strength")


def load_concrete(path: Path = DEFAULT_CONCRETE_PATH) -> pd.DataFrame:
    """Read the UCI .xls workbook or a CSV with canonical column names.

    The first seven columns are ingredients in kg/m³. Age is measured in days
    and the target, compressive strength, in MPa.
    """
    match path.suffix.lower():
        case ".xls":
            frame = pd.read_excel(path, engine="calamine")
            source_columns = tuple(
                str(header).split("(", 1)[0].strip().lower().replace(" ", "_")
                for header in frame.columns
            )
            if source_columns != _WORKBOOK_COLUMNS:
                raise ValueError(
                    f"Unexpected concrete workbook columns in {path}: {source_columns}"
                )
            frame.columns = pd.Index(CONCRETE_COLUMNS)
        case ".csv":
            frame = pd.read_csv(path)
            if tuple(frame.columns) != CONCRETE_COLUMNS:
                raise ValueError(f"Expected concrete CSV columns in {path}: {CONCRETE_COLUMNS}")
        case _:
            raise ValueError(f"Expected an .xls or .csv concrete table: {path}")

    if (
        frame.empty
        or frame.isna().any().any()
        or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
    ):
        raise ValueError(f"Expected a complete numeric concrete table: {path}")
    non_finite = [str(column) for column in frame.columns if not np.isfinite(frame[column]).all()]
    if non_finite:
        raise ValueError(f"Non-finite values in concrete table {path}: columns {non_finite}")
    return frame
