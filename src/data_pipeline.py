
from pathlib import Path
import pandas as pd

NUMERIC_COLS = [
    "Age",
    "Biomarker_Level",
    "Vaccine_Dose_mcg",
]

CATEGORICAL_COL = "Prior_Reaction_History"
TARGET_COL = "Reaction_Score"


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load the patient dataset and handle missing feature values."""

    df = pd.read_csv(path)

    required_columns = [
        "Patient_ID",
        "Age",
        "Biomarker_Level",
        "Pre_Existing_Conditions",
        "Vaccine_Dose_mcg",
        "Prior_Reaction_History",
        "Reaction_Score",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    # Fill missing numeric feature values with their medians.
    for column in NUMERIC_COLS:
        df[column] = df[column].fillna(df[column].median())

    # Fill missing reaction history with the most common value.
    mode_values = df[CATEGORICAL_COL].mode()
    default_value = mode_values.iloc[0] if not mode_values.empty else "None"

    df[CATEGORICAL_COL] = df[CATEGORICAL_COL].fillna(default_value)

    # Fill missing condition counts if present.
    df["Pre_Existing_Conditions"] = df[
        "Pre_Existing_Conditions"
    ].fillna(df["Pre_Existing_Conditions"].median())

    return df
