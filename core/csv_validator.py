import pandas as pd

REQUIRED_COLUMNS = {"category", "subcategory", "question", "answer"}

def validate_faq_csv(df: pd.DataFrame):
    errors = []

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if df["question"].isnull().any():
        errors.append("Some questions are empty")

    if df["answer"].isnull().any():
        errors.append("Some answers are empty")

    if df.empty:
        errors.append("CSV file is empty")

    return errors
