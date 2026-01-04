import pandas as pd
from utils.config import FAQ_PATH

def merge_faqs(new_df: pd.DataFrame):
    """
    Merge new FAQs into the master FAQ CSV.
    Avoids duplicates based on (category, subcategory, question).
    """
    try:
        existing_df = pd.read_csv(FAQ_PATH)
    except FileNotFoundError:
        new_df.to_csv(FAQ_PATH, index=False)
        return

    combined = pd.concat([existing_df, new_df], ignore_index=True)

    combined.drop_duplicates(
        subset=["category", "subcategory", "question"],
        keep="last",
        inplace=True
    )

    combined.to_csv(FAQ_PATH, index=False)
