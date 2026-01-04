import streamlit as st
import pandas as pd
from utils.config import ANALYTICS_LOG_PATH

REQUIRED_COLUMNS = {
    "timestamp",
    "query",
    "decision",
    "top_score",
    "second_score",
    "score_gap",
    "chosen_faq",
    "subcategory",
}

def show_dashboard():
    st.header("📊 FAQ Search Analytics")

    try:
        df = pd.read_csv(ANALYTICS_LOG_PATH)
    except FileNotFoundError:
        st.info("No analytics data yet.")
        return
    st.write("READING FROM:", ANALYTICS_LOG_PATH)

    # -------------------------------
    # Schema validation
    # -------------------------------
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        st.warning(
            "⚠️ Analytics data is from an older version.\n\n"
            f"Missing columns: {', '.join(missing_cols)}\n\n"
            "New analytics will be recorded correctly going forward."
        )

        # Add missing columns as empty so dashboard doesn’t crash
        for col in missing_cols:
            df[col] = None

    # -------------------------------
    # Decision Distribution
    # -------------------------------
    st.subheader("Decision Distribution")
    if df["decision"].notna().any():
        st.bar_chart(df["decision"].value_counts())
    else:
        st.info("No decision data available yet.")

    # -------------------------------
    # Fallback Queries
    # -------------------------------
    st.subheader("Fallback Queries")
    fallback_df = df[df["decision"] == "fallback"]
    if not fallback_df.empty:
        st.dataframe(
            fallback_df.sort_values("top_score").head(20)
        )
    else:
        st.info("No fallback queries recorded.")

    # -------------------------------
    # Ambiguous Queries (Low Score Gap)
    # -------------------------------
    st.subheader("Ambiguous Queries")
    ambiguous = df[
        (df["decision"] == "suggest")
        & (df["score_gap"].notna())
        & (df["score_gap"] < 0.08)
    ]
    if not ambiguous.empty:
        st.dataframe(
            ambiguous.sort_values("score_gap").head(20)
        )
    else:
        st.info("No ambiguous queries recorded.")

    # -------------------------------
    # Top Repeated Queries
    # -------------------------------
    st.subheader("Top Repeated Queries")
    st.dataframe(
        df["query"].value_counts().head(20)
    )
