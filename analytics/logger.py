import csv
from datetime import datetime
from utils.config import ANALYTICS_LOG_PATH

HEADER = [
    "timestamp",
    "query",
    "decision",
    "top_score",
    "second_score",
    "score_gap",
    "chosen_faq",
    "subcategory",
]

def init_analytics_file():
    try:
        with open(ANALYTICS_LOG_PATH, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
    except FileExistsError:
        pass


def log_event(
    query: str,
    decision: str,
    scores: list,
    chosen_faq: str | None,
    subcategory: str | None,
    
):
    top_score = scores[0] if len(scores) > 0 else None
    second_score = scores[1] if len(scores) > 1 else None
    gap = (
        top_score - second_score
        if top_score is not None and second_score is not None
        else None
    )
    print("LOGGING TO:", ANALYTICS_LOG_PATH)
    with open(ANALYTICS_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            query,
            decision,
            round(top_score, 4) if top_score else None,
            round(second_score, 4) if second_score else None,
            round(gap, 4) if gap else None,
            chosen_faq,
            subcategory,
        ])


def log_unanswered(query: str):
    """Log a query that couldn't be answered by the FAQ system."""
    log_event(
        query=query,
        decision="unanswered",
        scores=[],
        chosen_faq=None,
        subcategory=None,
    )
