import re

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "to", "for", "of", "in", "on"
}

ABBREVIATIONS = {
    "hl": "home loan",
    "sa": "savings account",
}

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    words = []
    for w in text.split():
        w = ABBREVIATIONS.get(w, w)
        if w not in STOPWORDS:
            words.append(w)

    return " ".join(words)
