import pandas as pd

def load_faqs(path):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df
