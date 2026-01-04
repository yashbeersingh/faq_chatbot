import pandas as pd
from core.embedder import Embedder
from core.retriever import Retriever

def rebuild_retriever(csv_path: str):
    df = pd.read_csv(csv_path)
    embedder = Embedder()
    retriever = Retriever(df, embedder)
    return df, retriever
