import numpy as np
from typing import Literal

from core.normalizer import normalize
from utils.config import (
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    SCORE_GAP_THRESHOLD,
    SEMANTIC_WEIGHT,
    LEXICAL_WEIGHT,
)

# Optional FAISS (safe import)
try:
    from core.faiss_index import FaissIndex
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


Decision = Literal["answer", "suggest", "fallback"]


class Retriever:
    """
    Production-grade FAQ Retriever
    --------------------------------
    Features:
    - Query normalization
    - Semantic + lexical hybrid scoring
    - Context-aware (subcategory-first) search
    - Score-gap based ambiguity detection
    - FAISS acceleration (optional, Phase 4)
    """

    def __init__(self, df, embedder, use_faiss: bool = True):
        # ---- Data
        self.df = df.reset_index(drop=True)
        self.embedder = embedder

        # ---- Normalize FAQ questions once
        self.df["norm_question"] = self.df["question"].apply(normalize)

        # ---- Precompute embeddings ONCE
        self.embeddings = self.embedder.encode(
            self.df["norm_question"].tolist()
        )

        # ---- FAISS index (semantic only)
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index = None

        if self.use_faiss:
            self.faiss_index = FaissIndex(self.embeddings)

    # =====================================================
    # Internal helpers
    # =====================================================
    def _lexical_overlap(self, query_norm: str, question_norm: str) -> float:
        """
        Lightweight lexical overlap score.
        Safe, deterministic, no keyword weighting hell.
        """
        q_tokens = set(query_norm.split())
        d_tokens = set(question_norm.split())

        if not q_tokens:
            return 0.0

        return len(q_tokens & d_tokens) / len(q_tokens)

    def _semantic_candidates(self, query_vec: np.ndarray, top_k: int = 50):
        """
        Phase 4:
        Use FAISS (if available) to fetch top semantic candidates.
        Falls back to linear scan if FAISS is disabled.
        """
        if self.use_faiss and self.faiss_index is not None:
            _, idxs = self.faiss_index.search(query_vec, top_k)
            return self.df.iloc[idxs]

        # Linear fallback (safe, slower)
        scores = self.embeddings @ query_vec
        idxs = np.argsort(scores)[::-1][:top_k]
        return self.df.iloc[idxs]

    def _rank(self, query_norm: str, query_vec: np.ndarray, df_subset):
        """
        Hybrid ranking:
        semantic similarity + lexical overlap
        """
        idxs = df_subset.index.to_numpy()
        emb_subset = self.embeddings[idxs]

        # Semantic similarity
        semantic_scores = emb_subset @ query_vec

        # Lexical overlap
        lexical_scores = np.array([
            self._lexical_overlap(query_norm, q)
            for q in df_subset["norm_question"]
        ])

        # Final hybrid score
        final_scores = (
            SEMANTIC_WEIGHT * semantic_scores
            + LEXICAL_WEIGHT * lexical_scores
        )

        ranked = df_subset.copy()
        ranked["score"] = final_scores

        return ranked.sort_values("score", ascending=False)

    # =====================================================
    # Public API
    # =====================================================
    def semantic_search(self, query: str, top_k: int = 5):
        """
        Used for:
        - Live suggestions while typing
        - UI hinting

        This does NOT make decisions.
        It only returns top-k ranked candidates safely.
        """
        query_norm = normalize(query)
        query_vec = self.embedder.encode([query_norm])[0]

        # Use global semantic candidates (Phase 4 safe)
        candidates = self._semantic_candidates(query_vec, top_k=top_k * 5)

        ranked = self._rank(query_norm, query_vec, candidates)

        return ranked.head(top_k)

    def search_with_context(self, query: str, memory):
        """
        Multi-layer retrieval:
        1. Subcategory-first (if context exists)
        2. Global fallback
        """
        query_norm = normalize(query)
        query_vec = self.embedder.encode([query_norm])[0]

        # ---- Layer 1: Context-narrowed search
        if memory.active_subcategory:
            sub_df = self.df[
                self.df["subcategory"] == memory.active_subcategory
            ]
            if not sub_df.empty:
                ranked = self._rank(query_norm, query_vec, sub_df)
                if ranked.iloc[0].score >= MEDIUM_THRESHOLD:
                    return ranked

        # ---- Layer 2: Global semantic candidate selection
        candidates = self._semantic_candidates(query_vec, top_k=50)
        return self._rank(query_norm, query_vec, candidates)

    
    def decide(self, ranked_df) -> Decision:
        """
        Phase 1 confidence logic:
        - Uses score thresholds
        - Uses score GAP to detect ambiguity
        """
        if ranked_df.empty:
            return "fallback"

        if len(ranked_df) == 1:
            return "answer"

        s1 = ranked_df.iloc[0].score
        s2 = ranked_df.iloc[1].score
        gap = s1 - s2

        if s1 >= HIGH_THRESHOLD and gap >= SCORE_GAP_THRESHOLD:
            return "answer"

        if s1 >= MEDIUM_THRESHOLD:
            return "suggest"

        return "fallback"
