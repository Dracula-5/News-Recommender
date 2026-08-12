"""
Enhanced NLP engine: BM25 + TF-IDF + LSA semantic embeddings.

Three complementary retrievers are fused into a single relevance score:

  BM25   (40%) — term-frequency saturation, document-length normalisation
  TF-IDF (40%) — cosine similarity with sublinear TF, tri-gram features
  LSA    (20%) — 64-dim truncated-SVD embedding catches semantic overlap

All three are fit once on the full article corpus (up to 20k docs) and
cached in memory.  `score_candidates` is the main public method.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Tokeniser (shared)
# ──────────────────────────────────────────────────────────

_STOP = frozenset(
    "a an the and or but in on at to for of with by from is was are were be been "
    "being have has had do does did will would could should may might shall can "
    "this that these those it its we us our they them their i me my you your he "
    "him his she her not no nor so yet both either neither each few more most "
    "such up out about after before if when as while since then there here all "
    "any some who whom which what where how than too also just only even though "
    "therefore however because since".split()
)


def _tokenise(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in _STOP]


# ──────────────────────────────────────────────────────────
#  BM25 (Okapi BM25, k1=1.5, b=0.75)
# ──────────────────────────────────────────────────────────

class BM25Retriever:
    """Pure-Python BM25 — no extra dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self._tokenised: List[List[str]] = []
        self._doc_freqs: List[Dict[str, int]] = []
        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 0.0

    def fit(self, corpus: List[str], doc_ids: List[str]) -> None:
        self.doc_ids = list(doc_ids)
        self._tokenised = [_tokenise(d) for d in corpus]
        n = len(self._tokenised)
        self._avg_dl = sum(len(d) for d in self._tokenised) / max(1, n)

        df: Dict[str, int] = defaultdict(int)
        self._doc_freqs = []
        for tokens in self._tokenised:
            freq: Dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            self._doc_freqs.append(dict(freq))
            for t in set(tokens):
                df[t] += 1

        self._idf = {
            t: math.log((n - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def score(self, query: str, idx: int) -> float:
        query_tokens = _tokenise(query)
        freq = self._doc_freqs[idx]
        dl = len(self._tokenised[idx])
        s = 0.0
        for t in query_tokens:
            if t not in freq:
                continue
            tf = freq[t]
            idf = self._idf.get(t, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1, self._avg_dl))
            s += idf * tf * (self.k1 + 1) / denom
        return s

    def batch_score(self, query: str, indices: List[int]) -> Dict[int, float]:
        return {i: self.score(query, i) for i in indices}


# ──────────────────────────────────────────────────────────
#  Enhanced NLP Engine
# ──────────────────────────────────────────────────────────

class EnhancedNLPEngine:
    """
    Three-component NLP ensemble:
      - TF-IDF (ngram 1-3, sublinear TF, 15k features)
      - BM25
      - LSA (64-dim SVD of TF-IDF matrix)

    Usage:
        engine = EnhancedNLPEngine()
        engine.fit(texts, doc_ids)
        scores = engine.score_candidates(query_text, candidate_ids)  # dict
        emb    = engine.get_lsa_embedding(text)                       # np.ndarray
    """

    def __init__(self, n_lsa: int = 64) -> None:
        self.n_lsa = n_lsa
        self.bm25 = BM25Retriever()
        self._tfidf = None          # sklearn TfidfVectorizer
        self._svd = None            # sklearn TruncatedSVD
        self._tfidf_matrix = None   # sparse, shape (n_docs, vocab)
        self._lsa_matrix = None     # dense  (n_docs, n_lsa), L2-normalised
        self._doc_ids: List[str] = []
        self._doc_id_to_idx: Dict[str, int] = {}
        self.fitted = False

    # ──────────────────────────────────────────────────────
    #  Fit
    # ──────────────────────────────────────────────────────

    def fit(self, corpus: List[str], doc_ids: List[str]) -> None:
        """Fit all three components on the article corpus."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize
        except ImportError:
            logger.warning("scikit-learn not available — NLP engine disabled")
            return

        self._doc_ids = list(doc_ids)
        self._doc_id_to_idx = {d: i for i, d in enumerate(self._doc_ids)}

        # TF-IDF
        self._tfidf = TfidfVectorizer(
            max_features=15_000,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(corpus)
        logger.info("TF-IDF fitted: %d docs, %d features", *self._tfidf_matrix.shape)

        # LSA (truncated SVD on TF-IDF)
        n_comp = min(self.n_lsa, self._tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        raw_lsa = self._svd.fit_transform(self._tfidf_matrix)
        self._lsa_matrix = normalize(raw_lsa)
        logger.info("LSA fitted: %d dims", n_comp)

        # BM25
        self.bm25.fit(corpus, doc_ids)
        logger.info("BM25 fitted: %d docs", len(doc_ids))

        self.fitted = True

    # ──────────────────────────────────────────────────────
    #  Score candidates
    # ──────────────────────────────────────────────────────

    def score_candidates(
        self,
        query_text: str,
        candidate_ids: List[str],
    ) -> Dict[str, float]:
        """
        Ensemble score for each candidate article.
        Returns dict  news_id → score  (higher = more relevant, roughly [0,1]).
        """
        if not self.fitted or not candidate_ids:
            return {nid: 0.0 for nid in candidate_ids}

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import normalize

            indices = [
                self._doc_id_to_idx[nid]
                for nid in candidate_ids
                if nid in self._doc_id_to_idx
            ]
            if not indices:
                return {nid: 0.0 for nid in candidate_ids}

            idx_to_cid = {
                self._doc_id_to_idx[nid]: nid
                for nid in candidate_ids
                if nid in self._doc_id_to_idx
            }

            # — TF-IDF cosine —
            q_tfidf = self._tfidf.transform([query_text])
            tfidf_sims = cosine_similarity(
                q_tfidf, self._tfidf_matrix[indices]
            ).ravel()
            tfidf_scores = {idx_to_cid[indices[i]]: float(tfidf_sims[i]) for i in range(len(indices))}

            # — LSA cosine —
            q_lsa = normalize(self._svd.transform(q_tfidf))
            lsa_sims = (self._lsa_matrix[indices] @ q_lsa.T).ravel()
            lsa_scores = {idx_to_cid[indices[i]]: max(0.0, float(lsa_sims[i])) for i in range(len(indices))}

            # — BM25 —
            bm25_raw = {idx_to_cid[i]: self.bm25.score(query_text, i) for i in indices}
            max_bm25 = max(bm25_raw.values()) if bm25_raw else 1.0
            bm25_scores = {k: v / max(max_bm25, 1e-9) for k, v in bm25_raw.items()}

            # — Ensemble: 0.40·BM25 + 0.40·TF-IDF + 0.20·LSA —
            result: Dict[str, float] = {}
            for nid in candidate_ids:
                b = bm25_scores.get(nid, 0.0)
                t = tfidf_scores.get(nid, 0.0)
                l = lsa_scores.get(nid, 0.0)
                result[nid] = 0.40 * b + 0.40 * t + 0.20 * l
            return result

        except Exception as exc:
            logger.warning("NLP score_candidates failed: %s", exc)
            return {nid: 0.0 for nid in candidate_ids}

    # ──────────────────────────────────────────────────────
    #  Embedding helpers
    # ──────────────────────────────────────────────────────

    def get_lsa_embedding(self, text: str) -> Optional[np.ndarray]:
        """Return the LSA embedding vector for an arbitrary text."""
        if not self.fitted:
            return None
        try:
            from sklearn.preprocessing import normalize
            vec = self._tfidf.transform([text])
            emb = normalize(self._svd.transform(vec))
            return emb[0]
        except Exception:
            return None

    def get_tfidf_vector(self, text: str):
        """Return the sparse TF-IDF vector for an arbitrary text."""
        if not self.fitted:
            return None
        try:
            return self._tfidf.transform([text])
        except Exception:
            return None

    def lsa_vectors_for_ids(self, doc_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Look up the already-fitted LSA embedding for each doc_id (no
        re-transform needed — used by MMR re-ranking to measure content
        similarity *between candidates*, as opposed to `score_candidates`
        which measures similarity to the user's query text).
        """
        if not self.fitted:
            return {}
        out: Dict[str, np.ndarray] = {}
        for nid in doc_ids:
            idx = self._doc_id_to_idx.get(nid)
            if idx is not None:
                out[nid] = self._lsa_matrix[idx]
        return out
