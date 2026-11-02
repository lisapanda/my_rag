"""Dense Chroma retrieval, local BM25 retrieval, and hybrid rank fusion."""
from collections import Counter
import math
import re
import unicodedata


def normalize_title(text):
    """Make Unicode formatting and letter case consistent for matching."""
    # NFKC normalizes compatibility characters (e.g. ½ becomes 1⁄2).
    # casefold() lets titles match regardless of capitalization.
    return unicodedata.normalize("NFKC", text).casefold()


def match_movie_title(question, titles):
    """Return the original stored title only when exactly one title matches."""
    normalized = normalize_title(question)
    # Search for each known title as literal text, not as a regex expression.
    # The boundary checks prevent "The Matrix" from matching "The Matrixes".
    # Keep the original title so the metadata filter uses its stored spelling.
    matches = [title for title in titles if re.search(
        r"(?<!\w)" + re.escape(normalize_title(title)) + r"(?!\w)", normalized
    )]
    # Multiple movies can indicate a comparison; keep that query unrestricted.
    return matches[0] if len(matches) == 1 else None


def _tokens(text):
    # Keep Unicode letters and numbers, including accented movie/actor names.
    return re.findall(r"\w+", normalize_title(text))


def _bm25_rank(question, ids, documents, limit):
    """Rank stored documents lexically using BM25 (k1=1.5, b=0.75)."""
    counts = [Counter(_tokens(document)) for document in documents]
    lengths = [sum(count.values()) for count in counts]
    average_length = sum(lengths) / len(lengths) if lengths else 0
    if not average_length:
        return []
    query_terms = set(_tokens(question))
    frequencies = Counter(term for count in counts for term in count if term in query_terms)
    scored = []
    for doc_id, count, length in zip(ids, counts, lengths):
        score = 0.0
        for term in query_terms:
            tf = count[term]
            if not tf:
                continue
            df = frequencies[term]
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average_length))
        # Do not return arbitrary zero-score documents for a keyword search.
        if score > 0:
            scored.append((doc_id, score))
    return [doc_id for doc_id, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]]


def _load_candidate_documents(collection, filters):
    """Load text for BM25, dropping the title filter only if it matches no IDs."""
    # Read all candidates so sparse search can find facts absent from dense hits.
    # Reload on each query to reflect collection updates; intended for small corpora.
    corpus = collection.get(include=["documents"], **filters)
    if not corpus["ids"] and filters:
        filters = {}
        corpus = collection.get(include=["documents"])
    texts = {doc_id: document for doc_id, document in
             zip(corpus["ids"], corpus["documents"] or []) if document}
    return texts, filters


def query_movie_documents(collection, question, titles, n_results=10, mode="hybrid"):
    """Return text using dense, sparse (BM25), or hybrid retrieval.

    Local Chroma handles dense search. BM25 runs in Python over documents read
    from Chroma; it is not Chroma's server-side sparse-vector Search API.
    Hybrid uses equal-weight reciprocal rank fusion rather than mixing unlike
    vector-distance and BM25 scores. Existing callers use hybrid by default.
    """
    if mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError("mode must be 'dense', 'sparse', or 'hybrid'")
    if n_results < 1:
        raise ValueError("n_results must be positive")
    title = match_movie_title(question, titles)
    filters = {"where": {"movie_title": title}} if title else {}

    # Mode 1: semantic similarity using Chroma's dense vectors.
    if mode == "dense":
        # Preserve the original dense-only behavior and empty-filter fallback.
        result = collection.query(query_texts=[question], n_results=n_results,
                                  include=["documents"], **filters)
        documents = (result["documents"] or [[]])[0]
        if documents or not filters:
            return documents
        return collection.query(query_texts=[question], n_results=n_results,
                                include=["documents"])["documents"][0]

    # Mode 2: keyword relevance using BM25, with no dense query.
    elif mode == "sparse":
        texts, _ = _load_candidate_documents(collection, filters)
        if not texts:
            return []
        sparse_ids = _bm25_rank(question, list(texts), list(texts.values()), n_results)
        return [texts[doc_id] for doc_id in sparse_ids[:n_results]]

    # Mode 3: combine dense and sparse rankings using reciprocal rank fusion.
    elif mode == "hybrid":
        texts, filters = _load_candidate_documents(collection, filters)
        if not texts:
            return []
        # Retrieve extra candidates from each method before selecting the final set.
        candidate_limit = min(len(texts), n_results * 3)
        sparse_ids = _bm25_rank(
            question, list(texts), list(texts.values()), candidate_limit
        )
        dense = collection.query(query_texts=[question], n_results=candidate_limit,
                                 include=["documents"], **filters)
        dense_ids = dense["ids"][0]
        # RRF rewards agreement without mixing incompatible raw relevance scores.
        scores = Counter()
        for ranking in (dense_ids, sparse_ids):
            for rank, doc_id in enumerate(ranking, start=1):
                if doc_id in texts:
                    scores[doc_id] += 1 / (60 + rank)
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [texts[doc_id] for doc_id in ranked[:n_results]]
