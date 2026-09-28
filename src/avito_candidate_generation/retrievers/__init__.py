"""Retrieval implementations."""

from .bm25 import BM25Index, retrieve_bm25
from .routed_e5 import E5QueryEncoder, RoutedDenseE5Retriever
from .routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
    extract_canonical_numbers,
    normalize_lexical_text,
    numeric_matching_rank,
    retrieve_routed,
)
from .tfidf import TFIDFIndex, retrieve_tfidf

__all__ = [
    "BM25Index",
    "E5QueryEncoder",
    "FieldAwareSparseIndex",
    "RoutedDenseE5Retriever",
    "SparseBranchIndex",
    "TFIDFIndex",
    "extract_canonical_numbers",
    "normalize_lexical_text",
    "numeric_matching_rank",
    "retrieve_bm25",
    "retrieve_routed",
    "retrieve_tfidf",
]
