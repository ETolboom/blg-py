import functools
import logging

import torch
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

logger = logging.getLogger(__name__)


def load_model() -> None:
    """Load the sentence transformer model. Must be called before using similarity functions."""
    global _model
    if _model is None:
        logger.info("Loading sentence transformer model...")
        _model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", cache_folder="./cache"
        )
        logger.info("Sentence transformer model loaded successfully")


def _get_model() -> SentenceTransformer:
    """Get the loaded model, raising an error if not loaded."""
    if _model is None:
        raise RuntimeError(
            "Sentence transformer model not loaded. Call load_model() first."
        )
    return _model


@functools.lru_cache(maxsize=512)
def _get_embedding(label: str) -> torch.Tensor:
    """Return the cached normalized embedding for a single label string."""
    return torch.tensor(_get_model().encode(label, normalize_embeddings=True))


def _embed(labels: list[str]) -> torch.Tensor:
    """Stack cached embeddings for a list of labels into a 2-D tensor."""
    return torch.stack([_get_embedding(label) for label in labels])


def create_similarity_matrix(
        target_labels: list[str],
        reference_labels: list[str],
        self_similarity: bool = False,
) -> torch.Tensor:
    """Compute a cosine-similarity matrix between target and reference label embeddings."""
    similarity_matrix = torch.mm(_embed(target_labels), _embed(reference_labels).t())

    if self_similarity:
        if reference_labels != target_labels:
            logger.warning("The labels do not match, are you sure that you want to evaluate self-similarity?")
        # When dealing with self-similarity such as with duplicate tasks
        similarity_matrix.fill_diagonal_(-1)

    return similarity_matrix


def match_labels(
        target: list[str],
        reference: list[str],
        match_threshold: float,
) -> list[tuple[int, int]]:
    """Greedily match target labels to reference labels above the similarity threshold."""
    similarity_matrix = create_similarity_matrix(target, reference)
    ranked_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

    matched_ref_indices: set[int] = set()
    matches: list[tuple[int, int]] = []

    for target_idx in range(len(target)):
        for ref_idx in ranked_indices[target_idx].tolist():
            if ref_idx in matched_ref_indices:
                continue
            score: float = similarity_matrix[target_idx, ref_idx].item()
            if score < match_threshold:
                break
            matched_ref_indices.add(ref_idx)
            matches.append((target_idx, ref_idx))
            break

    return matches