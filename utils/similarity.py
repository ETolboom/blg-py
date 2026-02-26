import torch
from sentence_transformers import SentenceTransformer

model: SentenceTransformer = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", cache_folder="./cache"
)


def _embed(labels: list[str]) -> torch.Tensor:
    return torch.tensor(model.encode(labels, normalize_embeddings=True))


def create_similarity_matrix(
        target_labels: list[str],
        reference_labels: list[str],
        self_similarity: bool = False,
) -> torch.Tensor:
    similarity_matrix = torch.mm(_embed(target_labels), _embed(reference_labels).t())

    if self_similarity:
        if reference_labels != target_labels:
            print("WARN: The labels do not match, are you sure you want self-similarity?")
        similarity_matrix.fill_diagonal_(-1)

    return similarity_matrix


def match_labels(
        target: list[str],
        reference: list[str],
        match_threshold: float,
) -> list[tuple[int, int]]:
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