from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir="./cache")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir="./cache")

def create_similarity_matrix(target_labels: List[str], reference_labels: List[str], self_similarity=False):
    target_inputs = tokenizer(target_labels, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        target_embeddings = model(**target_inputs).last_hidden_state.mean(dim=1)
    normalized_target_embeddings = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)

    reference_inputs = tokenizer(reference_labels, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        reference_embeddings = model(**reference_inputs).last_hidden_state.mean(dim=1)

    normalized_reference_embeddings = torch.nn.functional.normalize(reference_embeddings, p=2, dim=1)

    similarity_matrix = torch.mm(normalized_target_embeddings, normalized_reference_embeddings.t())

    if self_similarity:
        if reference_labels != target_labels:
            print("WARN: The labels do not match, are you sure that you want to evaluate self-similarity?")
        # When dealing with self-similarity such as with duplicate tasks
        similarity_matrix.fill_diagonal_(-1)

    # Compute cosine similarity matrix
    return similarity_matrix


def match_labels(target, reference, match_threshold):
    similarity_matrix = create_similarity_matrix(target, reference)

    matched_indices = set()

    matches = []

    for target_idx in range(len(target)):
        best_reference_idx = None
        best_score = -1

        for reference_idx in range(len(reference)):
            if reference_idx in matched_indices:
                continue

            score = similarity_matrix[target_idx, reference_idx]
            if score > best_score:
                best_score = score
                best_reference_idx = reference_idx

        if best_score < match_threshold:
            continue

        matched_indices.add(best_reference_idx)
        matches.append((target_idx, best_reference_idx))

    return matches