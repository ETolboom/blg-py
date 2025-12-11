from collections import defaultdict
from typing import ClassVar

import spacy
import torch
from thefuzz import fuzz

from algorithms import (
    Algorithm,
    AlgorithmComplexity,
    AlgorithmFormInput,
    AlgorithmResult,
)
from utils import extract_all_tasks, get_elements_by_type, ExtractedTask
from utils.similarity import create_similarity_matrix


class AtomicityCheck(Algorithm):
    id: ClassVar[str] = "atomicity_check"
    name: ClassVar[str] = "Label Atomicity"
    description: ClassVar[str] = "Check the task labels for atomicity"
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.SIMPLE
    threshold: ClassVar[float] = 0.85

    def analyze(
        self, inputs: list[AlgorithmFormInput] | None = None
    ) -> AlgorithmResult:
        tasks: list[ExtractedTask] = extract_all_tasks(self.model_xml)

        problematic_elements = []
        for task in tasks:
            single_action = check_single_action(task.name)
            atomicity = atomicity_score(task.name)
            if not (single_action or atomicity >= self.threshold):
                problematic_elements.append(task.id)

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class ExactDuplicateTasks(Algorithm):
    id: ClassVar[str] = "exact_duplicate_tasks"
    name: ClassVar[str] = "Exact Duplicate Tasks"
    description: ClassVar[str] = (
        "Check the model for any duplicate tasks based on fuzzy matching"
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.SIMPLE
    threshold: ClassVar[float] = 0.90

    def analyze(
        self, inputs: list[AlgorithmFormInput] | None = None
    ) -> AlgorithmResult:
        tasks: list[ExtractedTask] = extract_all_tasks(self.model_xml)

        if len(tasks) == 0:
            raise Exception("Cannot identify exact duplicates: no tasks found")

        problematic_elements = []

        def parse_duplicates(duplicates: dict[str, list[ExtractedTask]]):
            for key, group in duplicates.items():
                # print(f"Duplicate for {key}: {group}")
                pair_one_element_id = group[0].id
                if pair_one_element_id not in problematic_elements:
                    problematic_elements.append(pair_one_element_id)

                pair_two_element_id = group[1].id
                if pair_two_element_id not in problematic_elements:
                    problematic_elements.append(pair_two_element_id)

        parse_duplicates(find_fuzzy_duplicates(tasks, threshold=self.threshold))

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class SemanticDuplicateTasks(Algorithm):
    id: ClassVar[str] = "semantic_duplicate_tasks"
    name: ClassVar[str] = "Semantically Duplicate Tasks"
    description: ClassVar[str] = (
        "Check the model for any duplicate tasks based on semantic matching"
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.SIMPLE
    threshold: ClassVar[float] = 0.75

    def analyze(
        self, inputs: list[AlgorithmFormInput] | None = None
    ) -> AlgorithmResult:
        tasks: list[ExtractedTask] = extract_all_tasks(self.model_xml)
        if len(tasks) == 0:
            raise Exception("Cannot identify exact duplicates: no tasks found")

        problematic_elements = []

        def parse_duplicates(duplicates):
            for key, group in duplicates.items():
                # print(f"Duplicate for {key}: {group}")
                pair_one_element_id = group[0].id
                if pair_one_element_id not in problematic_elements:
                    problematic_elements.append(pair_one_element_id)

                pair_two_element_id = group[1].id
                if pair_two_element_id not in problematic_elements:
                    problematic_elements.append(pair_two_element_id)

        parse_duplicates(find_semantic_duplicates(tasks, threshold=self.threshold))

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


# Helpers

nlp = spacy.load("en_core_web_md")


def check_single_action(label: str) -> bool:
    doc = nlp(label)
    verbs = [token for token in doc if token.pos_ == "VERB"]
    return len(verbs) <= 1


def atomicity_score(label: str) -> float:
    words = label.split()
    conjunction_words = ["and", "or", "then", "after", "also"]

    penalties: float = 0.0
    penalties += len(words) * 0.1
    penalties += sum(1 for word in words if word.lower() in conjunction_words) * 2
    penalties /= 10  # Scale back to 0-1

    return max(0, 1 - penalties)


def find_semantic_duplicates(
    extracted_tasks: list[ExtractedTask], threshold: float
) -> dict:
    # Extract just task label
    labels = [t.name for t in extracted_tasks]

    similarity_matrix = create_similarity_matrix(labels, labels, self_similarity=True)

    # Find most similar vector for each embedding
    most_similar_indices = torch.argmax(similarity_matrix, dim=1)
    most_similar_scores = torch.max(similarity_matrix, dim=1)[0]

    # Create pairs
    pairs = []
    for i in range(len(labels)):
        similar_idx = most_similar_indices[i].item()
        similarity_score = most_similar_scores[i].item()
        pairs.append((i, similar_idx, similarity_score))

    groups = defaultdict(list)
    used_indices = set()
    duplicate_pairs = []
    processed_pairs = set()

    for idx1, idx2, score in pairs:
        pair_key = (min(idx1, idx2), max(idx1, idx2))

        if idx1 not in used_indices and idx2 not in used_indices:
            if idx1 != idx2 and score >= threshold and pair_key not in processed_pairs:
                used_indices.add(idx1)
                used_indices.add(idx2)
                processed_pairs.add(pair_key)
                duplicate_pairs.append((idx1, idx2, score))

    for idx1, idx2, _ in duplicate_pairs:
        original = extracted_tasks[idx1]
        group = [original, extracted_tasks[idx2]]
        groups[original.id] = group

    return groups


def find_fuzzy_duplicates(tasks: list[ExtractedTask], threshold: float) -> dict:
    groups = defaultdict(list)
    processed = set()
    threshold *= 100  # Thefuzz using values between 0-100

    for i, current_task in enumerate(tasks):
        if i in processed:
            continue

        current_key = current_task.id
        group = [current_task]
        processed.add(i)

        # Compare with remaining tuples
        for j, other_task in enumerate(tasks[i + 1 :], i + 1):
            if j in processed:
                continue

            other_key = other_task.id
            similarity = fuzz.ratio(current_key, other_key)

            if similarity >= threshold:
                group.append(other_task)
                processed.add(j)

        if len(group) > 1:  # Only keep groups with duplicates
            groups[current_key] = group

    return dict(groups)
