from collections import defaultdict

import spacy
import torch
from fuzzywuzzy import fuzz

from algorithms import *
from utils import get_elements_by_type
from utils.similarity import create_similarity_matrix

algorithm_category = AlgorithmKind.SEMANTIC


class AtomicityCheck(Algorithm):
    id: ClassVar[str] = "atomicity_check"
    name: ClassVar[str] = "Label Atomicity"
    description: ClassVar[str] = "Check the task labels for atomicity"
    algorithm_kind: ClassVar[AlgorithmKind] = algorithm_category
    threshold: ClassVar[float] = 0.85

    def analyze(self, inputs=None) -> AlgorithmResult:
        tasks: List[(str, str)] = get_elements_by_type(self.model_xml, "task")
        problematic_elements = []
        for (label, element_id) in tasks:
            single_action = check_single_action(label)
            atomicity = atomicity_score(label)
            if not (single_action or atomicity >= self.threshold):
                problematic_elements.append(element_id)

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_type,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class ExactDuplicateTasks(Algorithm):
    id: ClassVar[str] = "exact_duplicate_tasks"
    name: ClassVar[str] = "Exact Duplicate Tasks"
    description: ClassVar[str] = "Check the model for any duplicate tasks based on fuzzy matching"
    algorithm_kind: ClassVar[AlgorithmKind] = algorithm_category
    threshold: ClassVar[float] = 0.90

    def analyze(self, inputs=None) -> AlgorithmResult:
        tasks = get_elements_by_type(self.model_xml, "task")

        problematic_elements = []

        def parse_duplicates(duplicates):
            for key, group in duplicates.items():
                # print(f"Duplicate for {key}: {group}")
                pair_one_element_id = group[0][1]
                if pair_one_element_id not in problematic_elements:
                    problematic_elements.append(pair_one_element_id)

                pair_two_element_id = group[1][1]
                if pair_two_element_id not in problematic_elements:
                    problematic_elements.append(pair_two_element_id)

        parse_duplicates(find_fuzzy_duplicates(tasks, threshold=self.threshold))

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_type,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class SemanticDuplicateTasks(Algorithm):
    id: ClassVar[str] = "semantic_duplicate_tasks"
    name: ClassVar[str] = "Semantically Duplicate Tasks"
    description: ClassVar[str] = "Check the model for any duplicate tasks based on semantic matching"
    algorithm_kind: ClassVar[AlgorithmKind] = algorithm_category
    threshold: ClassVar[float] = 0.75

    def analyze(self, inputs=None) -> AlgorithmResult:
        tasks = get_elements_by_type(self.model_xml, "task")

        problematic_elements = []

        def parse_duplicates(duplicates):
            for key, group in duplicates.items():
                # print(f"Duplicate for {key}: {group}")
                pair_one_element_id = group[0][1]
                if pair_one_element_id not in problematic_elements:
                    problematic_elements.append(pair_one_element_id)

                pair_two_element_id = group[1][1]
                if pair_two_element_id not in problematic_elements:
                    problematic_elements.append(pair_two_element_id)

        parse_duplicates(find_semantic_duplicates(tasks, threshold=self.threshold))

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_type,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


# Helpers

nlp = spacy.load("en_core_web_md")


def check_single_action(label):
    doc = nlp(label)
    verbs = [token for token in doc if token.pos_ == "VERB"]
    return len(verbs) <= 1


def atomicity_score(label):
    words = label.split()
    conjunction_words = ['and', 'or', 'then', 'after', 'also']

    penalties = 0
    penalties += len(words) * 0.1
    penalties += sum(1 for word in words if word.lower() in conjunction_words) * 2
    penalties /= 10  # Scale back to 0-1

    return max(0, 1 - penalties)


def find_semantic_duplicates(tuples_list, threshold):
    # Extract just task label
    labels = [t[0] for t in tuples_list]

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
            if score >= threshold and pair_key not in processed_pairs:
                used_indices.add(idx1)
                used_indices.add(idx2)
                processed_pairs.add(pair_key)
                duplicate_pairs.append((idx1, idx2, score))

    for idx1, idx2, _ in duplicate_pairs:
        original = tuples_list[idx1]
        group = [original, tuples_list[idx2]]
        groups[original[0]] = group

    return groups


def find_fuzzy_duplicates(tuples_list, threshold):
    groups = defaultdict(list)
    processed = set()
    threshold *= 100  # Fuzzywuzzy using values between 0-100

    for i, current_tuple in enumerate(tuples_list):
        if i in processed:
            continue

        current_key = current_tuple[0]
        group = [current_tuple]
        processed.add(i)

        # Compare with remaining tuples
        for j, other_tuple in enumerate(tuples_list[i + 1:], i + 1):
            if j in processed:
                continue

            other_key = other_tuple[0]
            similarity = fuzz.ratio(current_key, other_key)

            if similarity >= threshold:
                group.append(other_tuple)
                processed.add(j)

        if len(group) > 1:  # Only keep groups with duplicates
            groups[current_key] = group

    return dict(groups)
