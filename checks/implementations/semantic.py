from collections import defaultdict
from typing import ClassVar

import spacy
import torch
from thefuzz import fuzz

from checks import (
    Check,
    CheckComplexity,
    CheckFormInput,
    CheckResult,
)
from utils import extract_all_tasks, ExtractedTask
from utils.similarity import create_similarity_matrix


class AtomicityCheck(Check):
    id: ClassVar[str] = "atomicity_check"
    name: ClassVar[str] = "Label Atomicity"
    description: ClassVar[str] = "Check the task labels for atomicity"
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.SIMPLE
    threshold: ClassVar[float] = 0.85

    def analyze(
        self, inputs: list[CheckFormInput] | None = None
    ) -> CheckResult:
        tasks: list[ExtractedTask] = extract_all_tasks(self.model_xml)

        problematic_elements: list[str] = []
        for task in tasks:
            single_action = check_single_action(task.name)
            atomicity = atomicity_score(task.name)
            if not (single_action or atomicity >= self.threshold):
                problematic_elements.append(task.id)

        return CheckResult(
            id=self.id,
            name=self.name,
            description=self.description,
            check_complexity=self.check_complexity,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> list[CheckFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class ExactDuplicateTasks(Check):
    id: ClassVar[str] = "exact_duplicate_tasks"
    name: ClassVar[str] = "Exact Duplicate Tasks"
    description: ClassVar[str] = (
        "Check the model for any duplicate tasks based on fuzzy matching"
    )
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.SIMPLE
    threshold: ClassVar[float] = 0.90

    def analyze(
        self, inputs: list[CheckFormInput] | None = None
    ) -> CheckResult:
        tasks: list[ExtractedTask] = extract_all_tasks(self.model_xml)
        if len(tasks) == 0:
            raise Exception("Cannot identify exact duplicates: no tasks found")

        duplicates = find_fuzzy_duplicates(tasks, threshold=self.threshold)

        problematic_elements: list[str] = [element.id for pair in duplicates for element in pair]

        return CheckResult(
            id=self.id,
            name=self.name,
            description=self.description,
            check_complexity=self.check_complexity,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> list[CheckFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class SemanticDuplicateTasks(Check):
    id: ClassVar[str] = "semantic_duplicate_tasks"
    name: ClassVar[str] = "Semantically Duplicate Tasks"
    description: ClassVar[str] = (
        "Check the model for any duplicate tasks based on semantic matching"
    )
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.SIMPLE
    threshold: ClassVar[float] = 0.75

    def analyze(
        self, inputs: list[CheckFormInput] | None = None
    ) -> CheckResult:
        tasks: list[ExtractedTask] = extract_all_tasks(self.model_xml)
        if len(tasks) == 0:
            raise Exception("Cannot identify exact duplicates: no tasks found")

        duplicates: list[tuple[ExtractedTask, ExtractedTask]] = find_semantic_duplicates(tasks, threshold=self.threshold)

        problematic_elements: list[str] = [element.id for pair in duplicates for element in pair]

        return CheckResult(
            id=self.id,
            name=self.name,
            description=self.description,
            check_complexity=self.check_complexity,
            fulfilled=(len(problematic_elements) == 0),
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> list[CheckFormInput]:
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

    return max(0.0, 1.0 - penalties)


def find_semantic_duplicates(
    extracted_tasks: list[ExtractedTask], threshold: float
) -> list[tuple[ExtractedTask, ExtractedTask]]:
    labels: list[str] = [t.name for t in extracted_tasks]
    similarity_matrix = create_similarity_matrix(labels, labels, self_similarity=True)

    ranked_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

    processed: set[int] = set()
    pairs: list[tuple[ExtractedTask, ExtractedTask]] = []

    for i in range(len(labels)):
        if i in processed:
            continue

        for j in ranked_indices[i].tolist():
            if j in processed:
                continue
            score: float = similarity_matrix[i, j].item()
            if score < threshold:
                break  # remaining candidates are only worse
            processed.add(i)
            processed.add(j)
            pairs.append((extracted_tasks[i], extracted_tasks[j]))
            break

    return pairs


def find_fuzzy_duplicates(tasks: list[ExtractedTask], threshold: float) -> list[tuple[ExtractedTask, ExtractedTask]]:
    groups: list[tuple[ExtractedTask, ExtractedTask]] = []
    processed = set()
    threshold *= 100  # Thefuzz produces values between 0-100

    for i, current_task in enumerate(tasks):
        if i in processed:
            continue

        processed.add(i)

        for j, other_task in enumerate(tasks[i + 1 :], i + 1):
            if j in processed:
                continue

            # Similarity score between 0-100
            similarity: int = fuzz.ratio(current_task.name, other_task.name)
            if similarity >= threshold:
                groups.append((current_task, other_task))
                processed.add(j)

    return groups
