import torch
from pydantic import BaseModel

from bpmn.bpmn import Bpmn
from utils.similarity import create_similarity_matrix


class TaskPairing(BaseModel):
    """Represents a matched pair of tasks between student and reference"""
    student_id: str
    student_label: str
    reference_id: str
    reference_label: str
    match_score: float


class SanityCheckResult(BaseModel):
    """Result of the sanity check comparison"""
    pairings: list[TaskPairing]
    missing: list[str]  # Reference task IDs that didn't match
    coverage: float  # Percentage of reference tasks that matched (0.0 to 1.0)
    total_reference_tasks: int
    total_student_tasks: int


def check_task_coverage(
    reference_xml: str,
    student_xml: str,
    threshold: float = 0.8
) -> SanityCheckResult:
    """
    Compare tasks between reference and student submissions globally.

    For each task in the reference model, find the best matching task in the
    student model. Tasks that don't match above the threshold are marked as
    missing/unmatched.

    Args:
        reference_xml: The reference BPMN XML
        student_xml: The student BPMN XML
        threshold: Minimum similarity score to consider a match (default: 0.8)

    Returns:
        SanityCheckResult with pairings, missing tasks, and coverage stats
    """
    # Parse both models
    reference_model = Bpmn(reference_xml)
    student_model = Bpmn(student_xml)

    # Extract all tasks from both models (excluding events, gateways, etc.)
    reference_tasks = []
    reference_task_labels = []
    for pool in reference_model.pools:
        for element in pool.elements:
            # Include only tasks and activities
            if element.label and (
                "task" in element.name.lower() or
                element.name in ["task", "subProcess"]
            ):
                reference_tasks.append(element)
                reference_task_labels.append(element.label)

    student_tasks = []
    student_task_labels = []
    for pool in student_model.pools:
        for element in pool.elements:
            if element.label and (
                "task" in element.name.lower() or
                element.name in ["task", "subProcess"]
            ):
                student_tasks.append(element)
                student_task_labels.append(element.label)

    print(f"[SanityCheck] Found {len(reference_tasks)} reference tasks")
    print(f"[SanityCheck] Found {len(student_tasks)} student tasks")

    # Handle edge cases
    if len(reference_tasks) == 0:
        return SanityCheckResult(
            pairings=[],
            missing=[],
            coverage=1.0,
            total_reference_tasks=0,
            total_student_tasks=len(student_tasks)
        )

    if len(student_tasks) == 0:
        # All reference tasks are unmatched
        return SanityCheckResult(
            pairings=[],
            missing=[task.id for task in reference_tasks],
            coverage=0.0,
            total_reference_tasks=len(reference_tasks),
            total_student_tasks=0
        )

    # Create similarity matrix: reference tasks x student tasks
    similarity_matrix = create_similarity_matrix(
        reference_task_labels,
        student_task_labels,
        self_similarity=False
    )

    # For each reference task, find the best matching student task
    best_matches = torch.max(similarity_matrix, dim=1)
    best_scores = best_matches.values
    best_indices = best_matches.indices

    # Track pairings and missing tasks
    pairings = []
    missing = []

    for i, (ref_task, best_score, best_idx) in enumerate(
        zip(reference_tasks, best_scores, best_indices)
    ):
        score = best_score.item()
        matched_student = student_tasks[best_idx.item()]

        print(
            f"[SanityCheck] Ref: '{ref_task.label}' -> "
            f"Student: '{matched_student.label}' (score: {score:.3f})"
        )

        if score >= threshold:
            # Good match found
            pairings.append(
                TaskPairing(
                    student_id=matched_student.id,
                    student_label=matched_student.label,
                    reference_id=ref_task.id,
                    reference_label=ref_task.label,
                    match_score=round(score, 3)
                )
            )
        else:
            # No good match found for this reference task
            print(f"[SanityCheck] MISSING: '{ref_task.label}' (best score: {score:.3f})")
            missing.append(ref_task.id)

    # Calculate coverage as percentage of matched tasks
    coverage = len(pairings) / len(reference_tasks) if len(reference_tasks) > 0 else 0.0

    print(f"[SanityCheck] Matched: {len(pairings)}/{len(reference_tasks)}")
    print(f"[SanityCheck] Coverage: {coverage:.3f}")

    return SanityCheckResult(
        pairings=pairings,
        missing=missing,
        coverage=round(coverage, 3),
        total_reference_tasks=len(reference_tasks),
        total_student_tasks=len(student_tasks)
    )
