from typing import ClassVar

import blg

from algorithms import (
    Algorithm,
    AlgorithmComplexity,
    AlgorithmFormInput,
    AlgorithmResult,
)


class Synchronization(Algorithm):
    id: ClassVar[str] = "synchronization"
    name: ClassVar[str] = "Synchronization"
    description: ClassVar[str] = (
        "The process model properly synchronizes concurrent activities."
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] =  AlgorithmComplexity.SIMPLE

    def analyze(self, inputs: list[AlgorithmFormInput] | None = None) -> AlgorithmResult:
        result = blg.analyze_safeness(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class DeadActivity(Algorithm):
    id: ClassVar[str] = "dead_activity"
    name: ClassVar[str] = "Dead Activities"
    description: ClassVar[str] = (
        "All activities in the process model are reachable and can be executed"
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.SIMPLE

    def analyze(self, inputs: list[AlgorithmFormInput] | None = None) -> AlgorithmResult:
        result = blg.analyze_dead_activities(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class ProperCompletion(Algorithm):
    id: ClassVar[str] = "proper_completion"
    name: ClassVar[str] = "Unique End Event Execution"
    description: ClassVar[str] = (
        "There is a single unambiguous way to reach the final end event."
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.SIMPLE

    def analyze(self, inputs: list[AlgorithmFormInput] | None = None) -> AlgorithmResult:
        result = blg.analyze_proper_completion(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class OptionToComplete(Algorithm):
    id: ClassVar[str] = "option_to_complete"
    name: ClassVar[str] = "No deadlocks"
    description: ClassVar[str] = (
        "The process model can definitively reach its end state. E.g., no deadlocks"
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.SIMPLE

    def analyze(self, inputs: list[AlgorithmFormInput] | None = None) -> AlgorithmResult:
        result = blg.analyze_option_to_complete(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True
