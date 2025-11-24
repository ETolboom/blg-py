from typing import List, ClassVar

import bpmn_analyzer

from algorithms import Algorithm, AlgorithmResult, AlgorithmFormInput, AlgorithmKind

algorithm_category = AlgorithmKind.STRUCTURAL

class Synchronization(Algorithm):
    id: ClassVar[str] = "synchronization"
    name: ClassVar[str] = "Synchronization"
    description: ClassVar[str] = "The process model properly synchronizes concurrent activities."
    algorithm_kind: ClassVar[AlgorithmKind] = algorithm_category

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_safeness(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            category=self.algorithm_type,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class DeadActivity(Algorithm):
    id: ClassVar[str] = "dead_activity"
    name: ClassVar[str] = "Dead Activities"
    description: ClassVar[str] = "All activities in the process model are reachable and can be executed"
    algorithm_kind: ClassVar[AlgorithmKind] = algorithm_category

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_dead_activities(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            category=self.algorithm_type,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class ProperCompletion(Algorithm):
    id: ClassVar[str] = "proper_completion"
    name: ClassVar[str] = "Unique End Event Execution"
    description: ClassVar[str] = "There is a single unambiguous way to reach the final end event."
    algorithm_kind: ClassVar[AlgorithmKind] = algorithm_category

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_proper_completion(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            category=self.algorithm_type,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class OptionToComplete(Algorithm):
    id: ClassVar[str] = "option_to_complete"
    name: ClassVar[str] = "No deadlocks"
    description: ClassVar[str] = "The process model can definitively reach its end state. E.g., no deadlocks"
    algorithm_kind: ClassVar[AlgorithmKind] = algorithm_category

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_option_to_complete(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_type,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True
