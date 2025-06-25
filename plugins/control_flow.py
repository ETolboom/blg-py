from typing import List

import bpmn_analyzer

from algorithms import Algorithm, AlgorithmResult, AlgorithmFormInput

algorithm_category = "Control Flow"


class Synchronization(Algorithm):
    id = "synchronization"
    name = "Synchronization"
    description = "The process model properly synchronizes concurrent activities."
    algorithm_type = algorithm_category

    def __init__(self, model_xml: str):
        super().__init__(model_xml)

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_safeness(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class DeadActivity(Algorithm):
    id = "dead_activity"
    name = "Dead Activities"
    description = "All activities in the process model are reachable and can be executed"
    algorithm_type = algorithm_category

    def __init__(self, model_xml: str):
        super().__init__(model_xml)

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_dead_activities(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class ProperCompletion(Algorithm):
    id = "proper_completion"
    name = "Unique End Event Execution"
    description = "There is a single unambiguous way to reach the final end event."
    algorithm_type = algorithm_category

    def __init__(self, model_xml: str):
        super().__init__(model_xml)

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_proper_completion(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True


class OptionToComplete(Algorithm):
    id = "option_to_complete"
    name = "No deadlocks"
    description = "The process model can definitively reach its end state. E.g., no deadlocks"
    algorithm_type = algorithm_category

    def __init__(self, model_xml: str):
        super().__init__(model_xml)

    def analyze(self, inputs=None) -> AlgorithmResult:
        result = bpmn_analyzer.analyze_option_to_complete(self.model_xml)
        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return []

    def is_applicable(self) -> bool:
        return True
