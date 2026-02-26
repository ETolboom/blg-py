from typing import ClassVar

import blg

from checks import (
    Check,
    CheckComplexity,
    CheckFormInput,
    CheckResult,
)


class Synchronization(Check):
    id: ClassVar[str] = "synchronization"
    name: ClassVar[str] = "Synchronization"
    description: ClassVar[str] = (
        "The process model properly synchronizes concurrent activities."
    )
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.SIMPLE
    input_scheme: ClassVar[list[CheckFormInput]] = []

    def analyze(self, inputs: list[CheckFormInput] | None = None) -> CheckResult:
        result = blg.analyze_safeness(self.model_xml)
        return CheckResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            check_complexity=self.check_complexity,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def is_applicable(self) -> bool:
        return True


class DeadActivity(Check):
    id: ClassVar[str] = "dead_activity"
    name: ClassVar[str] = "Dead Activities"
    description: ClassVar[str] = (
        "All activities in the process model are reachable and can be executed"
    )
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.SIMPLE
    input_scheme: ClassVar[list[CheckFormInput]] = []

    def analyze(self, inputs: list[CheckFormInput] | None = None) -> CheckResult:
        result = blg.analyze_dead_activities(self.model_xml)
        return CheckResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            check_complexity=self.check_complexity,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def is_applicable(self) -> bool:
        return True


class ProperCompletion(Check):
    id: ClassVar[str] = "proper_completion"
    name: ClassVar[str] = "Unique End Event Execution"
    description: ClassVar[str] = (
        "There is a single unambiguous way to reach the final end event."
    )
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.SIMPLE
    input_scheme: ClassVar[list[CheckFormInput]] = []

    def analyze(self, inputs: list[CheckFormInput] | None = None) -> CheckResult:
        result = blg.analyze_proper_completion(self.model_xml)
        return CheckResult(
            id=self.id,
            name=result.property_name,
            description=self.description,
            check_complexity=self.check_complexity,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def is_applicable(self) -> bool:
        return True


class OptionToComplete(Check):
    id: ClassVar[str] = "option_to_complete"
    name: ClassVar[str] = "No deadlocks"
    description: ClassVar[str] = (
        "The process model can definitively reach its end state. E.g., no deadlocks"
    )
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.SIMPLE
    input_scheme: ClassVar[list[CheckFormInput]] = []

    def analyze(self, inputs: list[CheckFormInput] | None = None) -> CheckResult:
        result = blg.analyze_option_to_complete(self.model_xml)
        return CheckResult(
            id=self.id,
            name=self.name,
            description=self.description,
            check_complexity=self.check_complexity,
            fulfilled=result.fulfilled,
            problematic_elements=result.problematic_elements,
        )

    def is_applicable(self) -> bool:
        return True
