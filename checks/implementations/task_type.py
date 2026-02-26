from typing import ClassVar

from checks import Check, CheckFormInput, CheckResult, CheckComplexity, CheckInputType, \
    CheckSelectionType, CheckSelectionPair
from utils import extract_all_tasks, TaskType, ExtractedTask
from utils.similarity import match_labels


class TaskTypeCheck(Check):
    id: ClassVar[str] = "task_type"
    name: ClassVar[str] = "Task Type"
    description: ClassVar[str] = "Check if all available tasks are of the correct type"
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.CONFIGURABLE
    threshold: ClassVar[float] = 0.7
    input_scheme: ClassVar[list[CheckFormInput]] = [
        CheckFormInput(
            input_label="Labels and Task Types",
            input_type=CheckInputType.SELECTION,
            data=CheckSelectionType(
                placeholder="Task Type",
                accepted_values=acceptable_task_types,
                pairs=[]
            ),
            multiple=True,
        ),
    ]

    acceptable_task_types: list[str] = [
        "serviceTask",
        "sendTask",
        "receiveTask",
        "userTask",
        "manualTask",
        "businessRuleTask",
        "scriptTask",
    ]

    def is_applicable(self) -> bool:
        tasks = extract_all_tasks(self.model_xml, allow_abstract=True)
        overall_types = set([element.task_type for element in tasks])
        if (len(overall_types) == 0) or (len(overall_types) == 1 and "task" in overall_types):
            # No tasks found or only abstract tasks in the model
            return False

        return True

    def analyze(self, inputs: list[CheckFormInput] | None = None) -> CheckResult:
        if inputs is None:
            # No inputs yet, extract it from the model
            tasks = extract_all_tasks(self.model_xml, allow_abstract=True)

            fulfilled = True

            abstract_tasks = [element.task_type for element in tasks if element.task_type is TaskType.ABSTRACT]

            if len(abstract_tasks) > 1:
                fulfilled = False

            return CheckResult(
                id=self.id,
                name=self.name,
                description=self.description,
                check_complexity=self.check_complexity,
                problematic_elements=[],
                fulfilled=fulfilled,
                inputs=[
                    CheckFormInput(
                        input_label="Labels and Task Types",
                        input_type=CheckInputType.SELECTION,
                        data=CheckSelectionType(
                            placeholder="Task Type",
                            accepted_values=self.acceptable_task_types,
                            pairs=[CheckSelectionPair(label=element.name, type=element.task_type) for element in tasks]
                        ),
                        multiple=True,
                    ),
                ],
            )

        target_tasks: list[ExtractedTask] = extract_all_tasks(self.model_xml, allow_abstract=True)

        reference_labels = [pair.label for pair in inputs[0].data.pairs]

        matches = match_labels(
            target=[element.name for element in target_tasks],
            reference=reference_labels,
            match_threshold=self.threshold,
        )

        problematic_elements: list[str] = []

        for (target_idx, ref_idx) in matches:
            target_task: ExtractedTask = target_tasks[target_idx]
            reference_task: CheckSelectionPair = inputs[0].data.pairs[ref_idx]

            if (target_task.task_type == TaskType.ABSTRACT) or (target_task.task_type != reference_task.type):
                problematic_elements.append(target_task.id)

        return CheckResult(
            id=self.id,
            name=self.name,
            check_complexity=self.check_complexity,
            description=self.description,
            fulfilled=len(problematic_elements) == 0,
            confidence=1.0,
            problematic_elements=problematic_elements,
            inputs=inputs,
        )
