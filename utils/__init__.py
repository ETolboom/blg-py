from xml.etree import ElementTree


def get_elements_by_type(model_xml: str, element_type: str) -> list[tuple[str, str]]:
    """
    Find elements in a model based on element type.

    :param model_xml: The raw XML of the BPMN model
    :param element_type: The type of element e.g. "task"
    :return: List containing tuples of (element_name, element_id)
    :raises ValueError: if the element type does not have a label attribute
    :raises TypeError: if the requested element type is not found in the model
    """

    root = ElementTree.fromstring(model_xml)
    namespace = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}
    elements = root.findall(f".//bpmn:{element_type}", namespace)

    if len(elements) == 0:
        return []
        # raise TypeError(f"Element {element_type} not found in model")

    labels = []
    for element in elements:
        if element.attrib.get("name") is not None:
            labels.append(
                (element.attrib["name"].lower().strip(), element.attrib["id"])
            )

    if len(labels) == 0:
        raise ValueError(f"Elements for type {element_type} have no labels")

    return labels


def extract_all_tasks(model_xml: str, allow_abstract=True) -> list[tuple[str, str]]:
    """
    Find all tasks in a model.

    :param model_xml: The raw XML of the BPMN model
    :param allow_abstract: Allow abstract tasks (tasks with no type specified)
    :return: List containing tuples of (element_name, element_id)
    :raises ValueError: if the element type does not have a label attribute
    :raises TypeError: if the requested element type is not found in the model
    """

    # Tasks with a specific type are different elements in BPMN
    # See Table 12.9 of the BPMN2.0.2 spec (P415)
    task_types = [
        "serviceTask",
        "sendTask",
        "receiveTask",
        "userTask",
        "manualTask",
        "businessRuleTask",
        "scriptTask",
    ]

    if allow_abstract:
        task_types.append("task")

    tasks: list[tuple[str, str]] = []

    for task_type in task_types:
        try:
            tasks.extend(get_elements_by_type(model_xml, task_type))
        except ValueError as e:
            print(f"Could not find tasks with type {task_type}: {e}")
            # Element type not found

    return tasks
