import os

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

import algorithms.manager
import templates.manager
from algorithms import Algorithm, AlgorithmComplexity, AlgorithmFormInput
from algorithms.implementations.behavioral import BehavioralRuleCheck, WorkflowData, BehavioralGroupEvaluator
from rubric import Rubric, RubricCriterion

router = APIRouter()


class NodeData(BaseModel):
    id: str
    name: str
    description: str


class Node(BaseModel):
    key: str
    data: NodeData
    children: list["Node"] | None = None


@router.get("/algorithms")
async def list_algorithms() -> list[dict[str, str | list[AlgorithmFormInput]]]:
    manager = algorithms.manager.get_manager("")
    return manager.list_algorithms()


@router.post("/algorithms/analyze", response_model=None)
async def analyze_submission(filename: str, request: Request) -> Response | Rubric:
    base_path = request.app.state.base_path
    rubric = request.app.state.rubric

    if filename == "":
        raise HTTPException(status_code=404, detail="No filename provided")

    if filename == "Reference":
        return rubric

    submission = os.path.join(base_path, "submissions", filename)

    if os.path.exists(submission + ".json"):
        # We already have an analyzed result
        with open(submission + ".json") as file:
            return Response(content=file.read(), media_type="application/json")

    if not os.path.exists(submission):
        raise HTTPException(status_code=404, detail="Submission not found")

    with open(submission, encoding="utf-8") as f:
        model_xml = f.read()

    manager = algorithms.manager.get_manager(model_xml)

    parsed_algorithms: list[RubricCriterion] = []
    for algorithm in rubric.criteria:
        # Check if this is a behavioral (template-based) criterion
        if algorithm.category == AlgorithmComplexity.COMPLEX:
            # This is a behavioral criterion - detect if it's a GROUP or INDIVIDUAL TEMPLATE
            criterion_id = algorithm.id
            template_manager = templates.manager.get_manager()

            # Check if this is a group (prefixed with "group:")
            if criterion_id.startswith("group:"):
                # === GROUP EVALUATION ===
                # Strip the "group:" prefix to get the actual group_id
                group_id = criterion_id[6:]  # Remove "group:" prefix
                group = template_manager.get_group(group_id)

                if group is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Group '{group_id}' not found on disk but referenced in rubric"
                    )

                evaluator = BehavioralGroupEvaluator(model_xml=model_xml)
                result = evaluator.evaluate_group(group)

                # Save evaluation results to group file
                template_manager.update_group_evaluation(group_id, result)

                parsed_algorithms.append(
                    RubricCriterion(
                        id=criterion_id,  # Keep the "group:" prefix in the result
                        name=group.name,
                        description=group.description,
                        category=AlgorithmComplexity.COMPLEX,
                        fulfilled=result.fulfilled,
                        inputs=algorithm.inputs,
                        confidence=result.overall_confidence,
                        problematic_elements=result.problematic_elements,
                        default_points=group.maxPoints,
                        custom_score=result.final_score if round(result.final_score, 2) != group.maxPoints else None,
                    )
                )
            else:
                # === INDIVIDUAL TEMPLATE EVALUATION (existing logic) ===
                template = template_manager.get_template(criterion_id)

                if template is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Template or group '{criterion_id}' not found on disk but referenced in rubric"
                    )

                # Run behavioral analysis
                workflow_data = WorkflowData(nodes=template.nodes, edges=template.edges)
                checker = BehavioralRuleCheck(model_xml=model_xml)
                result = checker.check_behavior(workflow=workflow_data)

                # Collect problematic elements
                problematic_elements = []
                for match in result.match_details:
                    if not match.is_correct or not match.is_ideal_match or not match.is_ideal_distance:
                        if match.bpmn_element_id not in problematic_elements:
                            problematic_elements.append(match.bpmn_element_id)

                parsed_algorithms.append(
                    RubricCriterion(
                        id=criterion_id,
                        name=template.name,
                        description=template.description,
                        category=AlgorithmComplexity.COMPLEX,
                        fulfilled=result.total_score > 0,
                        inputs=algorithm.inputs,  # Keep template_id reference
                        confidence=result.confidence,
                        problematic_elements=problematic_elements,
                        default_points=template.maxPoints,
                        custom_score=result.total_score if round(result.total_score, 2) != template.maxPoints else None,
                    )
                )
        else:
            # Standard algorithm - use algorithm manager
            result = manager.get_algorithm(algorithm.id).analyze(inputs=algorithm.inputs)
            parsed_algorithms.append(
                RubricCriterion(
                    id=result.id,
                    name=result.name,
                    description=result.description,
                    category=result.category,
                    fulfilled=result.fulfilled,
                    inputs=result.inputs,
                    confidence=result.confidence,
                    problematic_elements=result.problematic_elements,
                    default_points=1.0,
                    custom_score=None,
                )
            )

    parsed_submission = Rubric(
        criteria=parsed_algorithms,
        assignment=None,
    )

    with open(submission + ".json", "w") as f:
        f.write(parsed_submission.model_dump_json())

    return parsed_submission


@router.post("/algorithms/analyze/all")
async def analyze_all(req: Request) -> list[Node]:
    model_xml = await req.body()
    if not model_xml:
        raise HTTPException(status_code=400, detail="request body is missing")

    manager = algorithms.manager.get_manager(model_xml.decode())
    available_algorithms = manager.list_algorithms()

    applicable_algorithms: dict[str, list[Algorithm]] = {}
    for entry in available_algorithms:
        alg_id = str(entry["id"])
        algorithm = manager.get_algorithm(alg_id)
        if algorithm.is_applicable():
            # We order algorithms by category
            if algorithm.algorithm_kind in applicable_algorithms:
                applicable_algorithms[algorithm.algorithm_kind].append(algorithm)
            else:
                applicable_algorithms[algorithm.algorithm_kind] = [algorithm]

    nodes: list[Node] = []

    node_idx = 0
    for category in applicable_algorithms:
        inner_nodes = []
        for inner_node_idx, algorithm in enumerate(applicable_algorithms[category]):
            inner_nodes.append(
                Node(
                    key=str(node_idx) + "-" + str(inner_node_idx),
                    data=NodeData(
                        id=algorithm.id,
                        name=algorithm.name,
                        description=algorithm.description,
                    ),
                )
            )

        nodes.append(
            Node(
                key=str(node_idx),
                data=NodeData(
                    id="",
                    name=category,
                    description="",
                ),
                children=inner_nodes,
            )
        )

        node_idx += 1

    return nodes
