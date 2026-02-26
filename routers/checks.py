import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from checks import Check, CheckComplexity, CheckFormInput
from checks.implementations.behavioral import BehavioralRuleCheck, WorkflowData, BehavioralGroupEvaluator
from checks.manager import CheckRegistry
from dependencies import get_check_registry, get_rule_manager
from rubric import Rubric, RubricCriterion
from rules.manager import BehavioralRuleManager

router = APIRouter()


class NodeData(BaseModel):
    id: str
    name: str
    description: str


class Node(BaseModel):
    key: str
    data: NodeData
    children: list["Node"] | None = None


@router.get("/checks")
async def list_checks(registry: CheckRegistry = Depends(get_check_registry)) -> list[dict[str, str | list[CheckFormInput]]]:
    return registry.list_checks()


@router.post("/checks/analyze", response_model=None)
async def analyze_submission(filename: str, request: Request, registry: CheckRegistry = Depends(get_check_registry), rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> Response | Rubric:
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

    manager = registry.create_manager(model_xml)

    parsed_algorithms: list[RubricCriterion] = []
    for algorithm in rubric.criteria:
        # Check if this is a behavioral (template-based) criterion
        if algorithm.check_complexity == CheckComplexity.COMPLEX:
            # This is a behavioral criterion - detect if it's a GROUP or INDIVIDUAL TEMPLATE
            criterion_id = algorithm.id

            # Check if this is a group (prefixed with "group:")
            if criterion_id.startswith("group:"):
                # === GROUP EVALUATION ===
                # Strip the "group:" prefix to get the actual group_id
                group_id = criterion_id[6:]  # Remove "group:" prefix
                group = rule_manager.get_group(group_id)

                if group is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Group '{group_id}' not found on disk but referenced in rubric"
                    )

                evaluator = BehavioralGroupEvaluator(model_xml=model_xml, rule_manager=rule_manager)
                result = evaluator.evaluate_group(group)

                # Save evaluation results to group file
                rule_manager.update_group_evaluation(group_id, result)

                parsed_algorithms.append(
                    RubricCriterion(
                        id=criterion_id,  # Keep the "group:" prefix in the result
                        name=group.name,
                        description=group.description,
                        check_complexity=CheckComplexity.COMPLEX,
                        fulfilled=result.fulfilled,
                        inputs=algorithm.inputs,
                        confidence=result.overall_confidence,
                        problematic_elements=result.problematic_elements,
                        default_points=group.maxPoints,
                        custom_score=result.earned_points if round(result.earned_points, 2) != group.maxPoints else None,
                    )
                )
            else:
                # === INDIVIDUAL RULE EVALUATION (existing logic) ===
                rule = rule_manager.get_rule(criterion_id)

                if rule is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Rule or group '{criterion_id}' not found on disk but referenced in rubric"
                    )

                # Run behavioral analysis
                workflow_data = WorkflowData(nodes=rule.nodes, edges=rule.edges)
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
                        name=rule.name,
                        description=rule.description,
                        check_complexity=CheckComplexity.COMPLEX,
                        fulfilled=result.earned_points > 0,
                        inputs=algorithm.inputs,  # Keep template_id reference
                        confidence=result.confidence,
                        problematic_elements=problematic_elements,
                        default_points=rule.maxPoints,
                        custom_score=result.earned_points if round(result.earned_points, 2) != rule.maxPoints else None,
                    )
                )
        else:
            # Standard check - use check manager
            result = manager.get_check(algorithm.id).analyze(inputs=algorithm.inputs)
            parsed_algorithms.append(
                RubricCriterion(
                    id=result.id,
                    name=result.name,
                    description=result.description,
                    check_complexity=result.check_complexity,
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


@router.post("/checks/analyze/all")
async def analyze_all(req: Request, registry: CheckRegistry = Depends(get_check_registry)) -> list[Node]:
    model_xml = await req.body()
    if not model_xml:
        raise HTTPException(status_code=400, detail="request body is missing")

    manager = registry.create_manager(model_xml.decode())
    available_checks = manager.list_checks()

    applicable_checks: dict[str, list[Check]] = {}
    for entry in available_checks:
        check_id = str(entry["id"])
        check = manager.get_check(check_id)
        if check.is_applicable():
            # We order checks by category
            if check.check_complexity in applicable_checks:
                applicable_checks[check.check_complexity].append(check)
            else:
                applicable_checks[check.check_complexity] = [check]

    nodes: list[Node] = []

    node_idx = 0
    for category in applicable_checks:
        inner_nodes = []
        for inner_node_idx, check in enumerate(applicable_checks[category]):
            inner_nodes.append(
                Node(
                    key=str(node_idx) + "-" + str(inner_node_idx),
                    data=NodeData(
                        id=check.id,
                        name=check.name,
                        description=check.description,
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
