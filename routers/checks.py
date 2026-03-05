import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from checks import Check, CheckComplexity, CheckFormInput
from checks.implementations.behavioral import BehavioralRuleCheck, WorkflowData, BehavioralGroupEvaluator
from checks.manager import CheckRegistry
from dependencies import get_check_registry, get_rule_manager, get_submission_service
from rubric import Rubric, RubricCriterion, SubmissionCriterionResult, SubmissionResult
from rules.manager import BehavioralRuleManager
from services.submissions import SubmissionService

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
    """Return metadata for all registered checks."""
    return registry.list_checks()


@router.post("/checks/analyze", response_model=None)
async def analyze_submission(
    filename: str,
    request: Request,
    registry: CheckRegistry = Depends(get_check_registry),
    rule_manager: BehavioralRuleManager = Depends(get_rule_manager),
    submission_service: SubmissionService = Depends(get_submission_service),
) -> Rubric:
    """Analyze a student submission against the rubric and return the composed result."""
    rubric = request.app.state.rubric

    if filename == "":
        raise HTTPException(status_code=404, detail="No filename provided")

    if filename == "Reference":
        return rubric

    submission = os.path.join(request.app.state.base_path, "submissions", filename)

    if os.path.exists(submission + ".json"):
        # We already have an analyzed result — compose from reference rubric
        return submission_service.compose_rubric(filename)

    if not os.path.exists(submission):
        raise HTTPException(status_code=404, detail="Submission not found")

    with open(submission, encoding="utf-8") as f:
        model_xml = f.read()

    manager = registry.create_manager(model_xml)

    criterion_results: list[SubmissionCriterionResult] = []
    for check in rubric.criteria:
        # Check if this is a behavioral (template-based) criterion
        if check.check_complexity == CheckComplexity.COMPLEX:
            criterion_id = check.id

            if criterion_id.startswith("group:"):
                # === GROUP EVALUATION ===
                group_id = criterion_id[6:]
                group = rule_manager.get_group(group_id)

                if group is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Group '{group_id}' not found on disk but referenced in rubric"
                    )

                evaluator = BehavioralGroupEvaluator(model_xml=model_xml, rule_manager=rule_manager)
                result = evaluator.evaluate_group(group)

                criterion_results.append(
                    SubmissionCriterionResult(
                        id=criterion_id,
                        fulfilled=result.fulfilled,
                        confidence=result.overall_confidence,
                        problematic_elements=result.problematic_elements,
                        score=result.earned_points if round(result.earned_points, 2) != group.maxPoints else None,
                        inputs=check.inputs,
                    )
                )
            else:
                # === INDIVIDUAL RULE EVALUATION ===
                rule = rule_manager.get_rule(criterion_id)

                if rule is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Rule or group '{criterion_id}' not found on disk but referenced in rubric"
                    )

                workflow_data = WorkflowData(nodes=rule.nodes, edges=rule.edges)
                checker = BehavioralRuleCheck(model_xml=model_xml)
                try:
                    result = checker.check_behavior(workflow=workflow_data)
                except Exception:
                    criterion_results.append(
                        SubmissionCriterionResult(
                            id=criterion_id,
                            fulfilled=False,
                            confidence=0.0,
                            problematic_elements=[],
                            score=None,
                            inputs=check.inputs,
                        )
                    )
                    continue

                problematic_elements = []
                for match in result.match_details:
                    if not match.is_correct or not match.is_ideal_match or not match.is_ideal_distance:
                        if match.bpmn_element_id not in problematic_elements:
                            problematic_elements.append(match.bpmn_element_id)

                criterion_results.append(
                    SubmissionCriterionResult(
                        id=criterion_id,
                        fulfilled=result.earned_points > 0,
                        confidence=result.confidence,
                        problematic_elements=problematic_elements,
                        score=result.earned_points if round(result.earned_points, 2) != rule.maxPoints else None,
                        inputs=check.inputs,
                    )
                )
        else:
            # Standard check
            result = manager.get_check(check.id).analyze(inputs=check.inputs)
            criterion_results.append(
                SubmissionCriterionResult(
                    id=result.id,
                    fulfilled=result.fulfilled,
                    confidence=result.confidence,
                    problematic_elements=result.problematic_elements,
                    score=None,
                )
            )

    # Save lightweight delta to disk
    submission_result = SubmissionResult(criteria=criterion_results)
    with open(submission + ".json", "w") as f:
        f.write(submission_result.model_dump_json())

    # Build composed Rubric from reference metadata + fresh results
    result_map = {cr.id: cr for cr in criterion_results}
    composed: list[RubricCriterion] = []
    for ref in rubric.criteria:
        sr = result_map.get(ref.id)
        if sr is not None:
            composed.append(RubricCriterion(
                id=ref.id,
                name=ref.name,
                description=ref.description,
                check_complexity=ref.check_complexity,
                inputs=ref.inputs,
                default_points=ref.default_points,
                fulfilled=sr.fulfilled,
                score=sr.score,
                confidence=sr.confidence,
                problematic_elements=sr.problematic_elements,
            ))
        else:
            composed.append(RubricCriterion(
                id=ref.id,
                name=ref.name,
                description=ref.description,
                check_complexity=ref.check_complexity,
                inputs=ref.inputs,
                default_points=ref.default_points,
                fulfilled=None,
                score=None,
            ))

    return Rubric(criteria=composed, assignment=None)


@router.post("/checks/analyze/all")
async def analyze_all(req: Request, registry: CheckRegistry = Depends(get_check_registry)) -> list[Node]:
    """Return all applicable checks for a given BPMN model, grouped by complexity category."""
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
