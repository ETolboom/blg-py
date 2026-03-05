import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile

logger = logging.getLogger(__name__)

from checks import CheckComplexity, CheckFormInput, CheckInputType, CheckResult
from checks.implementations.behavioral import WorkflowData, BehavioralRuleCheck
from checks.manager import CheckRegistry
from dependencies import get_check_registry, get_rule_manager, save_rubric
from rubric import OnboardingRubric, Rubric, RubricCriterion
from rules.manager import BehavioralRule, BehavioralRuleManager

router = APIRouter()


@router.get("/rubric")
async def get_current_rubric(request: Request) -> Rubric:
    """Return the currently loaded rubric."""
    rubric = request.app.state.rubric
    if rubric is None:
        raise HTTPException(status_code=404, detail="Rubric not found")
    return rubric


@router.post("/rubric")
async def handle_onboarding_rubric(onboarding_rubric: OnboardingRubric, request: Request, registry: CheckRegistry = Depends(get_check_registry)) -> Rubric:
    """Create and persist a new rubric from an onboarding payload, running initial check analysis."""
    base_path = request.app.state.base_path

    ref_xml = (
        onboarding_rubric.assignment.reference_xml
        if onboarding_rubric.assignment and onboarding_rubric.assignment.reference_xml
        else ""
    )
    manager = registry.create_manager(ref_xml)

    parsed_algorithms = []
    if len(onboarding_rubric.checks) != 0:
        for algorithm in onboarding_rubric.checks:
            # Since we don't ask for inputs during onboarding
            # we assume that inputs are [] so the algorithm tries to
            # do a first pass / a best effort analysis.
            result = manager.get_check(algorithm).analyze(inputs=None)
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
                    score=None,
                )
            )

    new_rubric = Rubric(
        criteria=parsed_algorithms,
        assignment=onboarding_rubric.assignment,
    )

    # Write reference XML to separate file
    if ref_xml:
        with open(os.path.join(base_path, "reference.bpmn"), "w") as f:
            f.write(ref_xml)

    save_rubric(request, new_rubric)

    return new_rubric


@router.post("/rubric/criteria/behavioral/analyze")
def analyze_behavioral_criteria(data: WorkflowData, request: Request) -> CheckResult:
    """Run a behavioral rule check against the reference BPMN and return the result."""
    rubric = request.app.state.rubric

    try:
        alg = BehavioralRuleCheck(model_xml=rubric.assignment.reference_xml)
        return alg.check_behavior(workflow=data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rubric/criteria/behavioral/{behavioral_id}")
async def add_behavioral_criteria(behavioral_id: str, inputs: BehavioralRule, request: Request, rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> Rubric:
    """Save a behavioral rule and add (or replace) it as a criterion in the rubric."""
    rubric = request.app.state.rubric

    try:

        # If no nodes/edges provided, try loading existing rule or seeding from template
        if len(inputs.nodes) == 0 and len(inputs.edges) == 0:
            template = rule_manager.get_template(inputs.id)
            if template is not None:
                inputs = template

        # Prevent any duplicates by removing old instances of the algorithm.
        index = next(
            (
                i
                for i, criterion in enumerate(rubric.criteria)
                if criterion.id == behavioral_id
            ),
            -1,
        )
        if index != -1:
            del rubric.criteria[index]

        # Save rule to disk
        rule_manager.save_rule(inputs)

        # Store only a reference to the rule ID in the rubric
        # The actual rule data is loaded from disk when needed
        rubric.criteria.append(
            RubricCriterion(
                id=inputs.id,
                name=inputs.name,
                description=inputs.description,
                check_complexity=CheckComplexity.COMPLEX,
                inputs=[
                    CheckFormInput(
                        input_label="template_id",
                        input_type=CheckInputType.STRING,
                        data=inputs.id,  # Only store the template ID
                    ),
                ],
                fulfilled=True,
                confidence=1.0,
                problematic_elements=[],
                default_points=inputs.maxPoints,
                score=None,
            )
        )

        save_rubric(request, rubric)

        return rubric
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update criteria: {str(e)}"
        )


@router.post("/rubric/criteria/{algorithm_id}")
async def update_criteria(
    algorithm_id: str, inputs: list[CheckFormInput], request: Request, registry: CheckRegistry = Depends(get_check_registry),
) -> Rubric:
    """Run a standard check with the provided inputs and upsert the result as a rubric criterion."""
    rubric = request.app.state.rubric

    try:
        # Prevent any duplicates by removing old instances of the algorithm.
        index = next(
            (
                i
                for i, criterion in enumerate(rubric.criteria)
                if criterion.id == algorithm_id
            ),
            -1,
        )
        if index != -1:
            del rubric.criteria[index]

        if rubric and rubric.assignment and rubric.assignment.reference_xml:
            manager = registry.create_manager(rubric.assignment.reference_xml)
        else:
            manager = registry.create_manager("")
        result = manager.get_check(algorithm_id).analyze(inputs=inputs)
        rubric.criteria.append(
            RubricCriterion(
                id=algorithm_id,
                name=result.name,
                description=result.description,
                check_complexity=result.check_complexity,
                fulfilled=result.fulfilled,
                inputs=result.inputs,
                confidence=result.confidence,
                problematic_elements=result.problematic_elements,
                default_points=1.0,
                score=None,
            )
        )

        save_rubric(request, rubric)

        return rubric
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update criteria: {str(e)}"
        )


@router.post("/rubric/supplement")
async def upload_supplement(file: UploadFile, request: Request) -> dict:
    """Upload a supplement PDF for the rubric."""
    base_path = request.app.state.base_path

    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="File must be a PDF document"
        )

    # Validate content type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Expected application/pdf",
        )

    # Read and validate file size (10MB limit)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400, detail="PDF file size must not exceed 10MB"
        )

    # Save to disk
    supplement_path = os.path.join(base_path, "supplement.pdf")
    with open(supplement_path, "wb") as f:
        f.write(content)

    logger.info("Supplement PDF uploaded successfully")

    return {"message": "Supplement uploaded successfully", "filename": "supplement.pdf"}


@router.get("/rubric/supplement")
async def get_supplement(request: Request) -> Response:
    """Retrieve the supplement PDF if it exists."""
    base_path = request.app.state.base_path
    supplement_path = os.path.join(base_path, "supplement.pdf")

    if not os.path.exists(supplement_path):
        raise HTTPException(status_code=404, detail="No supplement PDF found")

    with open(supplement_path, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=supplement.pdf"},
    )


@router.delete("/rubric/supplement")
async def delete_supplement(request: Request) -> dict:
    """Delete the supplement PDF."""
    base_path = request.app.state.base_path
    supplement_path = os.path.join(base_path, "supplement.pdf")

    if not os.path.exists(supplement_path):
        raise HTTPException(status_code=404, detail="No supplement PDF found")

    os.remove(supplement_path)
    logger.info("Supplement PDF deleted")

    return {"message": "Supplement deleted successfully"}


@router.delete("/rubric/criteria/{criterion_id}")
async def delete_rubric_criterion(criterion_id: str, request: Request, rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> dict:
    """Delete a rubric criterion; for group criteria, restores the individual templates first."""
    rubric = request.app.state.rubric

    try:
        if rubric is None:
            raise HTTPException(status_code=404, detail="No rubric loaded")

        # Find criterion
        index = next((i for i, c in enumerate(rubric.criteria)
                      if c.id == criterion_id), -1)

        if index == -1:
            raise HTTPException(
                status_code=404,
                detail=f"Criterion '{criterion_id}' not found in rubric"
            )

        # Check if group (needs unmerge) or individual template (simple delete)
        if criterion_id.startswith("group:"):
            return await _unmerge_and_delete_group(criterion_id, index, rubric, request, rule_manager)
        else:
            # Simple deletion for individual templates
            del rubric.criteria[index]

            save_rubric(request, rubric)

            return {
                "message": f"Criterion '{criterion_id}' deleted successfully",
                "unmerged_templates": []
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete criterion: {str(e)}"
        )


async def _unmerge_and_delete_group(criterion_id: str, index: int, rubric: Rubric, request: Request, rule_manager: BehavioralRuleManager) -> dict:
    """Remove a group criterion and restore its constituent template criteria in the rubric."""
    # Extract group_id (remove "group:" prefix)
    group_id = criterion_id[6:]

    # Load group from disk
    group = rule_manager.get_group(group_id)

    if group is None:
        # Group file not found - cleanup orphaned reference
        del rubric.criteria[index]

        save_rubric(request, rubric)

        return {
            "message": f"Group criterion '{criterion_id}' deleted (group file not found)",
            "unmerged_templates": [],
            "warning": "Group metadata not found - could not restore templates"
        }

    # Restore templates at group's position
    restored = []
    missing = []
    insert_position = index  # Insert where the group was

    for template_id in group.rule_ids:
        template = rule_manager.get_rule(template_id)

        if template is None:
            missing.append(template_id)
            logger.warning("[Unmerge] Template '%s' not found on disk", template_id)
            continue

        # Check if already in rubric (avoid duplicates)
        exists = any(c.id == template_id for c in rubric.criteria)
        if exists:
            logger.info("[Unmerge] Template '%s' already in rubric, skipping", template_id)
            continue

        # Insert template at group's position
        rubric.criteria.insert(
            insert_position,
            RubricCriterion(
                id=template.id,
                name=template.name,
                description=template.description,
                check_complexity=CheckComplexity.COMPLEX,
                inputs=[
                    CheckFormInput(
                        input_label="template_id",
                        input_type=CheckInputType.STRING,
                        data=template.id,
                    ),
                ],
                fulfilled=True,
                confidence=1.0,
                problematic_elements=[],
                default_points=template.maxPoints,
                score=None,
            )
        )

        restored.append(template_id)
        insert_position += 1  # Next template inserts after this one
        logger.info("[Unmerge] Restored template '%s' at position %d", template_id, insert_position - 1)

    # Delete group criterion (now at insert_position due to insertions)
    del rubric.criteria[insert_position]

    save_rubric(request, rubric)

    result = {
        "message": f"Group '{criterion_id}' deleted and unmerged",
        "unmerged_templates": restored
    }

    if missing:
        result["warning"] = f"Some templates not found: {missing}"

    return result
