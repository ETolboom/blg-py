import asyncio
import os

from fastapi import APIRouter, HTTPException, Request

import algorithms.manager
import templates.manager
from algorithms import AlgorithmComplexity, AlgorithmFormInput, AlgorithmInputType, AlgorithmResult
from algorithms.implementations.behavioral import WorkflowData, BehavioralRuleCheck
from rubric import OnboardingRubric, Rubric, RubricCriterion
from templates.manager import RuleTemplate

router = APIRouter()


@router.get("/rubric")
async def get_current_rubric(request: Request) -> Rubric:
    rubric = request.app.state.rubric
    if rubric is None:
        raise HTTPException(status_code=404, detail="Rubric not found")
    return rubric


@router.post("/rubric")
async def handle_onboarding_rubric(onboarding_rubric: OnboardingRubric, request: Request) -> Rubric:
    base_path = request.app.state.base_path

    ref_xml = (
        onboarding_rubric.assignment.reference_xml
        if onboarding_rubric.assignment and onboarding_rubric.assignment.reference_xml
        else ""
    )
    manager = algorithms.manager.get_manager(ref_xml)

    parsed_algorithms = []
    if len(onboarding_rubric.algorithms) != 0:
        for algorithm in onboarding_rubric.algorithms:
            # Since we don't ask for inputs during onboarding
            # we assume that inputs are [] so the algorithm tries to
            # do a first pass / a best effort analysis.
            result = manager.get_algorithm(algorithm).analyze()
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

    new_rubric = Rubric(
        criteria=parsed_algorithms,
        assignment=onboarding_rubric.assignment,
    )

    # Update app state
    request.app.state.rubric = new_rubric

    # Write new rubric to file so it persists
    with open(os.path.join(base_path, "rubric.json"), "w") as f:
        f.write(new_rubric.model_dump_json())

    return new_rubric


@router.post("/rubric/criteria/behavioral/analyze")
def analyze_behavioral_criteria(data: WorkflowData, request: Request) -> AlgorithmResult:
    rubric = request.app.state.rubric

    try:
        alg = BehavioralRuleCheck(model_xml=rubric.assignment.reference_xml)
        return alg.check_behavior(workflow=data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rubric/criteria/behavioral/{behavioral_id}")
async def add_behavioral_criteria(behavioral_id: str, inputs: RuleTemplate, request: Request) -> Rubric:
    base_path = request.app.state.base_path
    rubric = request.app.state.rubric

    try:
        # After model_post_init, nodes and edges are always lists
        # Check if they're empty
        nodes_empty = len(inputs.nodes) == 0
        edges_empty = len(inputs.edges) == 0

        # If nodes/edges are empty, load template from disk
        if nodes_empty or edges_empty:
            template_manager = templates.manager.get_manager()
            loaded_template = template_manager.get_template(inputs.id)

            if loaded_template is not None:
                # Use loaded template data
                inputs = loaded_template


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

        # Save template to disk first
        with open(os.path.join(base_path, "templates", inputs.id+".json"), "w") as f:
            f.write(inputs.model_dump_json())

        # Store only a reference to the template ID in the rubric
        # The actual template data is loaded from disk when needed
        rubric.criteria.append(
            RubricCriterion(
                id=inputs.id,
                name=inputs.name,
                description=inputs.description,
                category=AlgorithmComplexity.COMPLEX,
                inputs=[
                    AlgorithmFormInput(
                        input_label="template_id",
                        input_type=AlgorithmInputType.STRING,
                        data=inputs.id,  # Only store the template ID
                    ),
                ],
                fulfilled=True,
                confidence=1.0,
                problematic_elements=[],
                default_points=inputs.maxPoints,
                custom_score=None,
            )
        )

        # Update app state
        request.app.state.rubric = rubric

        # Write new rubric to file so it persists
        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(rubric.model_dump_json())

        return rubric
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update criteria: {str(e)}"
        )


@router.post("/rubric/criteria/{algorithm_id}")
async def update_criteria(
    algorithm_id: str, inputs: list[AlgorithmFormInput], request: Request
) -> Rubric:
    base_path = request.app.state.base_path
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
            manager = algorithms.manager.get_manager(rubric.assignment.reference_xml)
        else:
            manager = algorithms.manager.get_manager("")
        result = manager.get_algorithm(algorithm_id).analyze(inputs=inputs)
        rubric.criteria.append(
            RubricCriterion(
                id=algorithm_id,
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

        # Update app state
        request.app.state.rubric = rubric

        # Write new rubric to file so it persists
        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(rubric.model_dump_json())

        return rubric
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update criteria: {str(e)}"
        )


@router.post("/rubric/description")
async def update_rubric_description(req: Request) -> None:
    base_path = req.app.state.base_path
    rubric = req.app.state.rubric

    description = await req.body()
    if not description:
        raise HTTPException(status_code=400, detail="request body is missing")

    description_lock = asyncio.Lock()
    async with description_lock:
        description = description.decode("utf-8")

        if rubric and rubric.assignment:
            rubric.assignment.description = description

        # Update app state
        req.app.state.rubric = rubric

        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(rubric.model_dump_json())


@router.delete("/rubric/criteria/{criterion_id}")
async def delete_rubric_criterion(criterion_id: str, request: Request) -> dict:
    base_path = request.app.state.base_path
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
            return await _unmerge_and_delete_group(criterion_id, index, base_path, rubric, request)
        else:
            # Simple deletion for individual templates
            del rubric.criteria[index]

            # Update app state
            request.app.state.rubric = rubric

            with open(os.path.join(base_path, "rubric.json"), "w") as f:
                f.write(rubric.model_dump_json())

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


async def _unmerge_and_delete_group(criterion_id: str, index: int, base_path: str, rubric: Rubric, request: Request) -> dict:
    # Extract group_id (remove "group:" prefix)
    group_id = criterion_id[6:]

    # Load group from disk
    template_manager = templates.manager.get_manager()
    group = template_manager.get_group(group_id)

    if group is None:
        # Group file not found - cleanup orphaned reference
        del rubric.criteria[index]

        # Update app state
        request.app.state.rubric = rubric

        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(rubric.model_dump_json())

        return {
            "message": f"Group criterion '{criterion_id}' deleted (group file not found)",
            "unmerged_templates": [],
            "warning": "Group metadata not found - could not restore templates"
        }

    # Restore templates at group's position
    restored = []
    missing = []
    insert_position = index  # Insert where the group was

    for template_id in group.template_ids:
        template = template_manager.get_template(template_id)

        if template is None:
            missing.append(template_id)
            print(f"[Unmerge] Warning: Template '{template_id}' not found on disk")
            continue

        # Check if already in rubric (avoid duplicates)
        exists = any(c.id == template_id for c in rubric.criteria)
        if exists:
            print(f"[Unmerge] Template '{template_id}' already in rubric, skipping")
            continue

        # Insert template at group's position
        rubric.criteria.insert(
            insert_position,
            RubricCriterion(
                id=template.id,
                name=template.name,
                description=template.description,
                category=AlgorithmComplexity.COMPLEX,
                inputs=[
                    AlgorithmFormInput(
                        input_label="template_id",
                        input_type=AlgorithmInputType.STRING,
                        data=template.id,
                    ),
                ],
                fulfilled=True,
                confidence=1.0,
                problematic_elements=[],
                default_points=template.maxPoints,
                custom_score=None,
            )
        )

        restored.append(template_id)
        insert_position += 1  # Next template inserts after this one
        print(f"[Unmerge] Restored template '{template_id}' at position {insert_position-1}")

    # Delete group criterion (now at insert_position due to insertions)
    del rubric.criteria[insert_position]

    # Update app state
    request.app.state.rubric = rubric

    # Save rubric
    with open(os.path.join(base_path, "rubric.json"), "w") as f:
        f.write(rubric.model_dump_json())

    result = {
        "message": f"Group '{criterion_id}' deleted and unmerged",
        "unmerged_templates": restored
    }

    if missing:
        result["warning"] = f"Some templates not found: {missing}"

    return result
