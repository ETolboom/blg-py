import os

from fastapi import APIRouter, HTTPException, Request

import rules.manager
from checks import CheckComplexity, CheckFormInput, CheckInputType
from checks.implementations.behavioral import BehavioralGroupEvaluator, GroupEvaluationResult
from rubric import RubricCriterion
from rules.manager import BehavioralRuleGroup

router = APIRouter()


@router.get("/behavioral-rule-groups")
async def list_rule_groups() -> list[dict]:
    """List all available template groups"""
    try:
        template_manager = rules.manager.get_manager()
        return template_manager.list_groups()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list groups: {str(e)}")


@router.get("/behavioral-rule-groups/{group_id}")
async def get_rule_group(group_id: str) -> BehavioralRuleGroup:
    """Get specific template group"""
    try:
        rule_manager = rules.manager.get_manager()
        group = rule_manager.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")
        return group
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get group: {str(e)}")


@router.post("/behavioral-rule-groups")
async def create_rule_group(group: BehavioralRuleGroup, request: Request) -> BehavioralRuleGroup:
    """Create new template group and auto-evaluate it if a reference model is loaded"""
    try:
        rule_manager = rules.manager.get_manager()

        # Check if group already exists
        if rule_manager.group_exists(group.group_id):
            raise HTTPException(
                status_code=409,
                detail=f"Group with ID '{group.group_id}' already exists. Use PUT to update."
            )

        # Validate that all templates exist
        rule_manager.validate_group_rules(group)
        saved_group = rule_manager.save_group(group)

        # Auto-evaluate if a reference model is available
        rubric = request.app.state.rubric
        if rubric and rubric.assignment and rubric.assignment.reference_xml:
            evaluator = BehavioralGroupEvaluator(model_xml=rubric.assignment.reference_xml)
            result = evaluator.evaluate_group(saved_group)
            saved_group = rule_manager.update_group_evaluation(saved_group.group_id, result)

        return saved_group
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create group: {str(e)}")


@router.put("/behavioral-rule-groups/{group_id}")
async def update_template_group(group_id: str, group: BehavioralRuleGroup) -> BehavioralRuleGroup:
    """Update existing template group"""
    try:
        # Ensure group_id matches
        if group.group_id != group_id:
            raise HTTPException(
                status_code=400,
                detail=f"Group ID in URL ('{group_id}') doesn't match ID in body ('{group.group_id}')"
            )

        rule_manager = rules.manager.get_manager()

        # Check if group exists
        if not rule_manager.group_exists(group_id):
            raise HTTPException(
                status_code=404,
                detail=f"Group '{group_id}' not found. Use POST to create."
            )

        # Validate that all templates exist
        rule_manager.validate_group_rules(group)
        return rule_manager.save_group(group)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update group: {str(e)}")


@router.delete("/behavioral-rule-groups/{group_id}")
async def delete_rule_group(group_id: str) -> dict:
    """Delete template group"""
    try:
        rule_manager = rules.manager.get_manager()
        success = rule_manager.delete_group(group_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Group '{group_id}' not found")
        return {"message": f"Group '{group_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete group: {str(e)}")


@router.post("/rubric/criteria/behavioral-group/analyze")
def analyze_behavioral_group(group: BehavioralRuleGroup, request: Request) -> GroupEvaluationResult:
    """
    Test evaluate a template group against reference model.
    Results are automatically saved to the group's JSON file.
    """
    rubric = request.app.state.rubric

    try:
        if not rubric or not rubric.assignment or not rubric.assignment.reference_xml:
            raise HTTPException(status_code=400, detail="No reference model loaded")

        # Evaluate the group
        evaluator = BehavioralGroupEvaluator(model_xml=rubric.assignment.reference_xml)
        result = evaluator.evaluate_group(group)

        # Save evaluation results to the group file (if it exists on disk)
        template_manager = rules.manager.get_manager()
        if template_manager.group_exists(group.group_id):
            template_manager.update_group_evaluation(group.group_id, result)

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Group evaluation failed: {str(e)}")


@router.post("/rubric/criteria/behavioral-group/{group_id}")
async def add_behavioral_group_to_rubric(group_id: str, group: BehavioralRuleGroup, request: Request):
    """Add template group as rubric criterion"""
    base_path = request.app.state.base_path
    rubric = request.app.state.rubric

    try:
        # Ensure group_id matches
        if group.group_id != group_id:
            raise HTTPException(
                status_code=400,
                detail=f"Group ID in URL ('{group_id}') doesn't match ID in body ('{group.group_id}')"
            )

        # Save group to disk first
        rule_manager = rules.manager.get_manager()
        rule_manager.validate_group_rules(group)
        rule_manager.save_group(group)

        # CONSUMPTION LOGIC: Remove individual templates from rubric
        for rule_id in group.rule_ids:
            index = next((i for i, c in enumerate(rubric.criteria)
                          if c.id == rule_id), -1)
            if index != -1:
                print(f"[Consumption] Removing template '{rule_id}' from rubric")
                del rubric.criteria[index]

        # Use "group:" prefix to distinguish from individual templates in rubric
        prefixed_group_id = f"group:{group.group_id}"

        # Remove duplicates (if group already in rubric)
        index = next((i for i, c in enumerate(rubric.criteria) if c.id == prefixed_group_id), -1)
        if index != -1:
            del rubric.criteria[index]

        # Add to rubric with prefixed ID (frontend can identify groups by "group:" prefix)
        rubric.criteria.append(
            RubricCriterion(
                id=prefixed_group_id,
                name=group.name,
                description=group.description,
                check_complexity=CheckComplexity.COMPLEX,
                inputs=[
                    CheckFormInput(
                        input_label="group_id",
                        input_type=CheckInputType.STRING,
                        data=group.group_id,
                    )
                ],
                fulfilled=True,
                confidence=1.0,
                problematic_elements=[],
                default_points=group.maxPoints or 0.0,
                custom_score=None,
            )
        )

        # Auto-evaluate the group and overwrite the placeholder criterion values
        if rubric.assignment and rubric.assignment.reference_xml:
            evaluator = BehavioralGroupEvaluator(model_xml=rubric.assignment.reference_xml)
            result = evaluator.evaluate_group(group)

            criterion_index = next(
                (i for i, c in enumerate(rubric.criteria) if c.id == prefixed_group_id), -1
            )
            if criterion_index != -1:
                rubric.criteria[criterion_index].fulfilled = result.fulfilled
                rubric.criteria[criterion_index].confidence = result.overall_confidence
                rubric.criteria[criterion_index].problematic_elements = result.problematic_elements
                earned = result.earned_points
                if round(earned, 2) != round(group.maxPoints or 0.0, 2):
                    rubric.criteria[criterion_index].custom_score = earned

            # Persist evaluation results into the group file
            rule_manager.update_group_evaluation(group.group_id, result)

        # Update app state
        request.app.state.rubric = rubric

        # Persist rubric
        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(rubric.model_dump_json())

        return rubric
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add group to rubric: {str(e)}")
