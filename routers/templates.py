import os

from fastapi import APIRouter, HTTPException, Request

import templates.manager
from algorithms.implementations.behavioral import WorkflowData, BehavioralRuleCheck, BehavioralGroupEvaluator
from templates.manager import RuleTemplate

router = APIRouter()


@router.get("/templates")
async def get_templates() -> list[dict]:
    """List all available rule templates"""
    try:
        template_manager = templates.manager.get_manager()
        return template_manager.list_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@router.get("/templates/{template_id}")
async def get_template(template_id: str) -> RuleTemplate:
    """Get a specific template by ID"""
    try:
        template_manager = templates.manager.get_manager()
        template = template_manager.get_template(template_id)

        if template is None:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

        return template
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@router.post("/templates")
async def create_template(template: RuleTemplate) -> RuleTemplate:
    """Create a new rule template"""
    try:
        template_manager = templates.manager.get_manager()

        # Check if template already exists
        if template_manager.template_exists(template.id):
            raise HTTPException(
                status_code=409,
                detail=f"Template with ID '{template.id}' already exists. Use PUT to update."
            )

        return template_manager.save_template(template)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@router.put("/templates/{template_id}")
async def update_template(template_id: str, template: RuleTemplate) -> RuleTemplate:
    """Update an existing rule template"""
    try:
        template_manager = templates.manager.get_manager()

        # Ensure the template ID in the URL matches the one in the body
        if template.id != template_id:
            raise HTTPException(
                status_code=400,
                detail=f"Template ID in URL ('{template_id}') doesn't match ID in body ('{template.id}')"
            )

        # Check if template exists
        if not template_manager.template_exists(template_id):
            raise HTTPException(
                status_code=404,
                detail=f"Template '{template_id}' not found. Use POST to create."
            )

        return template_manager.save_template(template)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str) -> dict:
    """Delete a rule template"""
    try:
        template_manager = templates.manager.get_manager()

        if not template_manager.delete_template(template_id):
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

        return {"message": f"Template '{template_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")


@router.post("/templates/{template_id}/validate")
async def validate_template(template_id: str, request: Request) -> dict:
    """
    Validate a rule template against the current rubric's reference BPMN.
    This runs the behavioral analysis and updates the rubric entry.

    If the template is part of any groups, those groups will also be
    automatically re-evaluated and their rubric entries updated.
    """
    base_path = request.app.state.base_path
    rubric = request.app.state.rubric

    try:
        # Get the template
        template_manager = templates.manager.get_manager()
        template = template_manager.get_template(template_id)

        if template is None:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

        # Ensure we have a rubric with reference XML
        if not rubric or not rubric.assignment or not rubric.assignment.reference_xml:
            raise HTTPException(
                status_code=400,
                detail="No reference BPMN model loaded. Please load a rubric first."
            )

        # Convert template to WorkflowData
        workflow_data = WorkflowData(
            nodes=template.nodes,
            edges=template.edges
        )

        # Run behavioral analysis
        checker = BehavioralRuleCheck(model_xml=rubric.assignment.reference_xml)
        result = checker.check_behavior(workflow=workflow_data)

        # Collect problematic BPMN element IDs
        problematic_elements = []
        for match in result.match_details:
            # Mark as problematic if:
            # - Below minimal threshold (is_correct=False, score < 0.6)
            # - Below ideal threshold (is_ideal_match=False, score < 0.8)
            # - Not at ideal distance
            if not match.is_correct or not match.is_ideal_match or not match.is_ideal_distance:
                if match.bpmn_element_id not in problematic_elements:
                    problematic_elements.append(match.bpmn_element_id)

        # Calculate score: use confidence as the score
        score = result.total_score

        # Update the rubric entry if it exists (individual template)
        criterion_index = next(
            (i for i, criterion in enumerate(rubric.criteria) if criterion.id == template_id),
            -1
        )

        if criterion_index != -1:
            # Update existing criterion
            rubric.criteria[criterion_index].fulfilled = score > 0
            rubric.criteria[criterion_index].confidence = result.confidence
            rubric.criteria[criterion_index].problematic_elements = problematic_elements

            if round(score, 2) != rubric.criteria[criterion_index].default_points:
                rubric.criteria[criterion_index].custom_score = score

        # === NEW: Re-evaluate any groups that contain this template ===
        affected_groups = []
        all_groups = template_manager.list_groups()

        for group_info in all_groups:
            if template_id in group_info.get('template_ids', []):
                # This group contains the updated template - re-evaluate it
                group = template_manager.get_group(group_info['group_id'])
                if group is not None:
                    # Re-evaluate the group
                    evaluator = BehavioralGroupEvaluator(model_xml=rubric.assignment.reference_xml)
                    group_result = evaluator.evaluate_group(group)

                    # Save evaluation results to group file
                    template_manager.update_group_evaluation(group.group_id, group_result)

                    # Find and update this group's rubric entry (search with "group:" prefix)
                    prefixed_group_id = f"group:{group.group_id}"
                    group_criterion_index = next(
                        (i for i, criterion in enumerate(rubric.criteria) if criterion.id == prefixed_group_id),
                        -1
                    )

                    if group_criterion_index != -1:
                        # Update the group's rubric entry
                        rubric.criteria[group_criterion_index].fulfilled = group_result.fulfilled
                        rubric.criteria[group_criterion_index].confidence = group_result.overall_confidence
                        rubric.criteria[group_criterion_index].problematic_elements = group_result.problematic_elements

                        if round(group_result.final_score, 2) != group.maxPoints:
                            rubric.criteria[group_criterion_index].custom_score = group_result.final_score
                        else:
                            rubric.criteria[group_criterion_index].custom_score = None

                        affected_groups.append({
                            "group_id": group.group_id,
                            "group_name": group.name,
                            "updated_score": group_result.final_score,
                            "best_template": group_result.best_template_id
                        })

        # Save updated rubric to disk (includes both template and group updates)
        if criterion_index != -1 or affected_groups:
            # Update app state
            request.app.state.rubric = rubric

            with open(os.path.join(base_path, "rubric.json"), "w") as f:
                f.write(rubric.model_dump_json())

        # Return validation results (including affected groups)
        return {
            "template_id": template_id,
            "template_name": template.name,
            "validation_result": {
                "fulfilled": result.fulfilled,
                "confidence": result.confidence,
                "total_matches": result.total_matches,
                "total_score": result.total_score,
                "match_details": [
                    {
                        "workflow_node_id": match.workflow_node_id,
                        "workflow_label": match.workflow_label,
                        "bpmn_element_id": match.bpmn_element_id,
                        "bpmn_label": match.bpmn_label,
                        "match_score": match.match_score,
                        "distance": match.distance,
                        "ideal_distance": match.ideal_distance,
                        "max_distance": match.max_distance,
                        "is_correct": match.is_correct,
                        "is_ideal_distance": match.is_ideal_distance,
                        "is_ideal_match": match.is_ideal_match,
                        "minimal_match_threshold": match.minimal_match_threshold,
                        "ideal_match_threshold": match.ideal_match_threshold,
                    }
                    for match in result.match_details
                ],
                "problematic_elements": result.problematic_elements
            },
            "affected_groups": affected_groups  # NEW: List of groups that were re-evaluated
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
