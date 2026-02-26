import os

from fastapi import APIRouter, Depends, HTTPException, Request

from checks.implementations.behavioral import WorkflowData, BehavioralRuleCheck, BehavioralGroupEvaluator
from dependencies import get_rule_manager
from rules.manager import BehavioralRule, BehavioralRuleManager

router = APIRouter()


@router.get("/behavioral-rule-templates")
async def get_templates(rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> list[dict]:
    """List all available rule templates"""
    try:
        return rule_manager.list_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@router.get("/behavioral-rules")
async def get_rules(rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> list[dict]:
    """List all available rules"""
    try:
        return rule_manager.list_rules()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list rules: {str(e)}")


@router.get("/behavioral-rules/{rule_id}")
async def get_rule(rule_id: str, rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> BehavioralRule:
    """Get a specific rule by ID"""
    try:
        rule = rule_manager.get_rule(rule_id)

        if rule is None:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        return rule
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rule: {str(e)}")


@router.post("/behavioral-rules")
async def create_rule(rule: BehavioralRule, rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> BehavioralRule:
    """Create a new rule rule"""
    try:
        # Check if rule already exists
        if rule_manager.rule_exists(rule.id):
            raise HTTPException(
                status_code=409,
                detail=f"Rule with ID '{rule.id}' already exists. Use PUT to update."
            )

        return rule_manager.save_rule(rule)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {str(e)}")


@router.put("/behavioral-rules/{rule_id}")
async def update_rule(rule_id: str, rule: BehavioralRule, rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> BehavioralRule:
    """Update an existing rule rule"""
    try:
        # Ensure the rule ID in the URL matches the one in the body
        if rule.id != rule_id:
            raise HTTPException(
                status_code=400,
                detail=f"Rule ID in URL ('{rule_id}') doesn't match ID in body ('{rule.id}')"
            )

        # Check if rule exists
        if not rule_manager.rule_exists(rule_id):
            raise HTTPException(
                status_code=404,
                detail=f"Rule '{rule_id}' not found. Use POST to create."
            )

        return rule_manager.save_rule(rule)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update rule: {str(e)}")


@router.delete("/behavioral-rules/{rule_id}")
async def delete_rule(rule_id: str, rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> dict:
    """Delete a rule rule"""
    try:
        if not rule_manager.delete_rule(rule_id):
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        return {"message": f"Rule '{rule_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete rule: {str(e)}")


@router.post("/behavioral-rules/{rule_id}/validate")
async def validate_rule(rule_id: str, request: Request, rule_manager: BehavioralRuleManager = Depends(get_rule_manager)) -> dict:
    """
    Validate a behavioral rule against the current rubric's reference BPMN.
    This runs the behavioral analysis and updates the rubric entry.

    If the rule is part of any groups, those groups will also be
    automatically re-evaluated and their rubric entries updated.
    """
    base_path = request.app.state.base_path
    rubric = request.app.state.rubric

    try:
        # Get the rule
        rule = rule_manager.get_rule(rule_id)

        if rule is None:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")

        # Ensure we have a rubric with reference XML
        if not rubric or not rubric.assignment or not rubric.assignment.reference_xml:
            raise HTTPException(
                status_code=400,
                detail="No reference BPMN model loaded. Please load a rubric first."
            )

        # Convert rule to WorkflowData
        workflow_data = WorkflowData(
            nodes=rule.nodes,
            edges=rule.edges
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

        # Calculate earned points
        earned_points = result.earned_points

        # Update the rubric entry if it exists (individual rule)
        criterion_index = next(
            (i for i, criterion in enumerate(rubric.criteria) if criterion.id == rule_id),
            -1
        )

        if criterion_index != -1:
            # Update existing criterion
            rubric.criteria[criterion_index].default_points = rule.maxPoints
            rubric.criteria[criterion_index].fulfilled = earned_points > 0
            rubric.criteria[criterion_index].confidence = result.confidence
            rubric.criteria[criterion_index].problematic_elements = problematic_elements

            if round(earned_points, 2) != round(rule.maxPoints, 2):
                rubric.criteria[criterion_index].score = earned_points
            else:
                rubric.criteria[criterion_index].score = None

        affected_groups = []
        all_groups = rule_manager.list_groups()

        for group_info in all_groups:
            if rule_id in group_info.get('rule_ids', []):
                # This group contains the updated rule - re-evaluate it
                group = rule_manager.get_group(group_info['group_id'])
                if group is not None:
                    # Re-evaluate the group
                    evaluator = BehavioralGroupEvaluator(model_xml=rubric.assignment.reference_xml, rule_manager=rule_manager)
                    group_result = evaluator.evaluate_group(group)

                    # Save evaluation results to group file
                    rule_manager.update_group_evaluation(group.group_id, group_result)

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

                        if round(group_result.earned_points, 2) != group.maxPoints:
                            rubric.criteria[group_criterion_index].score = group_result.earned_points
                        else:
                            rubric.criteria[group_criterion_index].score = None

                        affected_groups.append({
                            "group_id": group.group_id,
                            "group_name": group.name,
                            "updated_points": group_result.earned_points,
                            "best_rule": group_result.best_rule_id
                        })

        # Save updated rubric to disk (includes both rule and group updates)
        if criterion_index != -1 or affected_groups:
            # Update app state
            request.app.state.rubric = rubric
            request.app.state.submission_service.rubric = rubric

            with open(os.path.join(base_path, "rubric.json"), "w") as f:
                f.write(rubric.model_dump_json())

            request.app.state.submission_service.invalidate_all_results()

        # Return validation results (including affected groups)
        return {
            "rule_id": rule_id,
            "rule_name": rule.name,
            "validation_result": {
                "fulfilled": result.fulfilled,
                "confidence": result.confidence,
                "total_matches": result.total_matches,
                "earned_points": result.earned_points,
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
