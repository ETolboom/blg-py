import asyncio
import io
import json
import os
import sys

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json

import algorithms.manager
import templates.manager
from algorithms import (
    Algorithm,
    AlgorithmComplexity,
    AlgorithmFormInput,
    AlgorithmInputType, AlgorithmResult,
)
from algorithms.implementations.behavioral import WorkflowData, BehavioralRuleCheck
from rubric import OnboardingRubric, Rubric, RubricCriterion
from templates.manager import RuleTemplate

app = FastAPI()

origins = ["http://localhost:5173"]

base_path = ""


def get_rubric_from_disk() -> Rubric | None:
    if os.path.exists(os.path.join(base_path, "rubric.json")):
        try:
            with open(os.path.join(base_path, "rubric.json")) as file:
                rubric_data = json.load(file)
            print("Rubric loaded successfully")
            return Rubric(**rubric_data)
        except json.JSONDecodeError:
            print("Error: rubric.json contains invalid JSON")
            return None
        except ValidationError as e:
            print(f"Error: JSON data doesn't match Rubric model: {e}")
            return None
        except Exception as e:
            print(f"Error loading rubric: {e}")
            return None
    else:
        return None


@app.get("/submissions")
async def get_submissions_list() -> list[dict]:
    submissions_path = os.path.join(base_path, "submissions")
    os.makedirs(submissions_path, exist_ok=True)
    submissions = os.listdir(submissions_path)
    return [
        {"filename": f, "name": f.replace(".bpmn", "")}
        for f in submissions
        if f.endswith(".bpmn")
    ]


@app.get("/submissions/export")
async def export_submission(filename: str) -> Response:
    submission = os.path.join(base_path, "submissions", filename + ".json")
    with open(submission, encoding="utf-8") as f:
        submission_json = f.read()

    try:
        parsed_rubric: Rubric = Rubric.model_validate(
            from_json(submission_json, allow_partial=True)
        )
    except Exception as parse_error:
        raise HTTPException(status_code=500, detail=str(parse_error))

    return Response(
        content=parsed_rubric.to_excel(filename),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={filename.replace('.bpmn', '.xlsx')}"
        },
    )


@app.get("/submissions/export/all")
async def export_all_submission() -> Response:
    submissions_path = os.path.join(base_path, "submissions")
    submissions = os.listdir(submissions_path)
    submissions = [file for file in submissions if file.endswith(".json")]

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        for submission in submissions:
            submission_path = os.path.join(base_path, "submissions", submission)
            with open(submission_path, encoding="utf-8") as f:
                submission_json = f.read()

            try:
                parsed_rubric: Rubric = Rubric.model_validate(
                    from_json(submission_json, allow_partial=True)
                )
            except Exception as parse_error:
                raise HTTPException(status_code=500, detail=str(parse_error))

            parsed_rubric.to_excel_worksheet(writer, submission.replace(".json", ""))

        if "Sheet" in writer.book.sheetnames:
            writer.book.remove(writer.book["Sheet"])

    excel_buffer.seek(0)

    return Response(
        content=excel_buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=submissions.xlsx"},
    )


@app.get("/submissions/{filename}")
async def get_submission(filename: str) -> Response:
    if filename == "Reference":
        if _rubric and _rubric.assignment and _rubric.assignment.reference_xml:
            return Response(
                content=_rubric.assignment.reference_xml, media_type="application/xml"
            )
        else:
            raise HTTPException(status_code=404, detail="Reference XML not found")

    submissions_path = os.path.join(base_path, "submissions", filename)
    with open(submissions_path) as model:
        return Response(content=model.read(), media_type="application/xml")


@app.get("/rubric")
async def get_current_rubric() -> Rubric:
    if _rubric is None:
        raise HTTPException(status_code=404, detail="Rubric not found")
    return _rubric


@app.post("/rubric")
async def handle_onboarding_rubric(onboarding_rubric: OnboardingRubric) -> Rubric:
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

    global _rubric
    _rubric = new_rubric

    # Write new rubric to file so it persists
    with open(os.path.join(base_path, "rubric.json"), "w") as f:
        f.write(new_rubric.model_dump_json())

    return new_rubric


@app.post("/rubric/criteria/behavioral/analyze")
def analyze_behavioral_criteria(data: WorkflowData) -> AlgorithmResult:
    global _rubric
    try:
        alg = BehavioralRuleCheck(model_xml=_rubric.assignment.reference_xml)
        return alg.check_behavior(workflow=data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rubric/criteria/behavioral/{behavioral_id}")
async def add_behavioral_criteria(behavioral_id: str, inputs: RuleTemplate) -> Rubric:
    global _rubric
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
                for i, criterion in enumerate(_rubric.criteria)
                if criterion.id == behavioral_id
            ),
            -1,
        )
        if index != -1:
            del _rubric.criteria[index]

        # Serialize nodes and edges to JSON strings for storage
        # (They are always lists after field validation)
        nodes_json = json.dumps(inputs.nodes)
        edges_json = json.dumps(inputs.edges)

        _rubric.criteria.append(
            RubricCriterion(
                id=inputs.id,
                name=inputs.name,
                description=inputs.description,
                category=AlgorithmComplexity.COMPLEX,
                inputs=[
                    AlgorithmFormInput(
                        input_label="nodes",
                        input_type=AlgorithmInputType.STRING,
                        data=nodes_json,
                    ),
                    AlgorithmFormInput(
                        input_label="edges",
                        input_type=AlgorithmInputType.STRING,
                        data=edges_json,
                    ),
                ],
                fulfilled=True,
                confidence=1.0,
                problematic_elements=[],
                default_points=inputs.maxPoints,
                custom_score=None,
            )
        )

        # Write new rubric to file so it persists
        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(_rubric.model_dump_json())

        with open(os.path.join(base_path, "templates", inputs.id+".json"), "w") as f:
            f.write(inputs.model_dump_json())

        return _rubric
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update criteria: {str(e)}"
        )



@app.post("/rubric/criteria/{algorithm_id}")
async def update_criteria(
    algorithm_id: str, inputs: list[AlgorithmFormInput]
) -> Rubric:
    global _rubric

    try:
        # Prevent any duplicates by removing old instances of the algorithm.
        index = next(
            (
                i
                for i, criterion in enumerate(_rubric.criteria)
                if criterion.id == algorithm_id
            ),
            -1,
        )
        if index != -1:
            del _rubric.criteria[index]

        if _rubric and _rubric.assignment and _rubric.assignment.reference_xml:
            manager = algorithms.manager.get_manager(_rubric.assignment.reference_xml)
        else:
            manager = algorithms.manager.get_manager("")
        result = manager.get_algorithm(algorithm_id).analyze(inputs=inputs)
        _rubric.criteria.append(
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

        # Write new rubric to file so it persists
        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(_rubric.model_dump_json())

        return _rubric
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update criteria: {str(e)}"
        )


@app.post("/rubric/description")
async def update_rubric_description(req: Request) -> None:
    description = await req.body()
    if not description:
        raise HTTPException(status_code=400, detail="request body is missing")

    description_lock = asyncio.Lock()
    async with description_lock:
        description = description.decode("utf-8")

        global _rubric
        if _rubric and _rubric.assignment:
            _rubric.assignment.description = description

        with open(os.path.join(base_path, "rubric.json"), "w") as f:
            f.write(_rubric.model_dump_json())


@app.get("/algorithms")
async def list_algorithms() -> list[dict[str, str | list[AlgorithmFormInput]]]:
    manager = algorithms.manager.get_manager("")
    return manager.list_algorithms()


@app.patch("/submissions/{filename}")
async def update_submission(filename: str, criteria: list[RubricCriterion]) -> None:
    submission = os.path.join(base_path, "submissions", filename + ".json")
    if not os.path.exists(submission):
        raise HTTPException(status_code=404, detail="Submission not found")

    submission_lock = asyncio.Lock()

    async with submission_lock:
        with open(submission, encoding="utf-8") as f:
            submission_json = f.read()

        try:
            parsed_rubric: Rubric = Rubric.model_validate(
                from_json(submission_json, allow_partial=True)
            )
        except Exception as parse_error:
            raise HTTPException(status_code=500, detail=str(parse_error))

        parsed_rubric.criteria = criteria

        with open(submission, "w", encoding="utf-8") as f:
            f.write(parsed_rubric.model_dump_json())


@app.post("/algorithms/analyze", response_model=None)
async def analyze_submission(filename: str) -> Response | Rubric:
    if filename == "":
        raise HTTPException(status_code=404, detail="No filename provided")

    if filename == "Reference":
        global _rubric
        return _rubric

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
    for algorithm in _rubric.criteria:
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


class NodeData(BaseModel):
    id: str
    name: str
    description: str


class Node(BaseModel):
    key: str
    data: NodeData
    children: list["Node"] | None = None


@app.post("/algorithms/analyze/all")
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


@app.get("/templates")
async def get_templates() -> list[dict]:
    """List all available rule templates"""
    try:
        template_manager = templates.manager.get_manager()
        return template_manager.list_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@app.get("/templates/{template_id}")
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


@app.post("/templates")
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


@app.put("/templates/{template_id}")
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


@app.delete("/templates/{template_id}")
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


@app.post("/templates/{template_id}/validate")
async def validate_template(template_id: str) -> dict:
    """
    Validate a rule template against the current rubric's reference BPMN.
    This runs the behavioral analysis and updates the rubric entry.
    """
    global _rubric

    try:
        # Get the template
        template_manager = templates.manager.get_manager()
        template = template_manager.get_template(template_id)

        if template is None:
            raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

        # Ensure we have a rubric with reference XML
        if not _rubric or not _rubric.assignment or not _rubric.assignment.reference_xml:
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
        checker = BehavioralRuleCheck(model_xml=_rubric.assignment.reference_xml)
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

        # Update the rubric entry if it exists
        criterion_index = next(
            (i for i, criterion in enumerate(_rubric.criteria) if criterion.id == template_id),
            -1
        )

        if criterion_index != -1:
            # Update existing criterion
            _rubric.criteria[criterion_index].fulfilled = score > 0
            _rubric.criteria[criterion_index].confidence = result.confidence
            _rubric.criteria[criterion_index].problematic_elements = problematic_elements

            if round(score, 2) != _rubric.criteria[criterion_index].default_points:
                _rubric.criteria[criterion_index].custom_score = score

            # Save updated rubric to disk
            with open(os.path.join(base_path, "rubric.json"), "w") as f:
                f.write(_rubric.model_dump_json())

        # Return validation results
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
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python main.py <folder path>")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Error: Please provide a folder path")
        print("Usage: python main.py <folder path>")
        sys.exit(1)

    base_path = sys.argv[1]

    if not os.path.isdir(base_path):
        print("Error: Please provide a valid folder path")
        print("Usage: python main.py <folder path>")
        sys.exit(1)

    # Gets called during startup
    try:
        algorithms.manager.load_algorithms()
    except Exception as e:
        print(f"Could not load algorithms: {e}")
        sys.exit(1)

    _rubric = get_rubric_from_disk()

    uvicorn.run(app, host="0.0.0.0", port=8000)
