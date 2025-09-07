import asyncio
import io
import json
import os
import sys

from typing import List, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError
from pydantic_core import from_json

import algorithms.manager
from algorithms import Algorithm, AlgorithmFormInput, AlgorithmInput
from rubric import Rubric, OnboardingRubric, RubricCriterion

app = FastAPI()

origins = [
    "http://localhost:5173"
]

base_path = ""


def get_rubric_from_disk() -> Union[Rubric, None]:
    if os.path.exists(os.path.join(base_path, "rubric.json")):
        try:
            with open(os.path.join(base_path, "rubric.json"), "r") as file:
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
async def get_submissions_list() -> List[dict]:
    submissions_path = os.path.join(base_path, "submissions")
    os.makedirs(submissions_path, exist_ok=True)
    submissions = os.listdir(submissions_path)
    return [{"filename": f, "name": f.replace(".bpmn", "")} for f in submissions if f.endswith(".bpmn")]


@app.get("/submissions/export")
async def export_submission(filename: str) -> Response:
    submission = os.path.join(base_path, "submissions", filename + ".json")
    with open(submission, "r", encoding="utf-8") as f:
        submission_json = f.read()

    try:
        parsed_rubric: Rubric = Rubric.model_validate(from_json(submission_json, allow_partial=True))
    except Exception as parse_error:
        raise HTTPException(status_code=500, detail=str(parse_error))

    return Response(
        content=parsed_rubric.to_excel(filename),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename.replace('.bpmn', '.xlsx')}"}
    )


@app.get("/submissions/export/all")
async def export_all_submission() -> Response:
    submissions_path = os.path.join(base_path, "submissions")
    submissions = os.listdir(submissions_path)
    submissions = [file for file in submissions if file.endswith(".json")]

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        for submission in submissions:
            submission_path = os.path.join(base_path, "submissions", submission)
            with open(submission_path, "r", encoding="utf-8") as f:
                submission_json = f.read()

            try:
                parsed_rubric: Rubric = Rubric.model_validate(from_json(submission_json, allow_partial=True))
            except Exception as parse_error:
                raise HTTPException(status_code=500, detail=str(parse_error))

            parsed_rubric.to_excel_worksheet(writer, submission.replace(".json", ""))

        if 'Sheet' in writer.book.sheetnames:
            writer.book.remove(writer.book['Sheet'])

    excel_buffer.seek(0)

    return Response(
        content=excel_buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=submissions.xlsx"}
    )


@app.get("/submissions/{filename}")
async def get_submission(filename: str) -> Response:
    if filename == "Reference":
        return Response(content=rubric.assignment.reference_xml, media_type="application/xml")

    submissions_path = os.path.join(base_path, "submissions", filename)
    with open(submissions_path, "r") as model:
        return Response(content=model.read(), media_type="application/xml")


@app.get("/rubric")
async def get_current_rubric() -> Rubric:
    if rubric is None:
        raise HTTPException(status_code=404, detail="Rubric not found")
    return rubric


@app.post("/rubric")
async def handle_onboarding_rubric(onboarding_rubric: OnboardingRubric):
    manager = algorithms.manager.get_manager(onboarding_rubric.assignment.reference_xml)

    parsed_algorithms = []
    if len(onboarding_rubric.algorithms) != 0:
        for algorithm in onboarding_rubric.algorithms:
            # Since we don't ask for inputs during onboarding
            # we assume that inputs are [] so the algorithm tries to
            # do a first pass / a best effort analysis.
            result = manager.get_algorithm(algorithm).analyze()
            parsed_algorithms.append(RubricCriterion(
                id=result.id,
                name=result.name,
                description=result.description,
                fulfilled=result.fulfilled,
                inputs=result.inputs,
                confidence=result.confidence,
                problematic_elements=result.problematic_elements,
                default_points=1.0,
                custom_score=None,
            ))

    new_rubric = Rubric(
        criteria=parsed_algorithms,
        assignment=onboarding_rubric.assignment,
    )

    global rubric
    rubric = new_rubric

    # Write new rubric to file so it persists
    with open(os.path.join(base_path, "rubric.json"), "wt") as f:
        f.write(new_rubric.model_dump_json())

    return new_rubric


@app.post("/rubric/criteria/{algorithm_id}")
async def update_criteria(algorithm_id: str, inputs: List[AlgorithmInput]):
    global rubric

    try:
        # Prevent any duplicates by removing old instances of the algorithm.
        index = next((i for i, criterion in enumerate(rubric.criteria) if criterion.id == algorithm_id), -1)
        if index != -1:
            del rubric.criteria[index]

        manager = algorithms.manager.get_manager(rubric.assignment.reference_xml)
        result = manager.get_algorithm(algorithm_id).analyze(inputs=inputs)
        rubric.criteria.append(RubricCriterion(
            id=algorithm_id,
            name=result.name,
            description=result.description,
            fulfilled=result.fulfilled,
            inputs=result.inputs,
            confidence=result.confidence,
            problematic_elements=result.problematic_elements,
            default_points=1.0,
            custom_score=None,
        ))

        # Write new rubric to file so it persists
        with open(os.path.join(base_path, "rubric.json"), "wt") as f:
            f.write(rubric.model_dump_json())

        return rubric
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update criteria: {str(e)}")


@app.post("/rubric/description")
async def update_rubric_description(req: Request):
    description = await req.body()
    if not description:
        return "request body is missing"

    description_lock = asyncio.Lock()
    async with description_lock:
        description = description.decode("utf-8")

        global rubric
        rubric.assignment.description = description

        with open(os.path.join(base_path, "rubric.json"), "wt") as f:
            f.write(rubric.model_dump_json())


@app.get("/algorithms")
async def list_algorithms():
    manager = algorithms.manager.get_manager("")
    return manager.list_algorithms()


@app.patch("/submissions/{filename}")
async def update_submission(filename: str, criteria: List[RubricCriterion]):
    submission = os.path.join(base_path, "submissions", filename + ".json")
    if not os.path.exists(submission):
        raise HTTPException(status_code=404, detail="Submission not found")

    submission_lock = asyncio.Lock()

    async with submission_lock:
        with open(submission, "r", encoding="utf-8") as f:
            submission_json = f.read()

        try:
            parsed_rubric: Rubric = Rubric.model_validate(from_json(submission_json, allow_partial=True))
        except Exception as parse_error:
            raise HTTPException(status_code=500, detail=str(parse_error))

        parsed_rubric.criteria = criteria

        with open(submission, "wt", encoding="utf-8") as f:
            f.write(parsed_rubric.model_dump_json())


@app.post("/algorithms/analyze")
async def analyze_submission(filename: str):
    if filename == "":
        raise HTTPException(status_code=404, detail="No filename provided")

    submission = os.path.join(base_path, "submissions", filename)

    if os.path.exists(submission + ".json"):
        # We already have an analyzed result
        with open(submission + ".json", "r") as file:
            return Response(content=file.read(), media_type="application/json")

    if not os.path.exists(submission):
        raise HTTPException(status_code=404, detail="Submission not found")

    with open(submission, "r", encoding="utf-8") as f:
        model_xml = f.read()

    manager = algorithms.manager.get_manager(model_xml)

    parsed_algorithms: List[RubricCriterion] = []
    for algorithm in rubric.criteria:
        result = manager.get_algorithm(algorithm.id).analyze(inputs=algorithm.inputs)
        parsed_algorithms.append(RubricCriterion(
            id=result.id,
            name=result.name,
            description=result.description,
            fulfilled=result.fulfilled,
            inputs=result.inputs,
            confidence=result.confidence,
            problematic_elements=result.problematic_elements,
            default_points=1.0,
            custom_score=None,
        ))

    parsed_submission = Rubric(
        criteria=parsed_algorithms,
        assignment=None,
    )

    with open(submission + ".json", "wt") as f:
        f.write(parsed_submission.model_dump_json())

    return parsed_submission


@app.post("/algorithms/analyze/all")
async def analyze_all(req: Request):
    model_xml = await req.body()
    if not model_xml:
        return "request body is missing"

    manager = algorithms.manager.get_manager(model_xml.decode())
    available_algorithms = manager.list_algorithms()

    applicable_algorithms: dict[str, List[Algorithm]] = {}
    for entry in available_algorithms:
        algorithm = manager.get_algorithm(entry["id"])
        if algorithm.is_applicable():
            # We order algorithms by category
            if algorithm.algorithm_type in applicable_algorithms:
                applicable_algorithms[algorithm.algorithm_type].append(algorithm)
            else:
                applicable_algorithms[algorithm.algorithm_type] = [algorithm]

    class NodeData:
        id: str
        name: str
        description: str

        def __init__(self, alg_id, name, description):
            self.id = alg_id
            self.name = name
            self.description = description

    class Node:
        key: str
        data: NodeData
        children: Union[List, None] = None

        def __init__(self, key: str, data: NodeData, children: Union[List, None] = None):
            self.key = key
            self.data = data
            self.children = children

    nodes: List[Node] = []

    node_idx = 0
    for category in applicable_algorithms:
        inner_nodes = []
        for inner_node_idx, algorithm in enumerate(applicable_algorithms[category]):
            inner_nodes.append(Node(
                key=str(node_idx) + "-" + str(inner_node_idx),
                data=NodeData(
                    alg_id=algorithm.id,
                    name=algorithm.name,
                    description=algorithm.description,
                ),
            ))

        nodes.append(Node(
            key=str(node_idx),
            data=NodeData(
                alg_id="",
                name=category,
                description="",
            ),
            children=inner_nodes,
        ))

        node_idx += 1

    return nodes


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

    rubric = get_rubric_from_disk()

    uvicorn.run(app, host="0.0.0.0", port=8000)
