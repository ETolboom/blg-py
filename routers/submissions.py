import asyncio
import io
import os

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic_core import from_json

from rubric import Rubric, RubricCriterion

router = APIRouter()


@router.get("/submissions")
async def get_submissions_list(request: Request) -> list[dict]:
    base_path = request.app.state.base_path
    submissions_path = os.path.join(base_path, "submissions")
    os.makedirs(submissions_path, exist_ok=True)
    submissions = os.listdir(submissions_path)
    return [
        {"filename": f, "name": f.replace(".bpmn", "")}
        for f in submissions
        if f.endswith(".bpmn")
    ]


@router.get("/submissions/export")
async def export_submission(filename: str, request: Request) -> Response:
    base_path = request.app.state.base_path
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


@router.get("/submissions/export/all")
async def export_all_submission(request: Request) -> Response:
    base_path = request.app.state.base_path
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


@router.get("/submissions/{filename}")
async def get_submission(filename: str, request: Request) -> Response:
    base_path = request.app.state.base_path
    rubric = request.app.state.rubric

    if filename == "Reference":
        if rubric and rubric.assignment and rubric.assignment.reference_xml:
            return Response(
                content=rubric.assignment.reference_xml, media_type="application/xml"
            )
        else:
            raise HTTPException(status_code=404, detail="Reference XML not found")

    submissions_path = os.path.join(base_path, "submissions", filename)
    with open(submissions_path) as model:
        return Response(content=model.read(), media_type="application/xml")


@router.patch("/submissions/{filename}")
async def update_submission(filename: str, criteria: list[RubricCriterion], request: Request) -> None:
    base_path = request.app.state.base_path
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


@router.post("/submissions/{filename}/sanity-check")
async def sanity_check_submission(filename: str, request: Request):
    """
    Perform a sanity check on a submission by comparing its task labels
    with the reference model. This helps determine if behavioral checking
    is worth performing.

    Returns:
    - pairings: List of matched task pairs with their similarity scores
    - missing: List of reference task IDs that didn't match
    - coverage: Percentage of reference tasks that matched (0.0 to 1.0)
    - total_reference_tasks: Total number of tasks in reference model
    - total_student_tasks: Total number of tasks in student submission

    If coverage is low (e.g., < 0.5), behavioral checking may not be effective.
    """
    base_path = request.app.state.base_path
    rubric = request.app.state.rubric

    if not rubric or not rubric.assignment or not rubric.assignment.reference_xml:
        raise HTTPException(
            status_code=400,
            detail="No reference BPMN model loaded. Please load a rubric first."
        )

    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    submission_path = os.path.join(base_path, "submissions", filename)

    if not os.path.exists(submission_path):
        raise HTTPException(status_code=404, detail="Submission not found")

    # Read the student submission
    with open(submission_path, encoding="utf-8") as f:
        student_xml = f.read()

    # Import the sanity check utility
    from utils.sanity_check import check_task_coverage

    # Run the check
    result = check_task_coverage(
        reference_xml=rubric.assignment.reference_xml,
        student_xml=student_xml,
        threshold=0.8
    )

    return result
