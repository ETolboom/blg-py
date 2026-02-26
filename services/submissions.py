import io
import os

from fastapi import HTTPException, UploadFile
from openpyxl import Workbook
from pydantic_core import from_json

from rubric import Rubric, RubricCriterion


class SubmissionService:
    def __init__(self, base_path: str, rubric: Rubric | None):
        self.base_path = base_path
        self.submissions_path = os.path.join(base_path, "submissions")
        self.rubric = rubric
        self.current_submission: str | None = None

    def list_submissions(self) -> list[dict]:
        os.makedirs(self.submissions_path, exist_ok=True)
        return [
            {"filename": f, "name": f.replace(".bpmn", "")}
            for f in os.listdir(self.submissions_path)
            if f.endswith(".bpmn")
        ]

    def get_submission_xml(self, filename: str) -> str:
        if filename == "Reference":
            if self.rubric and self.rubric.assignment and self.rubric.assignment.reference_xml:
                return self.rubric.assignment.reference_xml
            else:
                raise HTTPException(status_code=404, detail="Reference XML not found")

        path = os.path.join(self.submissions_path, filename)
        with open(path) as f:
            return f.read()

    def get_submission_rubric(self, filename: str) -> Rubric:
        path = os.path.join(self.submissions_path, filename + ".json")
        with open(path, encoding="utf-8") as f:
            submission_json = f.read()

        try:
            return Rubric.model_validate(from_json(submission_json, allow_partial=True))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def upload_submissions(self, files: list[UploadFile]) -> list[dict]:
        os.makedirs(self.submissions_path, exist_ok=True)

        uploaded = []
        for file in files:
            if not file.filename or not file.filename.endswith(".bpmn"):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{file.filename}' is not a .bpmn file",
                )

            dest = os.path.join(self.submissions_path, file.filename)
            content = await file.read()
            with open(dest, "wb") as f:
                f.write(content)

            uploaded.append({"filename": file.filename, "name": file.filename.replace(".bpmn", "")})

        return uploaded

    def update_submission_criteria(self, filename: str, criteria: list[RubricCriterion]) -> None:
        path = os.path.join(self.submissions_path, filename + ".json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Submission not found")

        with open(path, encoding="utf-8") as f:
            submission_json = f.read()

        try:
            parsed_rubric = Rubric.model_validate(from_json(submission_json, allow_partial=True))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        parsed_rubric.criteria = criteria

        with open(path, "w", encoding="utf-8") as f:
            f.write(parsed_rubric.model_dump_json())

    def export_submission(self, filename: str) -> bytes:
        parsed_rubric = self.get_submission_rubric(filename)
        return parsed_rubric.to_excel(filename)

    def export_all_submissions(self) -> bytes:
        json_files = [f for f in os.listdir(self.submissions_path) if f.endswith(".json")]

        excel_buffer = io.BytesIO()
        workbook = Workbook()

        for json_file in json_files:
            path = os.path.join(self.submissions_path, json_file)
            with open(path, encoding="utf-8") as f:
                submission_json = f.read()

            try:
                parsed_rubric = Rubric.model_validate(from_json(submission_json, allow_partial=True))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            parsed_rubric.to_excel_worksheet(workbook, json_file.replace(".json", ""))

        if "Sheet" in workbook.sheetnames:
            workbook.remove(workbook["Sheet"])

        workbook.save(excel_buffer)
        excel_buffer.seek(0)
        return excel_buffer.getvalue()

    def select_submission(self, filename: str | None) -> None:
        self.current_submission = filename
