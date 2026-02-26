import glob
import io
import json
import os

from fastapi import HTTPException, UploadFile
from openpyxl import Workbook

from rubric import Rubric, RubricCriterion, SubmissionCriterionResult, SubmissionResult


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

    def _read_submission_result(self, filename: str) -> SubmissionResult | None:
        """Read a .bpmn.json file and return a SubmissionResult.

        Handles backward compatibility with old Rubric-format files by
        detecting the 'assignment' key and migrating on-the-fly.
        """
        path = os.path.join(self.submissions_path, filename + ".json")
        if not os.path.exists(path):
            return None

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Detect old Rubric format by presence of "assignment" key
        if "assignment" in data:
            # Migrate from old format: extract criterion results
            criteria = []
            for c in data.get("criteria", []):
                # Support both old "custom_score" and new "score" field names
                score = c.get("score", c.get("custom_score"))
                criteria.append(SubmissionCriterionResult(
                    id=c["id"],
                    score=score,
                    fulfilled=c.get("fulfilled"),
                    confidence=c.get("confidence", 0.0),
                    problematic_elements=c.get("problematic_elements", []),
                ))
            return SubmissionResult(criteria=criteria)

        return SubmissionResult.model_validate(data)

    def compose_rubric(self, filename: str) -> Rubric:
        """Compose a full Rubric by merging the reference rubric metadata
        with submission-specific analysis results."""
        result = self._read_submission_result(filename)
        if result is None:
            raise HTTPException(status_code=404, detail="Submission result not found")

        if self.rubric is None:
            raise HTTPException(status_code=500, detail="No reference rubric loaded")

        # Build a lookup from submission results by criterion id
        result_map = {cr.id: cr for cr in result.criteria}

        composed: list[RubricCriterion] = []
        for ref in self.rubric.criteria:
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
                # New criterion added to rubric after analysis — no results yet
                composed.append(RubricCriterion(
                    id=ref.id,
                    name=ref.name,
                    description=ref.description,
                    check_complexity=ref.check_complexity,
                    inputs=ref.inputs,
                    default_points=ref.default_points,
                    fulfilled=None,
                    score=None,
                    confidence=0.0,
                    problematic_elements=[],
                ))

        return Rubric(criteria=composed, assignment=None)

    def invalidate_all_results(self) -> None:
        """Delete all cached .bpmn.json result files."""
        for path in glob.glob(os.path.join(self.submissions_path, "*.bpmn.json")):
            os.remove(path)

    def get_submission_rubric(self, filename: str) -> Rubric:
        return self.compose_rubric(filename)

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

    def update_submission_criteria(self, filename: str, criteria: list[SubmissionCriterionResult]) -> None:
        path = os.path.join(self.submissions_path, filename + ".json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Submission not found")

        # Read existing result (handles backward compat)
        result = self._read_submission_result(filename)
        if result is None:
            result = SubmissionResult()

        # Build map from existing criteria, then overlay updates
        existing_map = {cr.id: cr for cr in result.criteria}
        for updated in criteria:
            existing_map[updated.id] = updated

        result.criteria = list(existing_map.values())

        with open(path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json())

    def export_submission(self, filename: str) -> bytes:
        parsed_rubric = self.compose_rubric(filename)
        return parsed_rubric.to_excel(filename)

    def export_all_submissions(self) -> bytes:
        json_files = [f for f in os.listdir(self.submissions_path) if f.endswith(".json")]

        excel_buffer = io.BytesIO()
        workbook = Workbook()

        for json_file in json_files:
            bpmn_filename = json_file.replace(".json", "")
            parsed_rubric = self.compose_rubric(bpmn_filename)
            parsed_rubric.to_excel_worksheet(workbook, bpmn_filename)

        if "Sheet" in workbook.sheetnames:
            workbook.remove(workbook["Sheet"])

        workbook.save(excel_buffer)
        excel_buffer.seek(0)
        return excel_buffer.getvalue()

    def select_submission(self, filename: str | None) -> None:
        self.current_submission = filename
