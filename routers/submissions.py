from fastapi import APIRouter, Depends, Response, UploadFile

from dependencies import get_submission_service
from rubric import SubmissionCriterionResult
from services.submissions import SubmissionService

router = APIRouter()


@router.get("/submissions")
async def get_submissions_list(service: SubmissionService = Depends(get_submission_service)) -> list[dict]:
    return service.list_submissions()


@router.get("/submissions/export")
async def export_submission(filename: str, service: SubmissionService = Depends(get_submission_service)) -> Response:
    content = service.export_submission(filename)
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={filename.replace('.bpmn', '.xlsx')}"
        },
    )


@router.get("/submissions/export/all")
async def export_all_submission(service: SubmissionService = Depends(get_submission_service)) -> Response:
    content = service.export_all_submissions()
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=submissions.xlsx"},
    )


@router.post("/submissions")
async def upload_submissions(files: list[UploadFile], service: SubmissionService = Depends(get_submission_service)) -> list[dict]:
    """Upload one or more BPMN files to the submissions folder."""
    return await service.upload_submissions(files)


@router.get("/submissions/{filename}")
async def get_submission(filename: str, service: SubmissionService = Depends(get_submission_service)) -> Response:
    xml = service.get_submission_xml(filename)
    return Response(content=xml, media_type="application/xml")


@router.patch("/submissions/{filename}")
async def update_submission(filename: str, criteria: list[SubmissionCriterionResult], service: SubmissionService = Depends(get_submission_service)) -> None:
    service.update_submission_criteria(filename, criteria)
