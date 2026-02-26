from fastapi import Request

from checks.manager import CheckRegistry
from services.submissions import SubmissionService


def get_check_registry(request: Request) -> CheckRegistry:
    return request.app.state.check_registry


def get_submission_service(request: Request) -> SubmissionService:
    return request.app.state.submission_service
