import os

from fastapi import Request

from checks.manager import CheckRegistry
from rubric import Rubric
from rules.manager import BehavioralRuleManager
from services.submissions import SubmissionService


def get_check_registry(request: Request) -> CheckRegistry:
    """Return the application-level CheckRegistry."""
    return request.app.state.check_registry


def get_submission_service(request: Request) -> SubmissionService:
    """Return the application-level SubmissionService."""
    return request.app.state.submission_service


def get_rule_manager(request: Request) -> BehavioralRuleManager:
    """Return the application-level BehavioralRuleManager."""
    return request.app.state.rule_manager


def save_rubric(request: Request, rubric: Rubric) -> None:
    """Persist *rubric* to app state and disk, then invalidate cached submission results."""
    request.app.state.rubric = rubric
    request.app.state.submission_service.rubric = rubric
    with open(os.path.join(request.app.state.base_path, "rubric.json"), "w") as f:
        f.write(rubric.to_disk_json())
    request.app.state.submission_service.invalidate_all_results()
