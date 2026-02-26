from fastapi import Request

from checks.manager import CheckRegistry
from rules.manager import BehavioralRuleManager
from services.submissions import SubmissionService


def get_check_registry(request: Request) -> CheckRegistry:
    return request.app.state.check_registry


def get_submission_service(request: Request) -> SubmissionService:
    return request.app.state.submission_service


def get_rule_manager(request: Request) -> BehavioralRuleManager:
    return request.app.state.rule_manager
