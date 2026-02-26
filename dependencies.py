from fastapi import Request

from checks.manager import CheckRegistry


def get_check_registry(request: Request) -> CheckRegistry:
    return request.app.state.check_registry
