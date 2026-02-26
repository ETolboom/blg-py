import json
import os
import sys

import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

import checks.manager
from routers import submissions, rubric
from routers import checks as checks_router
from routers import behavioral_rules, behavioral_rule_groups
from rubric import Rubric

app = FastAPI()


def get_rubric_from_disk(base_path: str) -> Rubric | None:
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


# Register routers
app.include_router(submissions.router, prefix="/api", tags=["submissions"])
app.include_router(rubric.router, prefix="/api", tags=["rubric"])
app.include_router(checks_router.router, prefix="/api", tags=["checks"])
app.include_router(behavioral_rules.router, prefix="/api", tags=["behavioral-rules"])
app.include_router(behavioral_rule_groups.router, prefix="/api", tags=["behavioral-rule-groups"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a folder path")
        print("Usage: python main.py <folder path>")
        sys.exit(1)

    base_path = sys.argv[1]

    if not os.path.isdir(base_path):
        print("Error: Please provide a valid folder path")
        print("Usage: python main.py <folder path>")
        sys.exit(1)

    # Load checks during startup
    try:
        checks.manager.load_checks()
    except Exception as e:
        print(f"Could not load checks: {e}")
        sys.exit(1)

    # Initialize app state
    app.state.base_path = base_path
    app.state.rubric = get_rubric_from_disk(base_path)

    uvicorn.run(app, host="127.0.0.1", port=8000)
