import json
import os
import sys

import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

import algorithms.manager
from routers import submissions, rubric
from routers import algorithms as algorithms_router
from routers import templates, template_groups
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
app.include_router(submissions.router, tags=["submissions"])
app.include_router(rubric.router, tags=["rubric"])
app.include_router(algorithms_router.router, tags=["algorithms"])
app.include_router(templates.router, tags=["templates"])
app.include_router(template_groups.router, tags=["template-groups"])


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

    # Load algorithms during startup
    try:
        algorithms.manager.load_algorithms()
    except Exception as e:
        print(f"Could not load algorithms: {e}")
        sys.exit(1)

    # Initialize app state
    app.state.base_path = base_path
    app.state.rubric = get_rubric_from_disk(base_path)

    uvicorn.run(app, host="0.0.0.0", port=8000)
