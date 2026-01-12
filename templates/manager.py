import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator


class RuleTemplate(BaseModel):
    id: str
    name: str
    description: str
    maxPoints: float
    nodes: list[dict] | str  # Using dict to match the flexible Node structure, or string for serialized JSON
    edges: list[dict] | str  # Using dict to match the flexible Edge structure, or string for serialized JSON

    @field_validator('nodes', 'edges', mode='before')
    @classmethod
    def parse_json_strings(cls, v):
        """Convert JSON strings to lists"""
        if isinstance(v, str):
            try:
                return json.loads(v) if v else []
            except json.JSONDecodeError:
                return []
        return v if v is not None else []


class TemplateManager:
    """Manages rule templates on disk"""

    def __init__(self, templates_dir: str = "example/templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def _get_template_path(self, template_id: str) -> Path:
        """Get the file path for a template"""
        # Sanitize the template ID to prevent directory traversal
        safe_id = template_id.replace("/", "_").replace("\\", "_")
        return self.templates_dir / f"{safe_id}.json"

    def list_templates(self) -> list[dict]:
        """List all available templates with basic info"""
        templates = []

        for file_path in self.templates_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    templates.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "description": data.get("description"),
                        "maxPoints": data.get("maxPoints"),
                    })
            except Exception as e:
                print(f"Error loading template {file_path}: {e}")
                continue

        return templates

    def get_template(self, template_id: str) -> Optional[RuleTemplate]:
        """Get a specific template by ID"""
        template_path = self._get_template_path(template_id)

        if not template_path.exists():
            return None

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return RuleTemplate(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid template format: {e}")
        except Exception as e:
            raise IOError(f"Error loading template: {e}")

    def save_template(self, template: RuleTemplate) -> RuleTemplate:
        """Save or update a template"""
        template_path = self._get_template_path(template.id)

        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                # Use model_dump() to convert to dict, then json.dump for pretty printing
                json.dump(template.model_dump(), f, indent=2, ensure_ascii=False)
            return template
        except Exception as e:
            raise IOError(f"Error saving template: {e}")

    def delete_template(self, template_id: str) -> bool:
        """Delete a template by ID"""
        template_path = self._get_template_path(template_id)

        if not template_path.exists():
            return False

        try:
            template_path.unlink()
            return True
        except Exception as e:
            raise IOError(f"Error deleting template: {e}")

    def template_exists(self, template_id: str) -> bool:
        """Check if a template exists"""
        return self._get_template_path(template_id).exists()


# Global manager instance
_manager: Optional[TemplateManager] = None


def get_manager(templates_dir: str = "example/templates") -> TemplateManager:
    """Get or create the global template manager"""
    global _manager
    if _manager is None:
        _manager = TemplateManager(templates_dir)
    return _manager
