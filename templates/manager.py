import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator, model_validator


def calculate_template_max_points(nodes: list[dict]) -> float:
    """Calculate maximum points as sum of all node scores in workflow"""
    total = sum(
        node.get('data', {}).get('score', 0)
        for node in nodes
    )
    return round(total, 2)  # Round to avoid floating-point errors


def calculate_group_max_points(
    template_ids: list[str],
    manager: 'TemplateManager'
) -> float:
    """Calculate group max points as MAX of member template maxPoints"""
    max_points_values = []
    for template_id in template_ids:
        template = manager.get_template(template_id)
        if template:
            max_points_values.append(template.maxPoints)
        else:
            print(f"Warning: Template '{template_id}' not found")

    return max(max_points_values) if max_points_values else 0.0


class RuleTemplate(BaseModel):
    id: str
    name: str
    description: str
    maxPoints: Optional[float] = None
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

    @model_validator(mode='after')
    def validate_max_points(self):
        """Auto-calculate maxPoints from node scores"""
        # Ensure nodes is a list (not a string)
        nodes = self.nodes if isinstance(self.nodes, list) else []
        calculated = calculate_template_max_points(nodes)

        # Log warning if stored value differs
        if self.maxPoints is not None and abs(self.maxPoints - calculated) > 0.01:
            print(f"Warning: Template {self.id}: Stored maxPoints ({self.maxPoints}) "
                  f"differs from calculated ({calculated}). Using calculated value.")

        # Always use calculated value
        self.maxPoints = calculated
        return self


class GroupCondition(str, Enum):
    """Condition for evaluating template groups"""
    XOR = "XOR"  # At least one template must match (alternative solutions)
    AND = "AND"  # All templates must match (required features)


class TemplateGroup(BaseModel):
    """Group of templates evaluated together as one rubric criterion"""
    group_id: str              # Unique identifier (e.g., "part_1_group")
    name: str                  # Display name in rubric
    description: str           # Criterion description
    maxPoints: Optional[float] = None  # Maximum points for the criterion (auto-calculated if not provided)
    condition: GroupCondition  # "XOR" or "AND"
    template_ids: list[str]    # List of template IDs (min 1)

    # Evaluation results (embedded after evaluation)
    last_evaluation: Optional[str] = None           # ISO 8601 timestamp of last evaluation
    final_score: Optional[float] = None             # MAX score from templates
    best_template_id: Optional[str] = None          # Template with best score
    fulfilled: Optional[bool] = None                # Whether group requirements met
    confidence: Optional[float] = None              # Overall confidence score
    problematic_elements: Optional[list[str]] = None  # BPMN elements with issues

    @field_validator('template_ids')
    @classmethod
    def validate_template_ids(cls, v):
        if len(v) == 0:
            raise ValueError("Group must contain at least one template")
        return v


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

    def _get_group_path(self, group_id: str) -> Path:
        """Get the file path for a group (uses _group_ prefix to distinguish from templates)"""
        # Sanitize the group ID to prevent directory traversal
        safe_id = group_id.replace("/", "_").replace("\\", "_")
        return self.templates_dir / f"_group_{safe_id}.json"

    def list_groups(self) -> list[dict]:
        """List all available template groups with basic info"""
        groups = []

        for file_path in self.templates_dir.glob("_group_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    groups.append({
                        "group_id": data.get("group_id"),
                        "name": data.get("name"),
                        "description": data.get("description"),
                        "maxPoints": data.get("maxPoints"),
                        "condition": data.get("condition"),
                        "template_ids": data.get("template_ids", []),
                        # Include evaluation results if present
                        "last_evaluation": data.get("last_evaluation"),
                        "final_score": data.get("final_score"),
                        "best_template_id": data.get("best_template_id"),
                        "fulfilled": data.get("fulfilled"),
                        "confidence": data.get("confidence"),
                        "problematic_elements": data.get("problematic_elements"),
                    })
            except Exception as e:
                print(f"Error loading group {file_path}: {e}")
                continue

        return groups

    def get_group(self, group_id: str) -> Optional[TemplateGroup]:
        """Get a specific group by ID"""
        group_path = self._get_group_path(group_id)

        if not group_path.exists():
            return None

        try:
            with open(group_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return TemplateGroup(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid group format: {e}")
        except Exception as e:
            raise IOError(f"Error loading group: {e}")

    def save_group(self, group: TemplateGroup) -> TemplateGroup:
        """Save or update a group"""
        # Validate that all referenced templates exist
        self.validate_group_templates(group)

        # Calculate maxPoints from member templates
        calculated_max = calculate_group_max_points(group.template_ids, self)

        # Log warning if differs
        if group.maxPoints is not None and abs(group.maxPoints - calculated_max) > 0.01:
            print(f"Warning: Group {group.group_id}: Stored maxPoints ({group.maxPoints}) "
                  f"differs from calculated ({calculated_max}). Using calculated value.")

        group.maxPoints = calculated_max

        group_path = self._get_group_path(group.group_id)

        try:
            with open(group_path, 'w', encoding='utf-8') as f:
                # Use model_dump() to convert to dict, then json.dump for pretty printing
                json.dump(group.model_dump(), f, indent=2, ensure_ascii=False)
            return group
        except Exception as e:
            raise IOError(f"Error saving group: {e}")

    def delete_group(self, group_id: str) -> bool:
        """Delete a group by ID"""
        group_path = self._get_group_path(group_id)

        if not group_path.exists():
            return False

        try:
            group_path.unlink()
            return True
        except Exception as e:
            raise IOError(f"Error deleting group: {e}")

    def group_exists(self, group_id: str) -> bool:
        """Check if a group exists"""
        return self._get_group_path(group_id).exists()

    def validate_group_templates(self, group: TemplateGroup) -> bool:
        """Ensure all referenced templates exist"""
        for template_id in group.template_ids:
            if not self.template_exists(template_id):
                raise ValueError(f"Template '{template_id}' not found in group '{group.group_id}'")
        return True

    def update_group_evaluation(self, group_id: str, evaluation_result) -> TemplateGroup:
        """
        Update a group with evaluation results and save to disk

        Args:
            group_id: The group to update
            evaluation_result: GroupEvaluationResult from evaluation

        Returns:
            Updated TemplateGroup with evaluation results embedded
        """
        # Load existing group
        group = self.get_group(group_id)
        if group is None:
            raise ValueError(f"Group '{group_id}' not found")

        # Update with evaluation results
        group.last_evaluation = datetime.utcnow().isoformat() + "Z"
        group.final_score = evaluation_result.final_score
        group.best_template_id = evaluation_result.best_template_id
        group.fulfilled = evaluation_result.fulfilled
        group.confidence = evaluation_result.overall_confidence
        group.problematic_elements = evaluation_result.problematic_elements

        # Save to disk
        return self.save_group(group)


# Global manager instance
_manager: Optional[TemplateManager] = None


def get_manager(templates_dir: str = "example/templates") -> TemplateManager:
    """Get or create the global template manager"""
    global _manager
    if _manager is None:
        _manager = TemplateManager(templates_dir)
    return _manager
