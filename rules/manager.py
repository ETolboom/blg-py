import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator, model_validator


def calculate_rule_max_points(nodes: list[dict]) -> float:
    """Calculate maximum points as sum of all node points in workflow"""
    total = sum(
        node.get('data', {}).get('points', 0)
        for node in nodes
    )
    return round(total, 2)  # Round to avoid floating-point errors


def calculate_group_max_points(
    rule_ids: list[str],
    manager: 'BehavioralRuleManager'
) -> float:
    """Calculate group max points as MAX of member rule maxPoints"""
    max_points_values = []
    for rule_id in rule_ids:
        rule = manager.get_rule(rule_id)
        if rule:
            max_points_values.append(rule.maxPoints)
        else:
            print(f"Warning: Rule '{rule_id}' not found")

    return max(max_points_values) if max_points_values else 0.0


class BehavioralRule(BaseModel):
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
        calculated = calculate_rule_max_points(nodes)

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


class RuleEvaluationSummary(BaseModel):
    """Summary of individual rule evaluation within a group"""
    rule_id: str
    rule_name: str
    description: Optional[str] = None  # Optional for backward compatibility
    earned_points: float
    confidence: float
    success: bool


class BehavioralRuleGroup(BaseModel):
    """Group of behavioral rules evaluated together as one rubric criterion"""
    group_id: str              # Unique identifier (e.g., "part_1_group")
    name: str                  # Display name in rubric
    description: str           # Criterion description
    maxPoints: Optional[float] = None  # Maximum points for the criterion (auto-calculated if not provided)
    condition: GroupCondition  # "XOR" or "AND"
    rule_ids: list[str]        # List of behavioral rule IDs (min 1)

    # Evaluation results (embedded after evaluation)
    last_evaluation: Optional[str] = None           # ISO 8601 timestamp of last evaluation
    earned_points: Optional[float] = None           # MAX points from rules
    best_rule_id: Optional[str] = None              # Rule with best points
    fulfilled: Optional[bool] = None                # Whether group requirements met
    confidence: Optional[float] = None              # Overall confidence score
    problematic_elements: Optional[list[str]] = None  # BPMN elements with issues
    rule_results: Optional[list[RuleEvaluationSummary]] = None  # Individual rule points

    @field_validator('rule_ids')
    @classmethod
    def validate_rule_ids(cls, v):
        if len(v) == 0:
            raise ValueError("Group must contain at least one template")
        return v


class BehavioralRuleManager:
    """Manages behavioral rules on disk"""

    def __init__(self, rules_dir: str = "example/rules"):
        self.rules_dir = Path(rules_dir)
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir = self.rules_dir.parent / "templates"

    def get_template(self, rule_id: str) -> Optional[BehavioralRule]:
        """Load a read-only template by ID from the templates directory"""
        safe_id = rule_id.replace("/", "_").replace("\\", "_")
        template_path = self.templates_dir / f"{safe_id}.json"

        if not template_path.exists():
            return None

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return BehavioralRule(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid template format: {e}")
        except Exception as e:
            raise IOError(f"Error loading template: {e}")

    def list_templates(self) -> list[dict]:
        """List all available templates with basic info"""
        templates = []

        if not self.templates_dir.exists():
            return templates

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

    def _get_rule_path(self, rule_id: str) -> Path:
        """Get the file path for a rule"""
        # Sanitize the template ID to prevent directory traversal
        safe_id = rule_id.replace("/", "_").replace("\\", "_")
        return self.rules_dir / f"{safe_id}.json"

    def list_rules(self) -> list[dict]:
        """List all available templates with basic info"""
        rules = []

        for file_path in self.rules_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    rules.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "description": data.get("description"),
                        "maxPoints": data.get("maxPoints"),
                    })
            except Exception as e:
                print(f"Error loading rule {file_path}: {e}")
                continue

        return rules

    def get_rule(self, rule_id: str) -> Optional[BehavioralRule]:
        """Get a specific behavioral rule by ID"""
        rule_path = self._get_rule_path(rule_id)

        if not rule_path.exists():
            return None

        try:
            with open(rule_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return BehavioralRule(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid rule format: {e}")
        except Exception as e:
            raise IOError(f"Error loading rule: {e}")

    def save_rule(self, rule: BehavioralRule) -> BehavioralRule:
        """Save or update a behavioral rule"""
        rule_path = self._get_rule_path(rule.id)

        try:
            with open(rule_path, 'w', encoding='utf-8') as f:
                # Use model_dump() to convert to dict, then json.dump for pretty printing
                json.dump(rule.model_dump(), f, indent=2, ensure_ascii=False)
            return rule
        except Exception as e:
            raise IOError(f"Error saving rule: {e}")

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a behavioral rule by ID"""
        rule_path = self._get_rule_path(rule_id)

        if not rule_path.exists():
            return False

        try:
            rule_path.unlink()
            return True
        except Exception as e:
            raise IOError(f"Error deleting rule: {e}")

    def rule_exists(self, rule_id: str) -> bool:
        """Check if a behavioral rule exists"""
        return self._get_rule_path(rule_id).exists()

    def _get_group_path(self, group_id: str) -> Path:
        """Get the file path for a group (uses _group_ prefix to distinguish from templates)"""
        # Sanitize the group ID to prevent directory traversal
        safe_id = group_id.replace("/", "_").replace("\\", "_")
        return self.rules_dir / f"_group_{safe_id}.json"

    def list_groups(self) -> list[dict]:
        """List all available template groups with basic info"""
        groups = []

        for file_path in self.rules_dir.glob("_group_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    groups.append({
                        "group_id": data.get("group_id"),
                        "name": data.get("name"),
                        "description": data.get("description"),
                        "maxPoints": data.get("maxPoints"),
                        "condition": data.get("condition"),
                        "rule_ids": data.get("rule_ids", []),
                        # Include evaluation results if present
                        "last_evaluation": data.get("last_evaluation"),
                        "earned_points": data.get("earned_points"),
                        "best_rule_id": data.get("best_rule_id"),
                        "fulfilled": data.get("fulfilled"),
                        "confidence": data.get("confidence"),
                        "problematic_elements": data.get("problematic_elements"),
                    })
            except Exception as e:
                print(f"Error loading group {file_path}: {e}")
                continue

        return groups

    def get_group(self, group_id: str) -> Optional[BehavioralRuleGroup]:
        """Get a specific group by ID"""
        group_path = self._get_group_path(group_id)

        if not group_path.exists():
            return None

        try:
            with open(group_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return BehavioralRuleGroup(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid group format: {e}")
        except Exception as e:
            raise IOError(f"Error loading group: {e}")

    def save_group(self, group: BehavioralRuleGroup) -> BehavioralRuleGroup:
        """Save or update a group"""
        # Validate that all referenced rules exist
        self.validate_group_rules(group)

        # Calculate maxPoints from member rules
        calculated_max = calculate_group_max_points(group.rule_ids, self)

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

    def validate_group_rules(self, group: BehavioralRuleGroup) -> bool:
        """Ensure all referenced rules exist"""
        for rule_id in group.rule_ids:
            if not self.rule_exists(rule_id):
                raise ValueError(f"Rule '{rule_id}' not found in group '{group.group_id}'")
        return True

    def update_group_evaluation(self, group_id: str, evaluation_result) -> BehavioralRuleGroup:
        """
        Update a group with evaluation results and save to disk

        Args:
            group_id: The group to update
            evaluation_result: GroupEvaluationResult from evaluation

        Returns:
            Updated BehavioralRuleGroup with evaluation results embedded
        """
        # Load existing group
        group = self.get_group(group_id)
        if group is None:
            raise ValueError(f"Group '{group_id}' not found")

        # Update with evaluation results
        group.last_evaluation = datetime.now().isoformat() + "Z"
        group.earned_points = evaluation_result.earned_points
        group.best_rule_id = evaluation_result.best_rule_id
        group.fulfilled = evaluation_result.fulfilled
        group.confidence = evaluation_result.overall_confidence
        group.problematic_elements = evaluation_result.problematic_elements

        # Save individual template results
        group.rule_results = [
            RuleEvaluationSummary(
                rule_id=tr.rule_id,
                rule_name=tr.rule_name,
                description=tr.description,
                earned_points=tr.earned_points,
                confidence=tr.confidence,
                success=tr.success
            )
            for tr in evaluation_result.rule_results
        ]

        # Save to disk
        return self.save_group(group)
