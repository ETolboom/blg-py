from typing import Union, List, Optional

from pydantic import BaseModel

from algorithms import AlgorithmResult

class Assignment(BaseModel):
    # Set default values in case we want to onboard without
    # a reference model/description
    reference_xml: Optional[str] = ""
    description: Optional[str] = ""

class RubricCriterion(AlgorithmResult):
    custom_score: Union[float, None]
    default_points: float

class OnboardingRubric(BaseModel):
    assignment: Assignment
    algorithms: List[str] = []

class Rubric(BaseModel):
    criteria: List[RubricCriterion]
    assignment: Assignment