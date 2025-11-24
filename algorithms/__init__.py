from abc import ABC, abstractmethod
from enum import Enum
from typing import List, ClassVar

from pydantic import BaseModel, ConfigDict

class AlgorithmFormInput(BaseModel):
    """This class describes the form elements required for the input for the algorithm."""

    # Label for the input
    input_label: str

    # Input type e.g. string, number, key-value
    input_type: str

    # Allow multiple inputs of this type
    multiple: bool

    # Only if input_type = key-value
    key_label: str
    value_label: str


class AlgorithmInput(BaseModel):
    """This class describes any elements required as input for the algorithm."""
    key: str = ""  # Only applicable to kv form type
    value: List[str]


class AlgorithmResult(BaseModel):
    """This class describes the format in which the algorithm is presented."""
    id: str = "algorithm_name"
    name: str = "Algorithm X"
    category: str = "Category"
    description: str = "This algorithm X checks for Y"
    fulfilled: bool | None = False
    confidence: float = 1.0
    problematic_elements: list[str] = []
    inputs: List[AlgorithmInput] = []


class AlgorithmKind(str, Enum):
    SEMANTIC    = "Semantic"
    STRUCTURAL  = "Structural"
    BEHAVIORAL = "Behavioural"


class Algorithm(BaseModel, ABC):
    """Every algorithm must implement this class."""
    model_config = ConfigDict(extra="forbid", strict=True)

    # These fields must be defined as class attributes with defaults in subclasses
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    algorithm_kind: ClassVar[AlgorithmKind]
    threshold: ClassVar[float] = 0.0

    # This field must be provided at instantiation
    model_xml: str

    @abstractmethod
    def analyze(self, inputs: List[AlgorithmInput] = None) -> AlgorithmResult:
        """Analyze a given property based on inputs if available"""
        pass

    @abstractmethod
    def inputs(self) -> List[AlgorithmFormInput]:
        """Return the available form inputs for the algorithm"""
        pass

    @abstractmethod
    def is_applicable(self) -> bool:
        """Check to see whether an algorithm is applicable to a given model"""
        pass
