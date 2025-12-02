from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)


class AlgorithmInputType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    KEY_VALUE = "key-value"


TYPE_MAP: dict[AlgorithmInputType, type] = {
    AlgorithmInputType.STRING: str,
    AlgorithmInputType.INTEGER: int,
    AlgorithmInputType.KEY_VALUE: dict,
}


class AlgorithmFormInput(BaseModel):
    """This class describes the form elements required for the input for the algorithm."""

    # Label for the input
    input_label: str

    # Input type e.g. string, number, key-value
    input_type: AlgorithmInputType

    # Allow multiple inputs of this type
    multiple: bool = False

    data: str | int | dict

    @classmethod
    @field_validator("data")
    def _data_matches_declared_type(cls, v: Any, values: dict[str, Any]) -> Any:
        expected_type = TYPE_MAP[values["input_type"]]
        if not isinstance(v, expected_type):
            raise TypeError(f"Input data must be of type {expected_type.__name__}")

        match v:
            case str() if not v.strip():  # empty string
                raise ValueError("String input must not be empty")
            case int() if v is None:  # number is null
                raise ValueError("Integer input must not be null")
            case dict() if not v:  # dict is empty
                raise ValueError("Key-value input must contain at least one pair")

        return v


class AlgorithmResult(BaseModel):
    """This class describes the format in which the algorithm is presented."""

    id: str = "algorithm_name"
    name: str = "Algorithm X"
    category: str = "Category"
    description: str = "This algorithm X checks for Y"
    fulfilled: bool | None = False
    confidence: float = 1.0
    problematic_elements: list[str] = []
    inputs: list[AlgorithmFormInput] = []


class AlgorithmComplexity(str, Enum):
    SIMPLE = "Simple"
    CONFIGURABLE = "Configurable"
    COMPLEX = "Complex"


class Algorithm(BaseModel, ABC):
    """Every algorithm must implement this class."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # These fields must be defined as class attributes with defaults in subclasses
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    algorithm_kind: ClassVar[AlgorithmComplexity]
    threshold: ClassVar[float] = 0.0

    # This field must be provided at instantiation
    model_xml: str

    @abstractmethod
    def analyze(self, inputs: list[AlgorithmFormInput] | None) -> AlgorithmResult:
        """Analyze a given property based on inputs if available"""
        pass

    @abstractmethod
    def inputs(self) -> list[AlgorithmFormInput]:
        """Return the available form inputs for the algorithm"""
        pass

    @abstractmethod
    def is_applicable(self) -> bool:
        """Check to see whether an algorithm is applicable to a given model"""
        pass
