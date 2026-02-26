from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator, ValidationInfo,
)


class CheckInputType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    KEY_VALUE = "key-value"
    SELECTION = "selection"

class CheckKeyValuePair(BaseModel):
    key: str
    value: list[str]

class CheckSelectionPair(BaseModel):
    label: str
    type: str

class CheckSelectionType(BaseModel):
    placeholder: str
    accepted_values: list[str]
    pairs: list[CheckSelectionPair]

class CheckKeyValueType(BaseModel):
    pairs: list[CheckKeyValuePair] = []
    key_label: str = Field(description="Label for outer key e.g. Pool-Name")
    value_label: str = Field(description="Label for inner list items e.g. Lane-Name")


TYPE_MAP: dict[CheckInputType, type] = {
    CheckInputType.STRING: str,
    CheckInputType.INTEGER: int,
    CheckInputType.KEY_VALUE: CheckKeyValuePair,
    CheckInputType.SELECTION: CheckSelectionType,
}


class CheckFormInput(BaseModel):
    """This class describes the form elements required for the input for the algorithm."""

    # Label for the input
    input_label: str

    # Input type e.g. string, number, key-value
    input_type: CheckInputType

    # Allow multiple inputs of this type
    multiple: bool = False

    data: str | int | CheckKeyValueType | CheckSelectionType

    @classmethod
    @field_validator("data")
    def _data_matches_declared_type(cls, v: Any, info: ValidationInfo) -> Any:
        expected_type = TYPE_MAP[info.data["input_type"]]
        if not isinstance(v, expected_type):
            raise TypeError(f"Input data must be of type {expected_type.__name__}")

        match v:
            case str() if not v.strip():  # empty string
                raise ValueError("String input must not be empty")
            case CheckKeyValueType() if not v.pairs:  # dict is empty
                raise ValueError("Key-value input must contain at least one pair")
            case CheckSelectionType() if not v.accepted_values:
                raise ValueError("Possible selection must contain at least one possible value")

        return v


class CheckComplexity(str, Enum):
    SIMPLE = "Quality Checks (Model-Agnostic)"
    CONFIGURABLE = "Simple (Model-Dependent)"
    COMPLEX = "Complex (Model-Dependent)"

class CheckResult(BaseModel):
    """This class describes the format in which the algorithm is presented."""

    id: str
    name: str
    check_complexity: CheckComplexity
    description: str
    fulfilled: bool
    confidence: float = 1.0
    problematic_elements: list[str] = []
    inputs: list[CheckFormInput] = []

class Check(BaseModel, ABC):
    """Every check must implement this class."""

    model_config = ConfigDict(extra="ignore", strict=True)

    # These fields must be defined as class attributes with defaults in subclasses
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    check_complexity: ClassVar[CheckComplexity]
    threshold: ClassVar[float] = 0.0
    input_scheme: ClassVar[list[CheckFormInput]]

    # This field must be provided at instantiation
    model_xml: str

    @classmethod
    def load_dependencies(cls) -> None:
        """
        Load any heavy dependencies (ML models, etc.) required by this check.

        This method is called once during startup before any checks are instantiated.
        Subclasses should override this to load models, tokenizers, etc.

        Example:
            @classmethod
            def load_dependencies(cls) -> None:
                global _my_model
                if _my_model is None:
                    print(f"Loading model for {cls.name}...")
                    _my_model = load_expensive_model()
        """
        pass  # Default: no dependencies

    @abstractmethod
    def analyze(self, inputs: list[CheckFormInput] | None) -> CheckResult:
        """Analyze a given property based on inputs if available"""
        pass

    @abstractmethod
    def is_applicable(self) -> bool:
        """Check to see whether a check is applicable to a given model"""
        pass
