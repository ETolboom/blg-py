import importlib
import inspect
import os
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel
from typing import List, Any, Type


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


class Algorithm(ABC):
    """Every algorithm must implement this class."""
    id: str
    name: str
    description: str
    algorithm_type: str

    def __init__(self, model_xml: str):
        self.model_xml = model_xml

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
