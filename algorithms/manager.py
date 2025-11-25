import importlib
import inspect
import os
from pathlib import Path

from pydantic import ValidationError

from algorithms import Algorithm, AlgorithmFormInput

algorithm_classes: list[type[Algorithm]] = []


def load_algorithms() -> None:
    implementations_path = Path("algorithms/implementations")
    if not implementations_path.exists():
        raise FileNotFoundError(
            f"Could not find plugin directory: {implementations_path}"
        )

    algorithm_names = []

    for root, dirs, files in os.walk(implementations_path):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                module_name = str(file_path.with_suffix("")).replace(os.sep, ".")

                try:
                    module = importlib.import_module(module_name)
                    for _, cls in inspect.getmembers(module, inspect.isclass):
                        # Ignore any classes that are not defined in the plugins directory
                        if cls.__module__ != module_name:
                            continue

                        # Skip classes that are not subclasses of Algorithm or are abstract
                        if cls is Algorithm or not issubclass(cls, Algorithm):
                            continue
                        if inspect.isabstract(cls):
                            continue

                        # Will raise a validation error if the class is not valid
                        cls.model_validate(
                            {
                                "model_xml": "<xml/>",
                            }
                        )

                        algorithm_classes.append(cls)
                        algorithm_names.append(cls.id)
                except TypeError as e:
                    raise Exception(
                        f"{module_name} failed to import due to a type error: {e}"
                    )
                except ValidationError as e:
                    raise Exception(
                        f"{module_name} failed to import due to a validation error: {e}"
                    )
                except ImportError as e:
                    raise Exception(f"{module_name} failed to import: {e}")
                except Exception as e:
                    raise Exception(f"could not load {module_name}: {e}")

    print(f"Algorithms loaded successfully ({len(algorithm_classes)}).")
    print(f"Found the following algorithms:\n{algorithm_names}")


class AlgorithmManager:
    algorithms: dict[str, Algorithm] = {}

    def __init__(self, model_xml: str):
        self.model_xml: str = model_xml
        self.algorithms: dict[str, Algorithm] = {}

        for algorithm_class in algorithm_classes:
            algorithm = algorithm_class(model_xml=model_xml)
            self.algorithms[algorithm.id] = algorithm

    def list_algorithms(
        self,
    ) -> list[dict[str, str | list[AlgorithmFormInput]]]:
        algorithms = []
        for algorithm in self.algorithms.values():
            entry: dict[str, str | list[AlgorithmFormInput]] = {
                "id": algorithm.id,
                "inputs": algorithm.inputs(),
                "category": algorithm.algorithm_kind,
                "name": algorithm.name,
            }
            algorithms.append(entry)

        return algorithms

    def get_algorithm(self, name: str) -> Algorithm:
        return self.algorithms[name]


def get_manager(model_xml: str) -> AlgorithmManager:
    return AlgorithmManager(model_xml=model_xml)
