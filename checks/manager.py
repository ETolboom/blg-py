import importlib
import inspect
import os
from pathlib import Path

from pydantic import ValidationError

from checks import Check, CheckFormInput

check_classes: list[type[Check]] = []


def load_checks() -> None:
    implementations_path = Path("checks/implementations")
    if not implementations_path.exists():
        raise FileNotFoundError(
            f"Could not find plugin directory: {implementations_path}"
        )

    check_names = []

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

                        # Skip classes that are not subclasses of Check or are abstract
                        if cls is Check or not issubclass(cls, Check):
                            continue
                        if inspect.isabstract(cls):
                            continue

                        # Will raise a validation error if the class is not valid
                        cls.model_validate(
                            {
                                "model_xml": "<xml/>",
                            }
                        )

                        check_classes.append(cls)
                        check_names.append(cls.id)
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

    print(f"Checks loaded successfully ({len(check_classes)}).")
    print(f"Found the following checks:\n{check_names}")


class CheckManager:
    checks: dict[str, Check] = {}

    def __init__(self, model_xml: str):
        self.model_xml: str = model_xml
        self.checks: dict[str, Check] = {}

        for check_class in check_classes:
            check = check_class(model_xml=model_xml)
            self.checks[check.id] = check

    def list_checks(
        self,
    ) -> list[dict[str, str | list[CheckFormInput]]]:
        checks = []
        for check in self.checks.values():
            entry: dict[str, str | list[CheckFormInput]] = {
                "id": check.id,
                "inputs": check.inputs(),
                "category": check.check_complexity,
                "name": check.name,
            }
            checks.append(entry)

        return checks

    def get_check(self, name: str) -> Check:
        return self.checks[name]


def get_manager(model_xml: str) -> CheckManager:
    return CheckManager(model_xml=model_xml)
