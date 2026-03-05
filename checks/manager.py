import importlib
import inspect
import logging
import os
from pathlib import Path

from pydantic import ValidationError

from checks import Check, CheckFormInput

logger = logging.getLogger(__name__)


class CheckRegistry:
    def __init__(self):
        """Initialize an empty registry."""
        self._check_classes: list[type[Check]] = []

    def _load_check_dependencies(self) -> None:
        """Load dependencies for all registered check classes."""
        print("\nLoading check dependencies...")

        # Track which dependencies have been loaded to avoid duplicates
        loaded_dependencies: set[str] = set()

        for check_class in self._check_classes:
            # Get the fully qualified name for this check's load_dependencies method
            dependency_key = f"{check_class.__module__}.{check_class.__name__}"

            # Skip if this exact method has already been called
            if dependency_key in loaded_dependencies:
                continue

            try:
                # Call the check's load_dependencies method
                check_class.load_dependencies()
                loaded_dependencies.add(dependency_key)
            except Exception as e:
                raise Exception(
                    f"Failed to load dependencies for {check_class.name}: {e}"
                )

        print(f"All check dependencies loaded successfully\n")

    def load(self) -> None:
        """Discover and register all concrete Check subclasses from the implementations directory."""
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

                            self._check_classes.append(cls)
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

        logger.info("Checks loaded successfully (%d).", len(self._check_classes))
        logger.info("Found the following checks: %s", check_names)

        # Load dependencies for all discovered checks
        self._load_check_dependencies()

    def create_manager(self, model_xml) -> "CheckManager":
        """Instantiate a CheckManager for the given BPMN XML."""
        return CheckManager(model_xml=model_xml, check_classes=self._check_classes)

    def list_checks(self) -> list[dict[str, str | list[CheckFormInput]]]:
        """Return metadata for all registered checks."""
        return self.create_manager("").list_checks()


class CheckManager:
    def __init__(self, model_xml: str, check_classes: list[type[Check]]):
        """Instantiate all registered checks for the given BPMN XML."""
        self.model_xml: str = model_xml
        self.checks: dict[str, Check] = {}

        for check_class in check_classes:
            check = check_class(model_xml=model_xml)
            self.checks[check.id] = check

    def list_checks(
        self,
    ) -> list[dict[str, str | list[CheckFormInput]]]:
        """Return id, name, complexity and input scheme for each registered check."""
        checks = []
        for check in self.checks.values():
            entry: dict[str, str | list[CheckFormInput]] = {
                "id": check.id,
                "inputs": check.input_scheme,
                "check_complexity": check.check_complexity,
                "name": check.name,
            }
            checks.append(entry)

        return checks

    def get_check(self, name: str) -> Check:
        """Return the Check instance for the given check ID."""
        return self.checks[name]
