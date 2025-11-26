class PyProperty:
    property_name: str
    fulfilled: bool
    problematic_elements: list[str]
    description: str

    def __init__(
        self,
        property_name: str,
        fulfilled: bool,
        problematic_elements: list[str],
        description: str,
    ) -> None: ...

def analyze_safeness(model: str) -> PyProperty: ...
def analyze_dead_activities(model: str) -> PyProperty: ...
def analyze_option_to_complete(model: str) -> PyProperty: ...
def analyze_proper_completion(model: str) -> PyProperty: ...
