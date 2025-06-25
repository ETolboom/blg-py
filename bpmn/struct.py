from dataclasses import dataclass, field
from typing import List
from xml.etree.ElementTree import Element


@dataclass
class FlowElement:
    """ FlowElement represents an edge between two nodes within a pool"""

    # Id is the unique identifier for each FlowElement
    id: str

    # Label stores the user-added textual description of the sequence flow.
    label: str

    # Source is the id of the element that this edge is originating from.
    source: str

    # Target is the id of the element that this edge is going to.
    target: str


@dataclass
class PoolElement:
    """Represents any element within a pool"""

    # Name describes the type of PoolElement. Examples are "startEvent" or "task".
    name: str

    # Id is the unique identifier for each PoolElement
    id: str

    # Label stores the user-added textual description of an activity or task.
    label: str

    # Incoming contains all the IDs of elements that the PoolElement has incoming edges from.
    incoming: List[str] = field(default_factory=list)

    # Outgoing contains all the IDs of elements that the PoolElement has outgoing edges to.
    outgoing: List[str] = field(default_factory=list)

    # Children stores all the PoolElement instances nested within the current PoolElement
    children: List['PoolElement'] = field(default_factory=list)

    # GatewayDirection indicates the direction in which the gateway is going.
    # This can either be "Diverging" or "Converging".
    # Furthermore, this property only applies to gateways.
    gateway_direction: str = ""

    def to_flow_element(self, source_ref: str, target_ref: str) -> FlowElement:
        return FlowElement(
            id=self.id,
            label=self.label,
            source=source_ref,
            target=target_ref,
        )


@dataclass
class LaneElement:
    """Lane represents a lane within a pool."""

    # Id is the unique identifier for each Lane.
    id: str

    # Name is the name given to each Lane.
    name: str

    # Elements contains a list of the ids of all elements that belong to the lane
    elements: List[str] = field(default_factory=list)


@dataclass
class Pool:
    """The main container for BPMN elements."""

    # Id is the unique identifier for each Pool.
    id: str

    # Name is the name given to each Pool.
    name: str

    # Elements contains all elements within a Pool regardless of lane.
    elements: List[PoolElement] = field(default_factory=list)

    # Flows contains all edges between nodes within a pool
    flows: List[FlowElement] = field(default_factory=list)

    # Lanes contains information on all lanes within a pool.
    lanes: List[LaneElement] = field(default_factory=list)


def parse_lane_set(lane_set: Element) -> List[LaneElement]:
    lanes: List[LaneElement] = []
    for lane in lane_set:
        parsed_lane = LaneElement(id=lane.get("id"), name=lane.get("name"))
        nodes: List[str] = []
        for child in lane:
            element_type = child.tag.split("}")[-1]
            if element_type != "flowNodeRef":
                continue
            nodes.append(child.text)
        parsed_lane.elements = nodes
        lanes.append(parsed_lane)

    return lanes
