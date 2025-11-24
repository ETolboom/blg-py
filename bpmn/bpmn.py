from xml.etree import ElementTree
from typing import List
from bpmn.struct import Pool, PoolElement, parse_lane_set


class Bpmn:
    """BPMN is an internal representation of the BPMN XML model."""

    def __init__(self, xml_string: str) -> None:
        self.pools: List[Pool] = []
        self.__parse_xml(xml_string)

    def __str__(self):
        """Returns a string representation of the BPMN"""
        out = f"Model has {len(self.pools)} pools.\n"
        for pool in self.pools:
            out += f"{pool.name} ({len(pool.lanes)} lanes)\n"
            if len(pool.lanes) > 0:
                for lane in pool.lanes:
                    out += f"\t{lane.name} ({len(lane.elements)} elements)\n"

        return out

    def __parse_xml(self, xml_string: str):
        root = ElementTree.fromstring(xml_string)

        namespace = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"}

        pools = root.findall(".//bpmn:process", namespace)

        for pool in pools:
            parsed_pool = Pool(name=pool.get("name"), id=pool.get("id"))
            pool_elements: List[PoolElement] = []
            for child in pool:
                element_type = child.tag.split("}")[-1]

                element = PoolElement(
                    name=element_type,
                    id=child.get("id"),
                    label=child.get("name"),
                    # We can always try getting the direction, since .get() returns None if not found.
                    gateway_direction=child.get("gatewayDirection"),
                )

                if not element.id:
                    # Id-less elements are not useful
                    continue

                if element_type == "laneSet":
                    parsed_pool.lanes = parse_lane_set(child)
                    continue
                elif element_type == "sequenceFlow":
                    parsed_pool.flows.append(
                        element.to_flow_element(
                            child.get("sourceRef"), child.get("targetRef")
                        )
                    )
                    continue

                for nested_child in child:
                    nested_child_type = nested_child.tag.split("}")[-1]
                    if nested_child_type == "incoming":
                        # TODO: Support parsing DataObjects
                        # These elements do not affect the flow of the entire graph
                        # case "ioSpecification":
                        # case "dataOutputAssociation":
                        element.incoming.append(nested_child.text)
                    elif nested_child_type == "outgoing":
                        element.outgoing.append(nested_child.text)
                pool_elements.append(element)

            parsed_pool.elements = pool_elements
            self.pools.append(parsed_pool)

    def extract_tasks(self) -> List[str]:
        tasks: list[str] = []
        for pool in self.pools:
            for element in pool.elements:
                if (element.name != "task") or (element.label == ""):
                    continue
                tasks.append(element.label)

        return tasks
