from xml.etree import ElementTree

from bpmn.struct import Pool, PoolElement, parse_lane_set
from utils.similarity import create_similarity_matrix


class Bpmn:
    """BPMN is an internal representation of the BPMN XML model."""

    def __init__(self, xml_string: str) -> None:
        self.pools: list[Pool] = []
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
            parsed_pool = Pool(name=pool.get("name") or "", id=pool.get("id") or "")
            pool_elements: list[PoolElement] = []
            for child in pool:
                element_type = child.tag.split("}")[-1]

                element = PoolElement(
                    name=element_type,
                    id=child.get("id") or "",
                    label=child.get("name") or "",
                    # We can always try getting the direction, since .get() returns None if not found.
                    gateway_direction=child.get("gatewayDirection") or "",
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
                            child.get("sourceRef") or "", child.get("targetRef") or ""
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
                        element.incoming.append(nested_child.text or "")
                    elif nested_child_type == "outgoing":
                        element.outgoing.append(nested_child.text or "")
                pool_elements.append(element)

            parsed_pool.elements = pool_elements
            self.pools.append(parsed_pool)

    def find_task(self, task_label: str, match_threshold: float = 0.6) -> tuple[PoolElement, float] | None:
        # Collect all elements with labels
        elements_with_labels = []
        labels = []

        for pool in self.pools:
            for element in pool.elements:
                if element.label:
                    elements_with_labels.append(element)
                    labels.append(element.label)

        if not labels:
            return None

        # Compute similarity for all labels at once
        similarity_matrix = create_similarity_matrix([task_label], labels)

        # Find the first element that meets the threshold
        for i, element in enumerate(elements_with_labels):
            similarity_score = similarity_matrix[0, i].item()
            if similarity_score >= match_threshold:
                return element, similarity_score

        return None

    def _search_task_in_path(self, flow_id: str, pool: Pool, task_label: str, match_threshold: float, max_distance: int, current_distance: int) -> tuple[int, PoolElement | None, float]:
        """Helper method to search for a task along a specific flow path"""
        # Find the target of this flow
        target_element_id = None
        for flow in pool.flows:
            if flow.id == flow_id:
                target_element_id = flow.target
                break

        if not target_element_id:
            print(f"[_search_task_in_path] Flow '{flow_id}' not found")
            return -1, None, 0.0

        # Find the target element
        target_element = None
        for element in pool.elements:
            if element.id == target_element_id:
                target_element = element
                break

        if not target_element:
            print(f"[_search_task_in_path] Target element '{target_element_id}' not found")
            return -1, None, 0.0

        print(f"[_search_task_in_path] Distance {current_distance}: Found element '{target_element.label}' (type: {target_element.name})")

        # Check if this element matches
        if target_element.label:
            similarity_matrix = create_similarity_matrix([task_label], [target_element.label])
            similarity_score = similarity_matrix[0, 0].item()
            print(f"[_search_task_in_path] Comparing '{target_element.label}' with '{task_label}': {similarity_score:.3f}")

            if similarity_score >= match_threshold:
                print(f"[_search_task_in_path] MATCH! Score {similarity_score:.3f} >= threshold {match_threshold}")
                return current_distance, target_element, similarity_score

        # If no match and we haven't reached max distance, continue searching
        if current_distance >= max_distance:
            print(f"[_search_task_in_path] Reached max_distance ({max_distance})")
            return -1, None, 0.0

        # Continue along the path if element has exactly 1 outgoing edge
        # OR if it's a gateway (which can have multiple outgoing edges)
        is_gateway = "gateway" in target_element.name.lower()

        if len(target_element.outgoing) == 1:
            return self._search_task_in_path(target_element.outgoing[0], pool, task_label, match_threshold, max_distance, current_distance + 1)
        elif is_gateway and len(target_element.outgoing) > 1:
            print(f"[_search_task_in_path] Element is a gateway with {len(target_element.outgoing)} outgoing edges, cannot continue (ambiguous path)")
            return -1, None, 0.0
        else:
            print(f"[_search_task_in_path] Element has {len(target_element.outgoing)} outgoing edges and is not a gateway, stopping search on this path")
            return -1, None, 0.0

    def find_next_task(self, starting_element_id: str, task_label: str, max_distance: int = 2, match_threshold: float = 0.8) -> tuple[int, PoolElement | None, float]:
        """Find the next task/event element following a linear flow (1 outgoing edge per element)"""
        print(f"\n[find_next_task] Starting search from element ID: {starting_element_id}")
        print(f"[find_next_task] Looking for TASK: '{task_label}' (max_distance={max_distance}, threshold={match_threshold})")

        # 1. Find the exact starting element
        starting_element: PoolElement | None = None
        pool_for_element: Pool | None = None

        for pool in self.pools:
            for element in pool.elements:
                if element.id == starting_element_id:
                    starting_element = element
                    pool_for_element = pool
                    break
            if starting_element:
                break

        if not starting_element or not pool_for_element:
            raise ValueError(f"Starting element with id '{starting_element_id}' not found")

        print(f"[find_next_task] Starting element: '{starting_element.label}' (ID: {starting_element.id})")
        print(f"[find_next_task] Outgoing connections: {starting_element.outgoing}")

        current_element = starting_element
        visit_count = 0
        is_first_iteration = True

        # Iterate through the flow
        while visit_count < max_distance:
            # 2. Find the next element based on the outgoing id
            if not current_element.outgoing:
                # No outgoing edges, can't continue
                print(f"[find_next_task] No outgoing edges from '{current_element.label}', stopping")
                return -1, None, 0.0

            # Starting element can have multiple outgoing edges (e.g., if starting from a gateway)
            # But intermediate elements should have exactly 1
            if not is_first_iteration and len(current_element.outgoing) != 1:
                # A task should have exactly 1 outgoing element
                print(f"[find_next_task] Intermediate element has {len(current_element.outgoing)} outgoing edges (expected 1), stopping")
                return -1, None, 0.0

            # If starting element has multiple outgoing edges, check all paths
            if is_first_iteration and len(current_element.outgoing) > 1:
                print(f"[find_next_task] Starting element has {len(current_element.outgoing)} outgoing edges, checking all paths")
                for outgoing_flow_id in current_element.outgoing:
                    print(f"[find_next_task] Trying flow ID: {outgoing_flow_id}")
                    # Try to find a match in this path
                    result = self._search_task_in_path(outgoing_flow_id, pool_for_element, task_label, match_threshold, max_distance, visit_count + 1)
                    if result[1] is not None:  # Found a match
                        return result
                # No match found in any path
                print(f"[find_next_task] No match found in any outgoing path")
                return -1, None, 0.0

            is_first_iteration = False

            # Get the first (and only) outgoing flow id
            outgoing_flow_id = current_element.outgoing[0]
            print(f"[find_next_task] Following flow ID: {outgoing_flow_id}")

            # Find the flow in the pool's flows
            target_element_id = None
            for flow in pool_for_element.flows:
                if flow.id == outgoing_flow_id:
                    target_element_id = flow.target
                    break

            if not target_element_id:
                # Flow not found
                print(f"[find_next_task] Flow '{outgoing_flow_id}' not found in pool flows, stopping")
                return -1, None, 0.0

            print(f"[find_next_task] Flow targets element ID: {target_element_id}")

            # Find the target element
            next_element = None
            for element in pool_for_element.elements:
                if element.id == target_element_id:
                    next_element = element
                    break

            if not next_element:
                # Target element not found
                print(f"[find_next_task] Target element '{target_element_id}' not found, stopping")
                return -1, None, 0.0

            # 3. Increment visit count
            visit_count += 1
            print(f"[find_next_task] Visit {visit_count}: Found element '{next_element.label}' (type: {next_element.name})")

            # 4. Check if element matches using label similarity
            if next_element.label:
                similarity_matrix = create_similarity_matrix([task_label], [next_element.label])
                similarity_score = similarity_matrix[0, 0].item()

                print(f"[find_next_task] Comparing '{next_element.label}' with '{task_label}': {similarity_score:.3f}")

                if similarity_score >= match_threshold:
                    print(f"[find_next_task] MATCH! Score {similarity_score:.3f} >= threshold {match_threshold}")
                    return visit_count, next_element, similarity_score
                else:
                    print(f"[find_next_task] No match (score {similarity_score:.3f} < threshold {match_threshold})")
            else:
                print(f"[find_next_task] Element has no label, skipping comparison")

            # 5. If visit count and max_distance are equal, stop
            if visit_count >= max_distance:
                print(f"[find_next_task] Reached max_distance ({max_distance}), stopping")
                return -1, None, 0.0

            # 6. Continue with next element
            current_element = next_element

        return -1, None, 0.0

    def find_next_gateway(self, starting_element_id: str, gateway_type: str, expected_outcomes: int, max_distance: int = 2) -> tuple[int, PoolElement | None, float]:
        """Find the next gateway element based on type and number of outcomes"""
        print(f"\n[find_next_gateway] Starting search from element ID: {starting_element_id}")
        print(f"[find_next_gateway] Looking for GATEWAY: type={gateway_type}, expected_outcomes={expected_outcomes} (max_distance={max_distance})")

        # Map gateway type aliases to BPMN names
        gateway_type_mapping = {
            "xor": "exclusive",
            "exclusive": "exclusive",
            "or": "inclusive",
            "inclusive": "inclusive",
            "and": "parallel",
            "parallel": "parallel",
            "event": "event",
            "eventbased": "event",
            "complex": "complex"
        }

        normalized_gateway_type = gateway_type_mapping.get(gateway_type.lower(), gateway_type.lower())
        print(f"[find_next_gateway] Normalized gateway type: '{normalized_gateway_type}'")

        # 1. Find the exact starting element
        starting_element: PoolElement | None = None
        pool_for_element: Pool | None = None

        for pool in self.pools:
            for element in pool.elements:
                if element.id == starting_element_id:
                    starting_element = element
                    pool_for_element = pool
                    break
            if starting_element:
                break

        if not starting_element or not pool_for_element:
            raise ValueError(f"Starting element with id '{starting_element_id}' not found")

        print(f"[find_next_gateway] Starting element: '{starting_element.label}' (ID: {starting_element.id})")
        print(f"[find_next_gateway] Outgoing connections: {starting_element.outgoing}")

        current_element = starting_element
        visit_count = 0

        # Iterate through the flow - for gateways, we can traverse through multiple outgoing edges
        while visit_count < max_distance:
            # 2. Check all outgoing flows from current element
            if not current_element.outgoing:
                print(f"[find_next_gateway] No outgoing edges from '{current_element.label}', stopping")
                return -1, None, 0.0

            print(f"[find_next_gateway] Element has {len(current_element.outgoing)} outgoing edge(s)")

            # For each outgoing flow, check the target element
            for outgoing_flow_id in current_element.outgoing:
                print(f"[find_next_gateway] Checking flow ID: {outgoing_flow_id}")

                # Find the flow in the pool's flows
                target_element_id = None
                for flow in pool_for_element.flows:
                    if flow.id == outgoing_flow_id:
                        target_element_id = flow.target
                        break

                if not target_element_id:
                    print(f"[find_next_gateway] Flow '{outgoing_flow_id}' not found, skipping")
                    continue

                print(f"[find_next_gateway] Flow targets element ID: {target_element_id}")

                # Find the target element
                next_element = None
                for element in pool_for_element.elements:
                    if element.id == target_element_id:
                        next_element = element
                        break

                if not next_element:
                    print(f"[find_next_gateway] Target element '{target_element_id}' not found, skipping")
                    continue

                print(f"[find_next_gateway] Found element '{next_element.label}' (type: {next_element.name})")

                # Check if this element is a gateway with matching criteria
                is_gateway = "gateway" in next_element.name.lower()
                print(f"[find_next_gateway] Is gateway: {is_gateway}")

                if is_gateway:
                    # Check gateway type match using normalized type
                    gateway_type_match = normalized_gateway_type in next_element.name.lower()
                    print(f"[find_next_gateway] Gateway type match: {gateway_type_match} (looking for '{normalized_gateway_type}', found '{next_element.name}')")

                    # Check number of outgoing edges
                    num_outgoing = len(next_element.outgoing)
                    outcomes_match = num_outgoing == expected_outcomes
                    print(f"[find_next_gateway] Outgoing edges: {num_outgoing}, Expected: {expected_outcomes}, Match: {outcomes_match}")

                    if gateway_type_match and outcomes_match:
                        # Match found at visit_count + 1 (since we're looking at next elements)
                        print(f"[find_next_gateway] GATEWAY MATCH! Found at distance {visit_count + 1}")
                        return visit_count + 1, next_element, 1.0

            # If no match found in immediate neighbors, we need to traverse deeper
            # Only continue if current element is a gateway (can have multiple outgoing edges)
            # Tasks should not have multiple outgoing edges
            is_current_gateway = "gateway" in current_element.name.lower()

            if len(current_element.outgoing) > 0:
                visit_count += 1

                if visit_count >= max_distance:
                    print(f"[find_next_gateway] Reached max_distance ({max_distance}), stopping")
                    return -1, None, 0.0

                # Only continue traversal if we're at a gateway or have exactly 1 outgoing edge
                if is_current_gateway or len(current_element.outgoing) == 1:
                    # Take first outgoing flow to continue
                    outgoing_flow_id = current_element.outgoing[0]
                    target_element_id = None
                    for flow in pool_for_element.flows:
                        if flow.id == outgoing_flow_id:
                            target_element_id = flow.target
                            break

                    if target_element_id:
                        for element in pool_for_element.elements:
                            if element.id == target_element_id:
                                current_element = element
                                print(f"[find_next_gateway] Moving to next element: '{current_element.label}' (type: {current_element.name})")
                                break
                    else:
                        return -1, None, 0.0
                else:
                    print(f"[find_next_gateway] Current element is not a gateway and has {len(current_element.outgoing)} outgoing edges, stopping")
                    return -1, None, 0.0
            else:
                return -1, None, 0.0

        return -1, None, 0.0

    def extract_tasks(self) -> list[str]:
        tasks: list[str] = []
        for pool in self.pools:
            for element in pool.elements:
                # Element is abstract "task" or ServiceTask, SendTask, XYZTask, etc.
                if (element.name == "task") or (element.label.endswith("Task")):
                    tasks.append(element.label)
                continue

        return tasks
