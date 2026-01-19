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
                    elif nested_child_type.endswith("EventDefinition"):
                        # Extract event type (e.g., "messageEventDefinition" -> "message")
                        event_type = nested_child_type.replace("EventDefinition", "")
                        element.event_definition = event_type.lower()

                pool_elements.append(element)

            # Second pass: Handle boundary events
            # Boundary events are attached to other elements and should be linked
            # via the boundary_events field
            for element in pool_elements:
                attached_to_ref = element.id
                # Find all boundary events attached to this element
                for child in pool:
                    element_type = child.tag.split("}")[-1]
                    if child.get("attachedToRef") == attached_to_ref:
                        # This is a boundary event attached to the current element
                        boundary_event_id = child.get("id") or ""
                        if boundary_event_id and boundary_event_id not in element.boundary_events:
                            element.boundary_events.append(boundary_event_id)

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

        print(f"[_search_task_in_path] Distance {current_distance}: Found element '{target_element.label}' (type: {target_element.name}, event_def: {target_element.event_definition})")

        # Check if this element matches
        # First check if it's a boundary event being searched for
        is_boundary_event = "boundary" in target_element.name.lower()
        task_label_lower = task_label.lower()

        if is_boundary_event:
            # Match boundary events by type
            is_match = False
            if "boundary" in task_label_lower and "event" in task_label_lower:
                is_match = True
                print(f"[_search_task_in_path] Generic boundary event match")
            elif target_element.event_definition and target_element.event_definition in task_label_lower:
                is_match = True
                print(f"[_search_task_in_path] Matched boundary event type '{target_element.event_definition}'")

            if is_match:
                print(f"[_search_task_in_path] MATCH on boundary event!")
                return current_distance, target_element, 1.0
        elif target_element.label:
            # Match other elements by label similarity
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
        print(f"[find_next_task] Boundary events: {starting_element.boundary_events}")

        current_element = starting_element
        visit_count = 0
        is_first_iteration = True

        # Iterate through the flow
        while visit_count < max_distance:
            # 2. Check boundary events first (if this is the first iteration)
            if is_first_iteration and current_element.boundary_events:
                print(f"[find_next_task] Element has {len(current_element.boundary_events)} boundary event(s), checking them first")
                for boundary_event_id in current_element.boundary_events:
                    # Find the boundary event element
                    boundary_event = None
                    for element in pool_for_element.elements:
                        if element.id == boundary_event_id:
                            boundary_event = element
                            break

                    if not boundary_event:
                        print(f"[find_next_task] Boundary event '{boundary_event_id}' not found, skipping")
                        continue

                    print(f"[find_next_task] Checking boundary event with type '{boundary_event.event_definition}' (ID: {boundary_event_id})")

                    # Check if the boundary event itself matches by type
                    # Extract event type from task_label (e.g., "message event" -> "message", "timer" -> "timer")
                    task_label_lower = task_label.lower()
                    is_boundary_event_match = False

                    # Match generic "boundary event" to any boundary event
                    if "boundary" in task_label_lower and "event" in task_label_lower:
                        is_boundary_event_match = True
                        print(f"[find_next_task] Generic boundary event match")
                    # Match specific event types (message, timer, error, signal, etc.)
                    elif boundary_event.event_definition:
                        # Check if the event type appears in the task label
                        if boundary_event.event_definition in task_label_lower:
                            is_boundary_event_match = True
                            print(f"[find_next_task] Matched event type '{boundary_event.event_definition}' in '{task_label}'")

                    if is_boundary_event_match:
                        print(f"[find_next_task] MATCH on boundary event!")
                        return 1, boundary_event, 1.0

                    # If boundary event doesn't match, continue down its outgoing path
                    if boundary_event.outgoing:
                        for outgoing_flow_id in boundary_event.outgoing:
                            print(f"[find_next_task] Trying boundary event outgoing flow ID: {outgoing_flow_id}")
                            result = self._search_task_in_path(outgoing_flow_id, pool_for_element, task_label, match_threshold, max_distance, visit_count + 1)
                            if result[1] is not None:  # Found a match
                                return result

                print(f"[find_next_task] No match found in boundary events, checking normal outgoing flows")

            # 3. Find the next element based on the outgoing id
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

    def _check_gateway_match(self, gateway: PoolElement, normalized_gateway_type: str, expected_outcomes: int, pool: Pool, gateway_label: str = "", outcome_labels: list[str] | None = None, check_gateway_label: bool = False, check_outcome_labels: bool = False, match_threshold: float = 0.8) -> bool:
        """Helper method to check if a gateway matches all required criteria"""
        if outcome_labels is None:
            outcome_labels = []

        # Check gateway type
        gateway_type_match = normalized_gateway_type in gateway.name.lower()
        if not gateway_type_match:
            print(f"[_check_gateway_match] Gateway type mismatch: expected '{normalized_gateway_type}', got '{gateway.name}'")
            return False

        # Check number of outgoing edges
        num_outgoing = len(gateway.outgoing)
        outcomes_match = num_outgoing == expected_outcomes
        if not outcomes_match:
            print(f"[_check_gateway_match] Outcomes mismatch: expected {expected_outcomes}, got {num_outgoing}")
            return False

        # Check gateway label if required
        if check_gateway_label:
            if not gateway.label:
                print(f"[_check_gateway_match] Gateway has no label, but label check is required")
                return False

            similarity_matrix = create_similarity_matrix([gateway_label], [gateway.label])
            label_score = similarity_matrix[0, 0].item()
            print(f"[_check_gateway_match] Gateway label similarity: '{gateway.label}' vs '{gateway_label}' = {label_score:.3f}")

            if label_score < match_threshold:
                print(f"[_check_gateway_match] Gateway label score {label_score:.3f} < threshold {match_threshold}")
                return False

        # Check outcome labels if required
        if check_outcome_labels and outcome_labels:
            # Get the labels of the outgoing flows
            flow_labels = []
            for flow_id in gateway.outgoing:
                for flow in pool.flows:
                    if flow.id == flow_id:
                        flow_labels.append(flow.label)
                        break

            print(f"[_check_gateway_match] Gateway outgoing flow labels: {flow_labels}")
            print(f"[_check_gateway_match] Expected outcome labels: {outcome_labels}")

            # Check if we have the right number of flow labels
            if len(flow_labels) != len(outcome_labels):
                print(f"[_check_gateway_match] Number of flow labels ({len(flow_labels)}) != expected ({len(outcome_labels)})")
                return False

            # Check if all expected outcome labels match (order doesn't matter)
            # For each expected label, find a matching flow label
            # Empty expected labels are automatically accepted (no matching needed)
            matched_flow_indices = set()
            for i, expected_label in enumerate(outcome_labels):
                # If expected label is empty, automatically accept it (don't check)
                if not expected_label or expected_label.strip() == "":
                    print(f"[_check_gateway_match] Outcome {i} has empty label, accepting without matching")
                    # Still consume one flow slot
                    for j in range(len(flow_labels)):
                        if j not in matched_flow_indices:
                            matched_flow_indices.add(j)
                            break
                    continue

                # Non-empty expected label - must find a match
                found_match = False
                for j, flow_label in enumerate(flow_labels):
                    if j in matched_flow_indices:
                        continue

                    # Compare with similarity
                    similarity_matrix = create_similarity_matrix([expected_label], [flow_label])
                    score = similarity_matrix[0, 0].item()
                    print(f"[_check_gateway_match] Comparing outcome '{expected_label}' with flow '{flow_label}': {score:.3f}")

                    if score >= match_threshold:
                        matched_flow_indices.add(j)
                        found_match = True
                        break

                if not found_match:
                    print(f"[_check_gateway_match] Could not find match for expected outcome '{expected_label}'")
                    return False

        print(f"[_check_gateway_match] Gateway MATCHES all criteria!")
        return True

    def find_next_gateway(self, starting_element_id: str, gateway_type: str, expected_outcomes: int, max_distance: int = 2, gateway_label: str = "", outcome_labels: list[str] | None = None, check_gateway_label: bool = False, check_outcome_labels: bool = False, match_threshold: float = 0.8) -> tuple[int, PoolElement | None, float]:
        """Find the next gateway element based on type and number of outcomes

        Args:
            starting_element_id: ID of element to start search from
            gateway_type: Type of gateway (xor, and, or, etc.)
            expected_outcomes: Number of expected outgoing edges
            max_distance: Maximum distance to search
            gateway_label: Label to match if check_gateway_label is True
            outcome_labels: List of outcome labels to match if check_outcome_labels is True
            check_gateway_label: Whether to check the gateway's label
            check_outcome_labels: Whether to check the outcome (flow) labels
            match_threshold: Similarity threshold for label matching
        """
        if outcome_labels is None:
            outcome_labels = []

        print(f"\n[find_next_gateway] Starting search from element ID: {starting_element_id}")
        print(f"[find_next_gateway] Looking for GATEWAY: type={gateway_type}, expected_outcomes={expected_outcomes} (max_distance={max_distance})")
        print(f"[find_next_gateway] Check gateway label: {check_gateway_label} ('{gateway_label}')")
        print(f"[find_next_gateway] Check outcome labels: {check_outcome_labels} ({outcome_labels})")

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
        print(f"[find_next_gateway] Boundary events: {starting_element.boundary_events}")

        current_element = starting_element
        visit_count = 0

        # Iterate through the flow - for gateways, we can traverse through multiple outgoing edges
        while visit_count < max_distance:
            # 2. Check boundary events first (only from starting element)
            if visit_count == 0 and current_element.boundary_events:
                print(f"[find_next_gateway] Element has {len(current_element.boundary_events)} boundary event(s), checking them first")
                for boundary_event_id in current_element.boundary_events:
                    # Find the boundary event element
                    boundary_event = None
                    for element in pool_for_element.elements:
                        if element.id == boundary_event_id:
                            boundary_event = element
                            break

                    if not boundary_event:
                        print(f"[find_next_gateway] Boundary event '{boundary_event_id}' not found, skipping")
                        continue

                    print(f"[find_next_gateway] Checking boundary event '{boundary_event.label}' (ID: {boundary_event_id})")

                    # Check if the boundary event itself is a gateway (unlikely but possible)
                    is_gateway = "gateway" in boundary_event.name.lower()
                    if is_gateway:
                        if self._check_gateway_match(boundary_event, normalized_gateway_type, expected_outcomes, pool_for_element, gateway_label, outcome_labels, check_gateway_label, check_outcome_labels, match_threshold):
                            print(f"[find_next_gateway] MATCH on boundary event gateway!")
                            return 1, boundary_event, 1.0

                    # Check the boundary event's outgoing paths for gateways
                    if boundary_event.outgoing:
                        for outgoing_flow_id in boundary_event.outgoing:
                            # Find the flow target
                            target_element_id = None
                            for flow in pool_for_element.flows:
                                if flow.id == outgoing_flow_id:
                                    target_element_id = flow.target
                                    break

                            if not target_element_id:
                                continue

                            # Find the target element
                            next_element = None
                            for element in pool_for_element.elements:
                                if element.id == target_element_id:
                                    next_element = element
                                    break

                            if not next_element:
                                continue

                            print(f"[find_next_gateway] Found element after boundary event: '{next_element.label}' (type: {next_element.name})")

                            # Check if this is the gateway we're looking for
                            is_gateway = "gateway" in next_element.name.lower()
                            if is_gateway:
                                if self._check_gateway_match(next_element, normalized_gateway_type, expected_outcomes, pool_for_element, gateway_label, outcome_labels, check_gateway_label, check_outcome_labels, match_threshold):
                                    print(f"[find_next_gateway] MATCH on gateway after boundary event!")
                                    return 1, next_element, 1.0

                print(f"[find_next_gateway] No match found in boundary events, checking normal outgoing flows")

            # 3. Check all outgoing flows from current element
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
                    if self._check_gateway_match(next_element, normalized_gateway_type, expected_outcomes, pool_for_element, gateway_label, outcome_labels, check_gateway_label, check_outcome_labels, match_threshold):
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
