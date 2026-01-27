from typing import ClassVar, Optional
from dataclasses import dataclass, field

from algorithms import Algorithm, AlgorithmComplexity, AlgorithmFormInput, AlgorithmResult
from pydantic import BaseModel

from bpmn.bpmn import Bpmn
from bpmn.struct import PoolElement


class NodeHandle(BaseModel):
    id: str
    type: str
    nodeId: str
    position: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class Handles(BaseModel):
    source: Optional[list[NodeHandle]] = None
    target: Optional[list[NodeHandle]] = None


class NodeData(BaseModel):
    label: str
    score: Optional[float] = None
    checkType: Optional[str] = None
    elementType: Optional[str] = None
    gatewayType: Optional[str] = None
    gatewayOutcomes: Optional[list[str]] = None
    isGatewayChecked: Optional[bool] = None
    isOutcomeChecked: Optional[bool] = None
    relationshipType: Optional[str] = None
    idealDistance: Optional[int] = None
    maxDistance: Optional[int] = None
    connectorType: Optional[str] = None
    eventType: Optional[str] = None
    eventPosition: Optional[str] = None
    eventBehavior: Optional[str] = None


class GraphNode(BaseModel):
    id: str
    type: str
    handleBounds: Optional[Handles] = None
    data: NodeData
    dimensions: Optional[dict] = None
    computedPosition: Optional[dict] = None
    position: Optional[dict] = None
    selected: Optional[bool] = None
    dragging: Optional[bool] = None
    resizing: Optional[bool] = None
    initialized: Optional[bool] = None
    isParent: Optional[bool] = None
    events: Optional[dict] = None


class Edge(BaseModel):
    id: str
    type: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    data: Optional[dict] = None
    events: Optional[dict] = None
    label: Optional[str] = None
    sourceNode: Optional[GraphNode] = None
    targetNode: Optional[GraphNode] = None
    sourceX: Optional[float] = None
    sourceY: Optional[float] = None
    targetX: Optional[float] = None
    targetY: Optional[float] = None
    outcome: Optional[str] = None


class WorkflowData(BaseModel):
    nodes: list[GraphNode]
    edges: list[Edge]

    def next(self, node: GraphNode) -> list[GraphNode]:
        """Get next nodes, automatically skipping over any notes nodes"""
        outgoing_nodes: list[GraphNode] = []
        for edge in self.edges:
            if edge.source == node.id:
                # Find the target node by ID
                for target_node in self.nodes:
                    if target_node.id == edge.target:
                        # If target is a notes node, recursively get its next nodes
                        if target_node.type == "notesNode":
                            outgoing_nodes.extend(self.next(target_node))
                        else:
                            outgoing_nodes.append(target_node)
                        break
        return outgoing_nodes


class DecisionTreeNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    score: float = 0.0

    # For gateway nodes
    outcomes: Optional[list[str]] = None
    children: Optional[dict[str, "DecisionTreeNode"]] = None

    # For element nodes
    next_node: Optional["DecisionTreeNode"] = None

    # For tracking problematic paths
    is_problematic: bool = False


class ParsedTree(BaseModel):
    total_points: int
    nodes: list[DecisionTreeNode]


class ConnectorNode(BaseModel):
    node_id: str
    node_type: str

    # There are only 'XOR' and 'AND' nodes at the moment which have 2 inputs.
    # Every time we pass by a node successfully we increment
    visit_count: int = 0

    minimum_visit_count: int

    # Track which branches have visited this connector
    visited_by_branches: set[str] = set()

    def register_visit(self, branch_id: str) -> bool:
        """Register a branch visit, return True if convergence complete"""
        if branch_id not in self.visited_by_branches:
            self.visited_by_branches.add(branch_id)
            self.visit_count += 1

        return self.visit_count >= self.minimum_visit_count


@dataclass
class MatchDetail:
    """Detailed information about a single match"""
    workflow_node_id: str
    workflow_label: str
    bpmn_element_id: str
    bpmn_label: str
    match_score: float
    distance: int
    ideal_distance: int
    max_distance: int
    minimal_match_threshold: float  # Minimum acceptable threshold (default: 0.6)
    ideal_match_threshold: float    # Ideal threshold (default: 0.8)
    is_correct: bool                # True if match_score >= minimal_match_threshold
    is_ideal_distance: bool         # True if distance == ideal_distance
    is_ideal_match: bool            # True if match_score >= ideal_match_threshold


@dataclass
class TemplateEvaluationResult:
    """Result from evaluating a single template within a group"""
    template_id: str
    template_name: str
    description: str
    score: float
    confidence: float
    match_details: list[MatchDetail]
    success: bool  # True if evaluation completed without errors


class BehavioralResult(BaseModel):
    """Extended result type for behavioral grading with detailed match information"""
    # AlgorithmResult fields
    id: str
    name: str
    category: str
    description: str
    fulfilled: bool
    confidence: float
    problematic_elements: list[str] = []
    inputs: list[AlgorithmFormInput] = []

    # Additional behavioral-specific fields
    match_details: list[MatchDetail] = []
    total_score: float = 0.0
    total_matches: int = 0


class GroupEvaluationResult(BaseModel):
    """Result from evaluating a template group"""
    group_id: str
    group_name: str
    group_description: str
    condition: str  # "XOR" or "AND"

    # Individual template results
    template_results: list[TemplateEvaluationResult]

    # Aggregated result (MAX scoring)
    final_score: float              # MAX of template scores
    best_template_id: str           # Which template achieved max score
    overall_confidence: float       # From best template

    # For rubric compatibility
    match_details: list[MatchDetail]
    problematic_elements: list[str]
    fulfilled: bool


@dataclass
class TraversalContext:
    """Complete traversal state for a branch"""
    workflow_pos: GraphNode          # Current position in workflow
    bpmn_pos: PoolElement           # Current position in BPMN model
    match_scores: list[float] = field(default_factory=list)  # All match scores so far
    match_details: list[MatchDetail] = field(default_factory=list)  # Detailed match info
    total_score: float = 0.0        # Accumulated penalty score
    ideal_distance: int = 1         # Updated by followedBy nodes
    max_distance: int = 2           # Updated by followedBy nodes
    minimal_match_threshold: float = 0.6  # Minimum acceptable match score
    ideal_match_threshold: float = 0.8    # Ideal match score
    visited_nodes: set[str] = field(default_factory=set)  # Cycle detection

    def clone(self) -> "TraversalContext":
        """Deep copy for branch exploration"""
        return TraversalContext(
            workflow_pos=self.workflow_pos,
            bpmn_pos=self.bpmn_pos,
            match_scores=self.match_scores.copy(),
            match_details=self.match_details.copy(),
            total_score=self.total_score,
            ideal_distance=self.ideal_distance,
            max_distance=self.max_distance,
            minimal_match_threshold=self.minimal_match_threshold,
            ideal_match_threshold=self.ideal_match_threshold,
            visited_nodes=self.visited_nodes.copy()
        )

    def update_distance_constraints(self, node: GraphNode):
        """Update from followedBy connector"""
        self.ideal_distance = node.data.idealDistance or 1
        self.max_distance = node.data.maxDistance or 2

    def apply_match_result(self, bpmn_result: tuple, workflow_node: GraphNode):
        """Apply BPMN match result to context"""
        visit_count, bpmn_elem, match_score = bpmn_result

        self.bpmn_pos = bpmn_elem
        self.match_scores.append(match_score)
        self.total_score += workflow_node.data.score or 0.0

        # Create detailed match record
        match_detail = MatchDetail(
            workflow_node_id=workflow_node.id,
            workflow_label=workflow_node.data.label,
            bpmn_element_id=bpmn_elem.id,
            bpmn_label=bpmn_elem.label,
            match_score=match_score,
            distance=visit_count,
            ideal_distance=self.ideal_distance,
            max_distance=self.max_distance,
            minimal_match_threshold=self.minimal_match_threshold,
            ideal_match_threshold=self.ideal_match_threshold,
            is_correct=match_score >= self.minimal_match_threshold,
            is_ideal_distance=visit_count == self.ideal_distance,
            is_ideal_match=match_score >= self.ideal_match_threshold
        )
        self.match_details.append(match_detail)

        if visit_count > self.max_distance:
            raise Exception(f"Distance {visit_count} exceeds max {self.max_distance}")

        if visit_count > self.ideal_distance:
            bpmn_elem.flagged = True

    @property
    def confidence(self) -> float:
        """Calculate average match confidence"""
        return sum(self.match_scores) / len(self.match_scores) if self.match_scores else 0.0


@dataclass
class DivergencePoint:
    """Info about gateway divergence"""
    gateway_node: GraphNode
    bpmn_state: PoolElement         # BPMN position to restore
    branch_starts: list[GraphNode]   # Outgoing branches
    connector_id: str                # Where branches converge
    connector_type: str              # "AND" or "XOR"


def _find_start_node(nodes: list[GraphNode], edges: list[Edge]) -> Optional[GraphNode]:
    # Collect all node IDs that are targets (have incoming edges)
    target_node_ids = {edge.target for edge in edges}

    # Find nodes that are not targets of any edge, excluding notes nodes
    start_nodes = [node for node in nodes if node.id not in target_node_ids and node.type != "notesNode"]

    if len(start_nodes) == 0:
        return None
    elif len(start_nodes) == 1:
        return start_nodes[0]
    else:
        # Multiple start nodes
        start_node_ids = [node.id for node in start_nodes]
        raise ValueError(f"Multiple start nodes found: {start_node_ids}. Expected exactly one start node.")


def _extract_connector_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    connector_types = ["andConnector", "xorConnector"]
    connector_nodes = [node for node in nodes if node.type in connector_types]
    return connector_nodes


class BehavioralRuleCheck(Algorithm):
    id: ClassVar[str] = "behavioral_rule"
    name: ClassVar[str] = "Behavioral Rule"
    description: ClassVar[str] = "Check the model based on a complex set of rules"
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.COMPLEX
    threshold: ClassVar[float] = 0.0

    def is_applicable(self) -> bool:
        # Should not appear during onboarding
        return False

    def inputs(self) -> list[AlgorithmFormInput]:
        # Behavioral rules have different logic
        return []

    def analyze_tree(self, tree: DecisionTreeNode) -> tuple[bool, list[str], float]:
        """
        Analyze the decision tree for behavioral rules
        Returns: (fulfilled, problematic_elements, confidence)
        """
        problematic = []
        total_score = 0.0
        node_count = 0

        def traverse(node: DecisionTreeNode):
            nonlocal total_score, node_count

            node_count += 1
            total_score += node.score

            # Check if node has problematic score
            if node.score > self.threshold:
                problematic.append(node.node_id)
                node.is_problematic = True

            # Traverse children
            if node.children:
                for child in node.children.values():
                    traverse(child)
            elif node.next_node:
                traverse(node.next_node)

        traverse(tree)

        avg_score = total_score / node_count if node_count > 0 else 0.0
        fulfilled = len(problematic) == 0
        confidence = 1.0 - min(avg_score, 1.0)

        return fulfilled, problematic, confidence

    def _extract_and_map_connectors(self, workflow: WorkflowData) -> dict[str, ConnectorNode]:
        """Extract connectors and create lookup map"""
        connector_list = _extract_connector_nodes(workflow.nodes)

        connector_map = {}
        for node in connector_list:
            minimum_visit = 2 if node.data.label == "AND" else 1
            connector_map[node.id] = ConnectorNode(
                node_id=node.id,
                node_type=node.data.label,
                minimum_visit_count=minimum_visit
            )

        return connector_map

    def _find_convergence_point(self, branches: list[GraphNode], workflow: WorkflowData) -> str:
        """Find connector ID where ALL branches converge"""
        # For each branch, find all reachable connectors
        branch_connectors: list[set[str]] = []

        for branch in branches:
            connectors_in_path: set[str] = set()
            visited: set[str] = set()
            queue = [branch]

            while queue:
                current = queue.pop(0)

                if current.id in visited:
                    continue
                visited.add(current.id)

                # Check if this is a connector
                if current.type in ["andConnector", "xorConnector"]:
                    connectors_in_path.add(current.id)

                # Add next nodes to queue
                next_nodes = workflow.next(current)
                queue.extend(next_nodes)

            branch_connectors.append(connectors_in_path)

        # Find connectors that are reachable from ALL branches (intersection)
        if not branch_connectors:
            raise Exception("No branches provided for convergence point search")

        common_connectors = branch_connectors[0]
        for connectors_set in branch_connectors[1:]:
            common_connectors = common_connectors.intersection(connectors_set)

        if not common_connectors:
            raise Exception("No common convergence connector found for branches")

        # If multiple common connectors, return the closest one (first one encountered in BFS from any branch)
        visited: set[str] = set()
        queue = [branches[0]]

        while queue:
            current = queue.pop(0)

            if current.id in visited:
                continue
            visited.add(current.id)

            if current.id in common_connectors:
                return current.id

            next_nodes = workflow.next(current)
            queue.extend(next_nodes)

        raise Exception("Failed to find closest common convergence connector")

    def _find_bpmn_match(self, context: TraversalContext, workflow_node: GraphNode,
                         model: Bpmn) -> tuple[int, PoolElement | None, float]:
        """Find matching BPMN element for workflow node"""
        if workflow_node.data.checkType == "gateway":
            gateway_type = getattr(workflow_node.data, 'gatewayType', None)
            gateway_outcomes = getattr(workflow_node.data, 'gatewayOutcomes', None)
            expected_outcomes = len(gateway_outcomes) if gateway_outcomes is not None else None

            # Get label checking flags (default to False if not provided)
            is_gateway_checked = getattr(workflow_node.data, 'isGatewayChecked', False) or False
            is_outcome_checked = getattr(workflow_node.data, 'isOutcomeChecked', False) or False

            if gateway_type and expected_outcomes:
                return model.find_next_gateway(
                    context.bpmn_pos.id,
                    gateway_type,
                    expected_outcomes,
                    max_distance=context.max_distance,
                    gateway_label=workflow_node.data.label,
                    outcome_labels=gateway_outcomes if gateway_outcomes else [],
                    check_gateway_label=is_gateway_checked,
                    check_outcome_labels=is_outcome_checked,
                    match_threshold=0.8
                )
            else:
                raise Exception(f"Gateway node missing gatewayType or gatewayOutcomes")
        else:
            return model.find_next_task(
                context.bpmn_pos.id, workflow_node.data.label,
                max_distance=context.max_distance, match_threshold=0.8
            )

    def _merge_contexts(self, branch_results: list[TraversalContext], base_score: float = 0.0) -> TraversalContext:
        """Merge multiple branch contexts into one

        Args:
            branch_results: List of contexts from each branch
            base_score: Score accumulated before divergence (to avoid double-counting)
        """
        # Use the last branch's context as base (it completed the convergence)
        merged = branch_results[-1].clone()

        # Merge match scores, details, and scores from all branches
        merged.match_scores = []
        merged.match_details = []
        merged.total_score = base_score  # Start with base score
        merged.visited_nodes = set()

        for ctx in branch_results:
            merged.match_scores.extend(ctx.match_scores)
            merged.match_details.extend(ctx.match_details)
            # Add delta score (score added during branch traversal)
            merged.total_score += (ctx.total_score - base_score)
            merged.visited_nodes.update(ctx.visited_nodes)

        return merged

    def _get_node_by_id(self, node_id: str, workflow: WorkflowData) -> GraphNode:
        """Find workflow node by ID"""
        for node in workflow.nodes:
            if node.id == node_id:
                return node
        raise Exception(f"Node {node_id} not found")

    def _traverse_from(self, context: TraversalContext, model: Bpmn,
                       connectors: dict[str, ConnectorNode], workflow: WorkflowData) -> TraversalContext:
        """Recursively traverse workflow with branch handling"""

        while True:
            next_nodes = list(workflow.next(context.workflow_pos))

            if not next_nodes:
                print("No more workflow nodes to process")
                return context  # End of path

            # MULTIPLE NEXT NODES (divergence point)
            if len(next_nodes) != 1:
                print(f"\n--- Detected divergence with {len(next_nodes)} branches ---")
                return self._handle_divergence(context, next_nodes, connectors, model, workflow)

            # SINGLE NEXT NODE (linear flow)
            next_node = next_nodes[0]

            print(f"\n--- Processing workflow node {next_node.id} ---")
            print(f"Node label: '{next_node.data.label}'")
            print(f"Node type: {next_node.type}")

            # Handle connector nodes
            if next_node.id in connectors:
                connector = connectors[next_node.id]
                branch_id = f"{context.workflow_pos.id}_br"

                print(f"Reached connector: {connector.node_type} (visit {connector.visit_count + 1}/{connector.minimum_visit_count})")

                if connector.register_visit(branch_id):
                    # Convergence complete, continue past connector
                    print(f"Connector convergence complete, continuing...")
                    context.workflow_pos = next_node
                    continue
                else:
                    # Need more branches, pause here
                    print(f"Connector needs more branches, pausing this branch")
                    return context

            # Handle followedBy connectors
            elif next_node.data.relationshipType == "followedBy":
                print(f"Updating distance constraints: ideal={next_node.data.idealDistance}, max={next_node.data.maxDistance}")
                context.update_distance_constraints(next_node)
                context.workflow_pos = next_node
                continue

            # Handle element/gateway nodes
            elif next_node.data.checkType in ["element", "gateway"]:
                # Check if we're already at the target element (can happen after branch merges)
                # Normalize labels by removing whitespace and converting to lowercase
                bpmn_label_norm = " ".join(context.bpmn_pos.label.split()).lower()
                workflow_label_norm = " ".join(next_node.data.label.split()).lower()

                if bpmn_label_norm == workflow_label_norm:
                    print(f"Already at target element '{next_node.data.label}', skipping search")
                    # Still record this as a perfect match
                    context.match_scores.append(1.0)

                    # Create match detail for this perfect match
                    match_detail = MatchDetail(
                        workflow_node_id=next_node.id,
                        workflow_label=next_node.data.label,
                        bpmn_element_id=context.bpmn_pos.id,
                        bpmn_label=context.bpmn_pos.label,
                        match_score=1.0,
                        distance=0,  # Already at position
                        ideal_distance=context.ideal_distance,
                        max_distance=context.max_distance,
                        minimal_match_threshold=context.minimal_match_threshold,
                        ideal_match_threshold=context.ideal_match_threshold,
                        is_correct=True,
                        is_ideal_distance=True,  # Distance 0 is better than ideal
                        is_ideal_match=True  # Score 1.0 meets ideal threshold
                    )
                    context.match_details.append(match_detail)

                    context.workflow_pos = next_node
                    continue

                bpmn_result = self._find_bpmn_match(context, next_node, model)
                if not bpmn_result[1]:  # No match found
                    print(f"Problematic: Could not find BPMN element for '{next_node.data.label}'")

                    # Create a problematic match detail with score 0
                    match_detail = MatchDetail(
                        workflow_node_id=next_node.id,
                        workflow_label=next_node.data.label,
                        bpmn_element_id="NOT_FOUND",
                        bpmn_label="NOT_FOUND",
                        match_score=0.0,
                        distance=999,  # Invalid distance
                        ideal_distance=context.ideal_distance,
                        max_distance=context.max_distance,
                        minimal_match_threshold=context.minimal_match_threshold,
                        ideal_match_threshold=context.ideal_match_threshold,
                        is_correct=False,
                        is_ideal_distance=False,
                        is_ideal_match=False
                    )
                    context.match_details.append(match_detail)
                    context.match_scores.append(0.0)

                    # Don't award points for unmatched elements
                    # Continue to next node without raising exception
                    context.workflow_pos = next_node
                    continue

                visit_count, bpmn_elem, match_score = bpmn_result
                print(f"Found BPMN match '{bpmn_elem.label}' at distance {visit_count} with score {match_score:.3f}")

                context.apply_match_result(bpmn_result, next_node)
                context.workflow_pos = next_node
                continue

            else:
                # Unknown node type, move forward anyway
                print(f"Unknown node type, moving to next")
                context.workflow_pos = next_node
                continue


    def _handle_divergence(self, context: TraversalContext, branches: list[GraphNode],
                           connectors: dict[str, ConnectorNode], model: Bpmn,
                           workflow: WorkflowData) -> TraversalContext:
        """Handle gateway with multiple outgoing branches"""

        # Find convergence connector
        connector_id = self._find_convergence_point(branches, workflow)
        connector = connectors[connector_id]

        print(f"Divergence will converge at connector: {connector.node_type} (ID: {connector_id})")

        # Save BPMN state at divergence
        divergence_bpmn_state = context.bpmn_pos

        if connector.node_type == "AND":
            return self._handle_and_branches(context, branches, connector,
                                             divergence_bpmn_state, connectors, model, workflow)
        elif connector.node_type == "XOR":
            return self._handle_xor_branches(context, branches, connector,
                                              divergence_bpmn_state, connectors, model, workflow)
        else:
            raise Exception(f"Unknown connector type: {connector.node_type}")

    def _handle_and_branches(self, context: TraversalContext, branches: list[GraphNode],
                             connector: ConnectorNode, bpmn_state: PoolElement,
                             connectors: dict[str, ConnectorNode], model: Bpmn,
                             workflow: WorkflowData) -> TraversalContext:
        """All branches must reach connector"""

        print(f"\n=== Handling AND branches ({len(branches)} branches) ===")

        # Save the score before divergence to avoid double-counting
        base_score = context.total_score
        print(f"Score before divergence: {base_score}")

        branch_results = []

        for i, branch_start in enumerate(branches):
            print(f"\n--- Exploring AND branch {i + 1}/{len(branches)} ---")

            # Clone context for this branch
            branch_ctx = context.clone()
            branch_ctx.workflow_pos = branch_start
            branch_ctx.bpmn_pos = bpmn_state  # Reset to divergence point

            # Traverse this branch recursively
            result_ctx = self._traverse_from(branch_ctx, model, connectors, workflow)
            branch_results.append(result_ctx)

            print(f"Branch {i + 1} complete with {len(result_ctx.match_scores)} matches, score: {result_ctx.total_score}")

        # Merge results with base score to avoid double-counting
        print(f"\n--- Merging {len(branch_results)} AND branch results ---")
        merged_ctx = self._merge_contexts(branch_results, base_score)
        merged_ctx.workflow_pos = self._get_node_by_id(connector.node_id, workflow)
        print(f"Merged score: {merged_ctx.total_score}")

        # Continue past connector
        print(f"Continuing past AND connector...")
        return self._traverse_from(merged_ctx, model, connectors, workflow)

    def _handle_xor_branches(self, context: TraversalContext, branches: list[GraphNode],
                             connector: ConnectorNode, bpmn_state: PoolElement,
                             connectors: dict[str, ConnectorNode], model: Bpmn,
                             workflow: WorkflowData) -> TraversalContext:
        """At least one branch must succeed"""

        print(f"\n=== Handling XOR branches ({len(branches)} branches) ===")
        successful_results = []

        for i, branch_start in enumerate(branches):
            print(f"\n--- Trying XOR branch {i + 1}/{len(branches)} ---")
            try:
                # Clone context for this branch
                branch_ctx = context.clone()
                branch_ctx.workflow_pos = branch_start
                branch_ctx.bpmn_pos = bpmn_state  # Reset to divergence

                # Traverse this branch recursively
                result_ctx = self._traverse_from(branch_ctx, model, connectors, workflow)
                successful_results.append(result_ctx)

                print(f"XOR branch {i + 1} succeeded with confidence {result_ctx.confidence:.3f}")
            except Exception as e:
                # Branch failed, try next
                print(f"XOR branch {i + 1} failed: {e}")
                continue

        if not successful_results:
            raise Exception("XOR: No branches succeeded")

        # Pick best scoring branch
        best_ctx = max(successful_results, key=lambda ctx: ctx.confidence)
        print(f"\n--- Selecting best XOR branch (confidence: {best_ctx.confidence:.3f}) ---")
        best_ctx.workflow_pos = self._get_node_by_id(connector.node_id, workflow)

        # Continue past connector
        print(f"Continuing past XOR connector...")
        return self._traverse_from(best_ctx, model, connectors, workflow)

    def check_behavior(self, workflow: WorkflowData) -> BehavioralResult:
        """Analyze behavioral rules with AND/XOR connector support"""

        print("\n" + "=" * 80)
        print("BEHAVIORAL RULE CHECK WITH BRANCHING SUPPORT")
        print("=" * 80)

        # 1. Find starting workflow node
        workflow_start = _find_start_node(workflow.nodes, workflow.edges)
        if not workflow_start:
            raise Exception("Could not find root node in workflow")

        print(f"\nFound starting workflow node: {workflow_start.id}")
        print(f"Starting node label: '{workflow_start.data.label}'")

        # 2. Extract and map connectors
        connectors = self._extract_and_map_connectors(workflow)
        print(f"\nFound {len(connectors)} connector nodes:")
        for conn_id, conn in connectors.items():
            print(f"  - {conn.node_type} connector (ID: {conn_id}, min_visits: {conn.minimum_visit_count})")

        # 3. Parse BPMN model
        model = Bpmn(self.model_xml)

        # 4. Find starting BPMN element
        result = model.find_task(workflow_start.data.label, match_threshold=0.8)
        if not result:
            raise Exception("Could not find start node in BPMN model")
        bpmn_start, start_score = result

        print(f"\nFound starting BPMN element: '{bpmn_start.label}' (score: {start_score:.3f})")

        # 5. Create initial traversal context
        start_match_detail = MatchDetail(
            workflow_node_id=workflow_start.id,
            workflow_label=workflow_start.data.label,
            bpmn_element_id=bpmn_start.id,
            bpmn_label=bpmn_start.label,
            match_score=start_score,
            distance=0,  # Starting position
            ideal_distance=1,
            max_distance=2,
            minimal_match_threshold=0.6,
            ideal_match_threshold=0.8,
            is_correct=start_score >= 0.6,
            is_ideal_distance=True,
            is_ideal_match=start_score >= 0.8
        )

        initial_context = TraversalContext(
            workflow_pos=workflow_start,
            bpmn_pos=bpmn_start,
            match_scores=[start_score],
            match_details=[start_match_detail],
            total_score=0.0
        )

        # 6. Traverse workflow with branch support
        print("\n" + "=" * 80)
        print("STARTING TRAVERSAL")
        print("=" * 80)

        final_context = self._traverse_from(initial_context, model, connectors, workflow)

        # 7. Calculate results
        print("\n" + "=" * 80)
        print("TRAVERSAL COMPLETE")
        print("=" * 80)

        confidence = final_context.confidence
        total_matches = len(final_context.match_scores)

        # Round total_score to avoid floating point errors (e.g., 1.20000002 -> 1.2)
        total_score_rounded = round(final_context.total_score, 2)

        print(f"\nFinal Results:")
        print(f"  - Total matches: {total_matches}")
        print(f"  - Overall confidence: {confidence:.3f}")
        print(f"  - Total score: {total_score_rounded}")

        return BehavioralResult(
            id=self.id,
            name=self.name,
            category=self.algorithm_kind,
            description=self.description,
            fulfilled=True,
            confidence=confidence,
            problematic_elements=[],
            inputs=[],
            match_details=final_context.match_details,
            total_score=total_score_rounded,
            total_matches=total_matches
        )

    def analyze(self, inputs: list[AlgorithmFormInput] | None = None) -> AlgorithmResult:
        raise Exception("Not applicable to behavioral rule check")


class BehavioralGroupEvaluator:
    """Evaluates template groups with XOR/AND conditions and MAX scoring"""

    def __init__(self, model_xml: str):
        self.model_xml = model_xml

    def evaluate_group(self, group) -> GroupEvaluationResult:
        """
        Evaluate all templates in group and aggregate results

        Steps:
        1. Load all templates from group.template_ids
        2. Evaluate each using BehavioralRuleCheck.check_behavior()
        3. Collect individual TemplateEvaluationResult for each
        4. Apply aggregation based on condition (XOR/AND)
        5. Return GroupEvaluationResult with MAX score
        """
        import templates.manager

        template_manager = templates.manager.get_manager()
        checker = BehavioralRuleCheck(model_xml=self.model_xml)

        template_results = []
        for template_id in group.template_ids:
            template = template_manager.get_template(template_id)
            if template is None:
                # Template not found, treat as failed
                template_results.append(TemplateEvaluationResult(
                    template_id=template_id,
                    template_name=template_id,
                    description="Template not found",
                    score=0.0,
                    confidence=0.0,
                    match_details=[],
                    success=False
                ))
                continue

            workflow_data = WorkflowData(nodes=template.nodes, edges=template.edges)

            try:
                result = checker.check_behavior(workflow=workflow_data)
                template_results.append(TemplateEvaluationResult(
                    template_id=template_id,
                    template_name=template.name,
                    description=template.description,
                    score=result.total_score,
                    confidence=result.confidence,
                    match_details=result.match_details,
                    success=True
                ))
            except Exception as e:
                # Template failed to evaluate
                print(f"Template {template_id} failed: {e}")
                template_results.append(TemplateEvaluationResult(
                    template_id=template_id,
                    template_name=template.name,
                    description=template.description,
                    score=0.0,
                    confidence=0.0,
                    match_details=[],
                    success=False
                ))

        return self._aggregate_results(group, template_results)

    def _aggregate_results(self, group, template_results: list[TemplateEvaluationResult]) -> GroupEvaluationResult:
        """
        Aggregate template results based on condition

        XOR Logic (Alternative Solutions):
        - At least ONE template must succeed
        - Score = MAX(successful_template_scores)
        - fulfilled = any template succeeded
        - Use best template's match_details and confidence

        AND Logic (Required Features):
        - ALL templates must succeed
        - Score = MAX(all_template_scores) if all succeeded, else 0
        - fulfilled = all templates succeeded
        - Merge match_details from all templates
        """
        if group.condition.value == "XOR":
            successful = [r for r in template_results if r.success and r.score > 0]

            if successful:
                best = max(successful, key=lambda r: r.score)
                return GroupEvaluationResult(
                    group_id=group.group_id,
                    group_name=group.name,
                    group_description=group.description,
                    condition=group.condition.value,
                    template_results=template_results,
                    final_score=best.score,
                    best_template_id=best.template_id,
                    overall_confidence=best.confidence,
                    match_details=best.match_details,
                    problematic_elements=self._extract_problematic(best.match_details),
                    fulfilled=True
                )
            else:
                # No templates succeeded
                return GroupEvaluationResult(
                    group_id=group.group_id,
                    group_name=group.name,
                    group_description=group.description,
                    condition=group.condition.value,
                    template_results=template_results,
                    final_score=0.0,
                    best_template_id="",
                    overall_confidence=0.0,
                    match_details=[],
                    problematic_elements=[],
                    fulfilled=False
                )

        elif group.condition.value == "AND":
            all_succeeded = all(r.success for r in template_results)

            if all_succeeded:
                best = max(template_results, key=lambda r: r.score)
                # Merge match_details from all templates
                merged_details = []
                for r in template_results:
                    merged_details.extend(r.match_details)

                return GroupEvaluationResult(
                    group_id=group.group_id,
                    group_name=group.name,
                    group_description=group.description,
                    condition=group.condition.value,
                    template_results=template_results,
                    final_score=best.score,
                    best_template_id=best.template_id,
                    overall_confidence=best.confidence,
                    match_details=merged_details,
                    problematic_elements=self._extract_problematic(merged_details),
                    fulfilled=True
                )
            else:
                # Not all templates succeeded
                return GroupEvaluationResult(
                    group_id=group.group_id,
                    group_name=group.name,
                    group_description=group.description,
                    condition=group.condition.value,
                    template_results=template_results,
                    final_score=0.0,
                    best_template_id="",
                    overall_confidence=0.0,
                    match_details=[],
                    problematic_elements=[],
                    fulfilled=False
                )
        else:
            raise ValueError(f"Unknown condition: {group.condition}")

    def _extract_problematic(self, match_details: list[MatchDetail]) -> list[str]:
        """Extract BPMN IDs where matches were suboptimal"""
        problematic = []
        for match in match_details:
            if not match.is_correct or not match.is_ideal_match or not match.is_ideal_distance:
                if match.bpmn_element_id not in problematic:
                    problematic.append(match.bpmn_element_id)
        return problematic
