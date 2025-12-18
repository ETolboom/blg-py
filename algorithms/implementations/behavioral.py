from typing import ClassVar, Optional

from algorithms import Algorithm, AlgorithmComplexity, AlgorithmFormInput, AlgorithmResult
from pydantic import BaseModel


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

def process_edges(edges: list[Edge], nodes: list[GraphNode]) -> list[Edge]:
    # Create node lookup
    node_map = {node.id: node for node in nodes}

    processed_edges = []
    for edge in edges:
        # Check if this is a gateway edge (sourceHandle like "outcome-0")
        if edge.sourceHandle and edge.sourceHandle.startswith("outcome-"):
            source_node = node_map.get(edge.source)
            if source_node and source_node.data.gatewayOutcomes:
                # Extract index from sourceHandle (e.g., "outcome-0" -> 0)
                try:
                    outcome_idx = int(edge.sourceHandle.split("-")[1])
                    if 0 <= outcome_idx < len(source_node.data.gatewayOutcomes):
                        edge.outcome = source_node.data.gatewayOutcomes[outcome_idx]
                except (IndexError, ValueError):
                    pass

        processed_edges.append(edge)

    return processed_edges

def build_decision_tree(nodes: list[GraphNode], edges: list[Edge]) -> Optional[DecisionTreeNode]:
    # Create a mapping of node_id to GraphNode
    node_map = {node.id: node for node in nodes}

    # Create edge lookup structures
    # outgoing_edges: source_node_id -> list of edges
    outgoing_edges: dict[str, list[Edge]] = {}
    for edge in edges:
        if edge.source not in outgoing_edges:
            outgoing_edges[edge.source] = []
        outgoing_edges[edge.source].append(edge)

    # Find start node (first element node or node with no incoming connections)
    # For now, we'll use the first elementCheck node as the start
    start_node = None
    for node in nodes:
        if node.type == "elementCheck":
            start_node = node
            break

    if not start_node:
        return None

    # Build tree recursively
    visited = set()
    return build_tree_recursive(start_node, node_map, outgoing_edges, visited)

def build_tree_recursive(
    graph_node: GraphNode,
    node_map: dict[str, GraphNode],
    outgoing_edges: dict[str, list[Edge]],
    visited: set[str]
) -> Optional[DecisionTreeNode]:
    """Recursively build decision tree from graph nodes"""

    # Prevent infinite loops
    if graph_node.id in visited:
        return None
    visited.add(graph_node.id)

    # Skip connector nodes
    if graph_node.type in ["followedByConnector", "andConnector"]:
        # Follow through to the next node
        next_node = find_next_node(graph_node.id, node_map, outgoing_edges)
        if next_node:
            return build_tree_recursive(next_node, node_map, outgoing_edges, visited)
        return None

    # Create decision tree node based on graph node type
    if graph_node.type == "gatewayCheck":
        # Gateway node - decision point
        tree_node = DecisionTreeNode(
            node_id=graph_node.id,
            node_type="gateway",
            label=graph_node.data.label,
            score=graph_node.data.score or 0.0,
            outcomes=graph_node.data.gatewayOutcomes,
            children={}
        )

        # For each outcome, find the connected node
        if graph_node.data.gatewayOutcomes:
            for outcome in graph_node.data.gatewayOutcomes:
                # Find the edge with this outcome
                next_node = find_next_node_for_outcome(
                    graph_node.id,
                    outcome,
                    node_map,
                    outgoing_edges
                )
                if next_node:
                    child = build_tree_recursive(
                        next_node,
                        node_map,
                        outgoing_edges,
                        visited.copy()
                    )
                    if child:
                        tree_node.children[outcome] = child

        return tree_node

    elif graph_node.type == "elementCheck":
        # Element node - task or event
        tree_node = DecisionTreeNode(
            node_id=graph_node.id,
            node_type=graph_node.data.elementType or "element",
            label=graph_node.data.label,
            score=graph_node.data.score or 0.0
        )

        # Find next node in sequence
        next_node = find_next_node(graph_node.id, node_map, outgoing_edges)
        if next_node:
            tree_node.next_node = build_tree_recursive(
                next_node,
                node_map,
                outgoing_edges,
                visited.copy()
            )

        return tree_node

    elif graph_node.type == "endNode":
        # End node
        return DecisionTreeNode(
            node_id=graph_node.id,
            node_type="end",
            label=graph_node.data.label,
            score=0.0
        )

    return None

def find_next_node(
    node_id: str,
    node_map: dict[str, GraphNode],
    outgoing_edges: dict[str, list[Edge]]
) -> Optional[GraphNode]:
    """Find the next node connected to this node"""
    if node_id not in outgoing_edges:
        return None

    # Get the first outgoing edge (for non-gateway nodes)
    edges = outgoing_edges[node_id]
    if not edges:
        return None

    target_id = edges[0].target
    return node_map.get(target_id)

def find_next_node_for_outcome(
    gateway_id: str,
    outcome: str,
    node_map: dict[str, GraphNode],
    outgoing_edges: dict[str, list[Edge]]
) -> Optional[GraphNode]:
    """Find the next node connected to a specific gateway outcome"""
    if gateway_id not in outgoing_edges:
        return None

    # Find the edge with the matching outcome
    for edge in outgoing_edges[gateway_id]:
        if edge.outcome == outcome:
            target_id = edge.target
            return node_map.get(target_id)

    return None


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

    def check_behavior(self, workflow: WorkflowData) -> AlgorithmResult:
        """Analyze the behavioral rules using decision tree"""
        # Process edges to compute outcome names for gateway edges
        workflow.edges = process_edges(workflow.edges, workflow.nodes)

        # Build decision tree
        tree = build_decision_tree(workflow.nodes, workflow.edges)

        if tree is None:
            return AlgorithmResult(
                id=self.id,
                name=self.name,
                category=self.algorithm_kind,
                description=self.description,
                fulfilled=False,
                confidence=0.0,
                problematic_elements=["Failed to build decision tree"],
                inputs=[],
            )

        # Analyze the tree
        fulfilled, problematic, confidence = self.analyze_tree(tree)

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            category=self.algorithm_kind,
            description=self.description,
            fulfilled=fulfilled,
            confidence=confidence,
            problematic_elements=problematic,
            inputs=[],
        )
    def analyze(self, inputs: list[AlgorithmFormInput] | None = None) -> AlgorithmResult:
        raise Exception("Not applicable to behavioral rule check")
