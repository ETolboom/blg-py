import json
from algorithms.implementations.behavioral import BehavioralRuleCheck, WorkflowData, process_edges, build_decision_tree


def print_tree(node, prefix="", is_last=True, outcome=None):
    if outcome:
        connector = "└─" if is_last else "├─"
        print(f"{prefix}{connector} [{outcome}]")
        extension = "   " if is_last else "│  "
        prefix = prefix + extension

    node_type_label = {
        "task": "[TASK]",
        "event": "[EVENT]",
        "gateway": "[GATEWAY]",
        "end": "[END]"
    }.get(node.node_type, "[UNKNOWN]")

    score_indicator = f"  (score: {node.score:0.01})" if node.score > 0 else ""

    # For nodes without outcome (root or sequential), print without connector
    if outcome:
        print(f"{prefix}{node.label} {node_type_label}{score_indicator}")
    else:
        print(f"{prefix}{node.label} {node_type_label}{score_indicator}")

    if node.children:
        outcomes = list(node.children.items())
        for i, (outcome_name, child) in enumerate(outcomes):
            is_last_child = (i == len(outcomes) - 1)
            print_tree(child, prefix, is_last_child, outcome_name)
    elif node.next_node:
        # For sequential nodes, continue with a vertical line
        print(f"{prefix}│")
        print_tree(node.next_node, prefix, True, None)


def main():
    # Load nodes.json
    with open("nodes.json", "r") as f:
        data = json.load(f)

    workflow = WorkflowData(**data)

    # Create checker instance
    checker = BehavioralRuleCheck(model_xml="<dummy/>")

    workflow.edges = process_edges(workflow.edges, workflow.nodes)

    tree = build_decision_tree(workflow.nodes, workflow.edges)

    if not tree:
        print("Failed to build decision tree")
        return

    print("\n" + "-" * 70)
    print()

    print_tree(tree)

    print("\n" + "-" * 70)

    result = checker.check_behavior(workflow)

    print(f"\nAnalysis Results:")
    print(f"Template name: {result.name}")
    print(f"Fulfilled: {result.fulfilled}")
    print(f"Confidence: {result.confidence:.1%}")

    if result.problematic_elements:
        print(f"\nProblematic Elements ({len(result.problematic_elements)}):")
        for elem_id in result.problematic_elements:
            node_label = next(
                (n.data.label for n in workflow.nodes if n.id == elem_id),
                elem_id
            )
            print(f"  - {elem_id}: {node_label}")
    else:
        print("\nNo problematic elements found")


if __name__ == "__main__":
    main()
