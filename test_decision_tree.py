import json
from algorithms.implementations.behavioral import BehavioralRuleCheck, WorkflowData

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
    with open("example/submissions/1BM05-2-21-ExamQ4 Solution.bpmn") as f:
        model_xml = f.read()

    checker = BehavioralRuleCheck(model_xml=model_xml)

    result = checker.check_behavior(workflow)

    # Display match details
    print("\n" + "=" * 80)
    print("MATCH DETAILS")
    print("=" * 80)
    print(f"\nTotal matches: {result.total_matches}")
    print(f"Overall confidence: {result.confidence:.1%}")
    print(f"Total penalty score: {result.total_score}")

    # Calculate statistics
    correct_matches = sum(1 for m in result.match_details if m.is_correct)
    ideal_distance_matches = sum(1 for m in result.match_details if m.is_ideal_distance)
    non_ideal_matches = [m for m in result.match_details if not m.is_ideal_distance]

    print(f"\nStatistics:")
    print(f"  - Correct matches: {correct_matches}/{result.total_matches} ({correct_matches/result.total_matches:.1%})")
    print(f"  - Ideal distance matches: {ideal_distance_matches}/{result.total_matches} ({ideal_distance_matches/result.total_matches:.1%})")
    print(f"  - Non-ideal distance matches: {len(non_ideal_matches)}")

    if non_ideal_matches:
        print(f"\nNon-ideal distance matches:")
        for match in non_ideal_matches:
            print(f"  - '{match.workflow_label}': distance={match.distance}, ideal={match.ideal_distance}")

    print(f"\nIndividual matches:")
    print("-" * 80)

    for i, match in enumerate(result.match_details, 1):
        status_icon = "✓" if match.is_correct else "✗"
        distance_status = "ideal" if match.is_ideal_distance else f"dist={match.distance}"

        print(f"\n{i}. {status_icon} {match.workflow_label}")
        print(f"   Workflow ID: {match.workflow_node_id}")
        print(f"   BPMN Match: '{match.bpmn_label}' (ID: {match.bpmn_element_id})")
        print(f"   Match Score: {match.match_score:.3f}")
        print(f"   Distance: {match.distance} (ideal={match.ideal_distance}, max={match.max_distance}) [{distance_status}]")
        print(f"   Correct: {match.is_correct}")

    # if not tree:
    #     print("Failed to build decision tree")
    #     return
    #
    # print("\n" + "-" * 70)
    # print()
    #
    # print_tree(tree)
    #
    # print("\n" + "-" * 70)
    #
    # result = checker.check_behavior(workflow)
    #
    # print(f"\nAnalysis Results:")
    # print(f"Template name: {result.name}")
    # print(f"Fulfilled: {result.fulfilled}")
    # print(f"Confidence: {result.confidence:.1%}")
    #
    # if result.problematic_elements:
    #     print(f"\nProblematic Elements ({len(result.problematic_elements)}):")
    #     for elem_id in result.problematic_elements:
    #         node_label = next(
    #             (n.data.label for n in workflow.nodes if n.id == elem_id),
    #             elem_id
    #         )
    #         print(f"  - {elem_id}: {node_label}")
    # else:
    #     print("\nNo problematic elements found")


if __name__ == "__main__":
    main()
