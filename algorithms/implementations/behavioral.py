from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from algorithms import (
    Algorithm,
    AlgorithmComplexity,
    AlgorithmFormInput,
    AlgorithmInput,
    AlgorithmResult,
)
from bpmn.bpmn import Bpmn
from utils.similarity import create_similarity_matrix


class GatewayCheck(Algorithm):
    id: ClassVar[str] = "gateway_check"
    name: ClassVar[str] = "Gateway Check"
    description: ClassVar[str] = (
        "Check if a gateway is mapped implemented according to the algorithm input"
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.CONFIGURABLE
    supported_gateway_types: ClassVar[list[str]] = [
        "exclusiveGateway",
        "parallelGateway",
    ]
    threshold: ClassVar[float] = 0.8

    def analyze(self, inputs: list[AlgorithmInput] | None) -> AlgorithmResult:
        # Parse the BPMN model from the input XML.
        model = Bpmn(self.model_xml)

        if not inputs:
            raise Exception("invalid input: input is missing")

        gateway_type = inputs[0].key
        reference_branches = [branch.split(",") for branch in inputs[0].value]

        all_gateways_in_model = find_gateways_with_branches(model)

        candidate_gateways = [
            gw for gw in all_gateways_in_model if gw.gateway_type == gateway_type
        ]

        if not candidate_gateways:
            # raise Exception(f"invalid input: no gateways found for type \"{gateway_type}\"")
            return AlgorithmResult(
                id=self.id,
                name=self.name,
                description=self.description,
                category=self.algorithm_kind,
                fulfilled=None,
                confidence=1.0,
                problematic_elements=[],
                inputs=inputs,
            )

        elements_by_id = {
            element.id: element for pool in model.pools for element in pool.elements
        }

        max_confidence = 0.0
        best_effort = None

        for gateway in candidate_gateways:
            # Extract the labels of tasks within each branch of the current gateway.
            actual_branches = []
            for branch in gateway.branches:
                branch_labels = []
                for _, element_id in branch:
                    if (
                        element_id in elements_by_id
                        and elements_by_id[element_id].label
                    ):
                        branch_labels.append(elements_by_id[element_id].label)
                actual_branches.append(branch_labels)

            if len(actual_branches) != len(reference_branches):
                continue

            num_branches = len(actual_branches)
            if num_branches == 0:
                branch_match_confidence = 1.0
            else:
                # Calculate a similarity matrix between actual and reference branches.
                branch_sim_matrix = np.zeros((num_branches, num_branches))
                for i, actual_labels in enumerate(actual_branches):
                    for j, reference_labels in enumerate(reference_branches):
                        if not actual_labels or not reference_labels:
                            branch_sim_matrix[i, j] = (
                                1.0
                                if not actual_labels and not reference_labels
                                else 0.0
                            )
                            continue

                        # Compute semantic similarity between task labels.
                        similarity_matrix = create_similarity_matrix(
                            actual_labels, reference_labels
                        )

                        # Calculate a symmetric similarity score for the branches.
                        score1 = torch.mean(
                            torch.max(similarity_matrix, dim=1).values
                        ).item()
                        score2 = torch.mean(
                            torch.max(similarity_matrix, dim=0).values
                        ).item()
                        branch_sim_matrix[i, j] = (score1 + score2) / 2.0

                # Find the optimal assignment to maximize total similarity.
                cost_matrix = 1 - branch_sim_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                total_similarity = branch_sim_matrix[row_ind, col_ind].sum()
                branch_match_confidence = total_similarity / num_branches

            if branch_match_confidence > max_confidence:
                max_confidence = branch_match_confidence
                best_effort = gateway

            if max_confidence >= self.threshold:
                return AlgorithmResult(
                    id=self.id,
                    name=self.name,
                    category=self.algorithm_kind,
                    description=self.description,
                    fulfilled=True,
                    confidence=max_confidence,
                    problematic_elements=[],
                    inputs=inputs,
                )

        problematic_elements = []
        if best_effort:
            for branch in best_effort.branches:
                for element in branch:
                    problematic_elements.append(element[1])

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_kind,
            fulfilled=False,
            confidence=max_confidence,
            problematic_elements=problematic_elements,
            inputs=inputs,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return [
            AlgorithmFormInput(
                input_label="Check a specific gateway",
                input_type="key-value",
                key_label="Gateway Type [exclusiveGateway, parallelGateway]",
                value_label="Task Labels [Label1,Label2,Label3]",
                multiple=True,
            )
        ]

    def is_applicable(self) -> bool:
        # Guessing what gateways becomes really messy
        return False


@dataclass
class Gateway:
    gateway_type: str
    element_before: tuple[str, str]
    element_after: tuple[str, str]
    branches: list[list[tuple[str, str]]]


def find_gateways_with_branches(bpmn: Bpmn) -> list[Gateway]:
    gateways = []
    for pool in bpmn.pools:
        elements_by_id = {elem.id: elem for elem in pool.elements}
        flows_by_source: dict[str, list[str]] = {}
        flows_by_target: dict[str, list[str]] = {}

        for flow in pool.flows:
            if flow.source not in flows_by_source:
                flows_by_source[flow.source] = []
            flows_by_source[flow.source].append(flow.target)

            if flow.target not in flows_by_target:
                flows_by_target[flow.target] = []
            flows_by_target[flow.target].append(flow.source)

        for element in pool.elements:
            if (
                element.name in ["exclusiveGateway", "parallelGateway"]
                and element.gateway_direction == "Diverging"
            ):
                element_before = None
                if element.id in flows_by_target and flows_by_target[element.id]:
                    before_elem = elements_by_id.get(flows_by_target[element.id][0])
                    if before_elem:
                        element_before = (before_elem.label, before_elem.id)

                branches = []
                converging_gateway_id = None
                element_after = None

                if element.id in flows_by_source:
                    for path_start in flows_by_source[element.id]:
                        branch, convergence_point = trace_branch(
                            path_start, elements_by_id, flows_by_source
                        )
                        if branch:
                            branches.append(branch)
                        if convergence_point and not converging_gateway_id:
                            converging_gateway_id = convergence_point

                if (
                    converging_gateway_id
                    and converging_gateway_id in flows_by_source
                    and flows_by_source[converging_gateway_id]
                ):
                    after_elem = elements_by_id.get(
                        flows_by_source[converging_gateway_id][0]
                    )
                    if after_elem:
                        element_after = (after_elem.label, after_elem.id)

                if element_before and branches:
                    gateways.append(
                        Gateway(
                            gateway_type=element.name,
                            element_before=element_before,
                            element_after=element_after or ("", ""),
                            branches=branches,
                        )
                    )
    return gateways


def trace_branch(
    start_id: str, elements_by_id: dict, flows_by_source: dict
) -> tuple[list[tuple[str, str]], str | None]:
    branch: list[tuple[str, str]] = []
    current_id = start_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        current_element = elements_by_id.get(current_id)

        if not current_element:
            break

        if (
            current_element.name in ["exclusiveGateway", "parallelGateway"]
            and current_element.gateway_direction == "Converging"
        ):
            return branch, current_id

        # Add the element's label and ID to the branch
        branch.append((current_element.label, current_element.id))

        if current_id in flows_by_source and flows_by_source[current_id]:
            current_id = flows_by_source[current_id][0]
        else:
            break

    return branch, None
