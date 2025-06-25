from typing import List

from utils.similarity import match_labels
from utils import get_elements_by_type
from algorithms import Algorithm, AlgorithmResult, AlgorithmFormInput, AlgorithmInput

from bpmn.bpmn import Bpmn

algorithm_category = "Resources"


class PoolLaneCheck(Algorithm):
    id = "pool_lane_check"
    name = "Pool-Lane Check"
    description = "Check for specific amount and label of the existing pools and lanes in a model"
    algorithm_type = algorithm_category
    threshold = 0.70

    def __init__(self, model_xml: str):
        super().__init__(model_xml)

    def analyze(self, inputs: List[AlgorithmInput] = None) -> AlgorithmResult:
        if inputs is None:
            # Analyze pools & lanes whilst taking reference xml as ground truth.
            inputs: List[AlgorithmInput] = []
            model = Bpmn(self.model_xml)

            for pool in model.pools:
                inputs.append(AlgorithmInput(
                    key=pool.name,
                    # In case you have a pool with a single lane then technically a lane exists that has no name
                    # hence the type check.
                    value=[lane.name for lane in pool.lanes if lane.name is not None],
                ))

            if len(inputs) == 0:
                raise Exception("No pools found")

            return AlgorithmResult(
                id=self.id,
                name=self.name,
                description=self.description,
                problematic_elements=[],
                fulfilled=True,
                inputs=inputs,
            )

        # Parse model_xml into Bpmn
        model = Bpmn(self.model_xml)

        # Extract pools
        pools = [(pool.name, pool.id) for pool in model.pools]
        submission_pools = [pool[0] for pool in pools if pool[0] is not None]

        reference_pools = [pool.key for pool in inputs]

        if reference_pools and not submission_pools:
            return AlgorithmResult(
                id=self.id,
                name=self.name,
                description=self.description,
                problematic_elements=[],
                fulfilled=None,
                inputs=inputs,
            )


        pool_matches = match_labels(target=submission_pools, reference=reference_pools, match_threshold=self.threshold)
        if len(pool_matches) != len(reference_pools):
            # Add unmatched pool ids to problematic elements list and return early
            matched_pool_ids = []
            for (submission_idx, best_reference_idx) in pool_matches:
                matched_pool_ids.append(pools[submission_idx][1])

            missing_matches = set([pool[1] for pool in pools if pool[0] is not None]).difference(matched_pool_ids)

            # Also add lanes clarity sake
            for pool in model.pools:
                for lane in pool.lanes:
                    missing_matches.add(lane.id)

            return AlgorithmResult(
                id=self.id,
                name=self.name,
                description=self.description,
                problematic_elements=list(missing_matches),
                fulfilled=False,
                inputs=inputs,
            )

        missing_ids = []
        for (submission_idx, reference_idx) in pool_matches:
            submission_lane_labels = [task.name for task in model.pools[submission_idx].lanes]
            reference_lane_labels = inputs[reference_idx].value

            if len(submission_lane_labels) != len(reference_lane_labels):
                for task in model.pools[submission_idx].lanes:
                    missing_ids.append(task.id)
                continue

            lane_pairs = match_labels(target=submission_lane_labels, reference=reference_lane_labels,
                                      match_threshold=self.threshold)
            if len(lane_pairs) != len(reference_lane_labels):
                current_lane = model.pools[submission_idx].lanes
                # Add unmatched pool ids to problematic elements list and return early
                matched_lane_ids = []
                for (submission_lane_idx, _) in lane_pairs:
                    matched_lane_ids.append(current_lane[submission_lane_idx].id)

                missing_matches = set([lane.id for lane in current_lane if not None]).difference(matched_lane_ids)
                for missed_match in missing_matches:
                    missing_ids.append(missed_match)

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            problematic_elements=missing_ids,
            fulfilled=(len(missing_ids) == 0),
            inputs=inputs,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return [
            AlgorithmFormInput(
                input_label="Pools and lanes",
                input_type="key-value",
                key_label="Pool name",
                value_label="Lane name(s)",
                multiple=True
            ),
        ]

    def is_applicable(self) -> bool:
        try:
            get_elements_by_type(self.model_xml, "process")
            get_elements_by_type(self.model_xml, "lane")
        except TypeError as e:
            print(f"Algorithm {self.name} is not applicable: {e}")
            return False
        except ValueError as e:
            print(f"Algorithm {self.name} is not applicable: {e}")
            return False

        return True


class ResourceCheck(Algorithm):
    id = "resource_check"
    name = "Check for use of a specific resource"
    description = "Check the model for the use of a specific element such as a data store."
    algorithm_type = algorithm_category

    def __init__(self, model_xml: str):
        super().__init__(model_xml)

    def analyze(self, inputs: List[AlgorithmInput] = None) -> AlgorithmResult:
        if inputs is None:
            raise Exception("Invalid input: missing")

        if (len(inputs) != 1) or (len(inputs[0].value) != 1):
            raise Exception("Invalid input: more than one input provided")

        element_type = inputs[0].value[0]

        fulfilled = True

        problematic_elements = []
        try:
            elements = get_elements_by_type(self.model_xml, element_type)

            # We flag them as "problematic" elements so that we can highlight them.
            problematic_elements = [element[1] for element in elements]
        except TypeError as e:
            print(f"Algorithm {self.name} is not applicable: {e}")
            fulfilled = False
        except ValueError as e:
            print(f"Algorithm {self.name} is not applicable: {e}")
            fulfilled = False

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            fulfilled=fulfilled,
            problematic_elements=problematic_elements,
        )

    def inputs(self) -> List[AlgorithmFormInput]:
        return [
            AlgorithmFormInput(
                input_label="Check for a specific resource",
                input_type="string",
                key_label="Element/Resource name",
                value_label="Element Name [e.g., dataStoreReference]",
                multiple=False
            )
        ]

    def is_applicable(self) -> bool:
        # We cannot guess what the user wants to test for
        return False
