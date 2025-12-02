from typing import ClassVar

from algorithms import (
    Algorithm,
    AlgorithmComplexity,
    AlgorithmFormInput,
    AlgorithmInput,
    AlgorithmResult,
)
from bpmn.bpmn import Bpmn
from utils import get_elements_by_type
from utils.similarity import match_labels


class PoolLaneCheck(Algorithm):
    id: ClassVar[str] = "pool_lane_check"
    name: ClassVar[str] = "Pool-Lane Check"
    description: ClassVar[str] = (
        "Check for specific amount and label of the existing pools and lanes in a model"
    )
    algorithm_kind: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.CONFIGURABLE
    threshold: ClassVar[float] = 0.70

    def analyze(self, inputs: list[AlgorithmInput] | None = None) -> AlgorithmResult:
        if inputs is None:
            # Analyze pools & lanes whilst taking reference xml as ground truth.
            inputs = []
            model = Bpmn(self.model_xml)

            for pool in model.pools:
                inputs.append(
                    AlgorithmInput(
                        key=pool.name,
                        # In case you have a pool with a single lane then technically a
                        # lane exists that has no name hence the type check.
                        value=[
                            lane.name for lane in pool.lanes if lane.name is not None
                        ],
                    )
                )

            if len(inputs) == 0:
                raise Exception("No pools found")

            return AlgorithmResult(
                id=self.id,
                name=self.name,
                description=self.description,
                category=self.algorithm_kind,
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
                category=self.algorithm_kind,
                problematic_elements=[],
                fulfilled=None,
                inputs=inputs,
            )

        pool_matches = match_labels(
            target=submission_pools,
            reference=reference_pools,
            match_threshold=self.threshold,
        )
        if len(pool_matches) != len(reference_pools):
            # Add unmatched pool ids to problematic elements list and return early
            matched_pool_ids = []
            for submission_idx, best_reference_idx in pool_matches:
                matched_pool_ids.append(pools[submission_idx][1])

            missing_matches = set(
                [pool[1] for pool in pools if pool[0] is not None]
            ).difference(matched_pool_ids)

            # Also add lanes clarity sake
            for pool in model.pools:
                for lane in pool.lanes:
                    missing_matches.add(lane.id)

            return AlgorithmResult(
                id=self.id,
                name=self.name,
                description=self.description,
                category=self.algorithm_kind,
                problematic_elements=list(missing_matches),
                fulfilled=False,
                inputs=inputs,
            )

        missing_ids = []
        for submission_idx, reference_idx in pool_matches:
            submission_lane_labels = [
                task.name for task in model.pools[submission_idx].lanes
            ]
            reference_lane_labels = inputs[reference_idx].value

            if len(submission_lane_labels) != len(reference_lane_labels):
                for task in model.pools[submission_idx].lanes:
                    missing_ids.append(task.id)
                continue

            lane_pairs = match_labels(
                target=submission_lane_labels,
                reference=reference_lane_labels,
                match_threshold=self.threshold,
            )
            if len(lane_pairs) != len(reference_lane_labels):
                current_lane = model.pools[submission_idx].lanes
                # Add unmatched pool ids to problematic elements list and return early
                matched_lane_ids = []
                for submission_lane_idx, _ in lane_pairs:
                    matched_lane_ids.append(current_lane[submission_lane_idx].id)

                missing_matches = set(
                    [lane.id for lane in current_lane if not None]
                ).difference(matched_lane_ids)
                for missed_match in missing_matches:
                    missing_ids.append(missed_match)

        return AlgorithmResult(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.algorithm_kind,
            problematic_elements=missing_ids,
            fulfilled=(len(missing_ids) == 0),
            inputs=inputs,
        )

    def inputs(self) -> list[AlgorithmFormInput]:
        return [
            AlgorithmFormInput(
                input_label="Pools and lanes",
                input_type="key-value",
                key_label="Pool name",
                value_label="Lane name(s)",
                multiple=True,
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
