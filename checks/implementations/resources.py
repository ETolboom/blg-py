import logging
from typing import ClassVar

from checks import (
    Check,
    CheckComplexity,
    CheckFormInput,
    CheckInputType,
    CheckKeyValuePair,
    CheckKeyValueType,
    CheckResult,
)
from bpmn.bpmn import get_bpmn
from utils import get_elements_by_type
from utils.similarity import match_labels

logger = logging.getLogger(__name__)


class PoolLaneCheck(Check):
    id: ClassVar[str] = "pool_lane_check"
    name: ClassVar[str] = "Pool-Lane Check"
    description: ClassVar[str] = (
        "Check for specific amount and label of the existing pools and lanes in a model"
    )
    check_complexity: ClassVar[CheckComplexity] = CheckComplexity.CONFIGURABLE
    threshold: ClassVar[float] = 0.70

    key_label: ClassVar[str] = "Pool name"
    value_label: ClassVar[str] = "Lane name"
    input_scheme: ClassVar[list[CheckFormInput]] = [
            CheckFormInput(
                input_label="Pools and lanes",
                input_type=CheckInputType.KEY_VALUE,
                data=CheckKeyValueType(
                    key_label=key_label, value_label=value_label, pairs=[]
                ),
                multiple=True,
            ),
        ]

    def analyze(
        self, inputs: list[CheckFormInput] | None = None
    ) -> CheckResult:
        if inputs is None:
            # Analyze pools & lanes whilst taking reference xml as ground truth.
            inputs = []
            model = get_bpmn(self.model_xml)

            for pool in model.pools:
                inputs.append(
                    CheckFormInput(
                        input_label="Pool(s) and Lane(s)",
                        input_type=CheckInputType.KEY_VALUE,
                        multiple=True,
                        # In case you have a pool with a single lane then technically a
                        # lane exists that has no name hence the type check.
                        data=CheckKeyValueType(
                            key_label=self.key_label,
                            value_label=self.value_label,
                            pairs=[
                                CheckKeyValuePair(
                                    key=pool.name,
                                    value=[
                                        lane.name
                                        for lane in pool.lanes
                                        if lane.name is not None
                                    ],
                                )
                            ],
                        ),
                    )
                )

            if len(inputs) == 0:
                raise Exception("No pools found")

            return CheckResult(
                id=self.id,
                name=self.name,
                description=self.description,
                check_complexity=self.check_complexity,
                problematic_elements=[],
                fulfilled=True,
                inputs=inputs,
            )

        # Parse model_xml into Bpmn
        model = get_bpmn(self.model_xml)

        # Extract pools
        pools = [(pool.name, pool.id) for pool in model.pools]
        submission_pools = [pool[0] for pool in pools if pool[0] is not None]

        reference_pools: list[str] = []

        v: CheckFormInput
        for v in inputs:
            # TODO: Improper typing
            for pool in v.data.pairs:
                reference_pools.append(pool.key)

        if reference_pools and not submission_pools:
            return CheckResult(
                id=self.id,
                name=self.name,
                description=self.description,
                check_complexity=self.check_complexity,
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

            return CheckResult(
                id=self.id,
                name=self.name,
                description=self.description,
                check_complexity=self.check_complexity,
                problematic_elements=list(missing_matches),
                fulfilled=False,
                inputs=inputs,
            )

        missing_ids = []
        for submission_idx, reference_idx in pool_matches:
            submission_lane_labels = [
                task.name for task in model.pools[submission_idx].lanes
            ]

            # TODO: Unsafe access
            reference_lane_labels: list[str] = inputs[0].data.pairs[reference_idx].value

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
                    [lane.id for lane in current_lane if lane.id is not None]
                ).difference(matched_lane_ids)
                for missed_match in missing_matches:
                    missing_ids.append(missed_match)

        return CheckResult(
            id=self.id,
            name=self.name,
            description=self.description,
            check_complexity=self.check_complexity,
            problematic_elements=missing_ids,
            fulfilled=(len(missing_ids) == 0),
            inputs=inputs,
        )


    def is_applicable(self) -> bool:
        try:
            get_elements_by_type(self.model_xml, "process")
            get_elements_by_type(self.model_xml, "lane")
        except TypeError as e:
            logger.debug("Check '%s' is not applicable: %s", self.name, e)
            return False
        except ValueError as e:
            logger.debug("Check '%s' is not applicable: %s", self.name, e)
            return False

        return True
