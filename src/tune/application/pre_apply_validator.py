from __future__ import annotations

from dataclasses import dataclass

from tune.domain.hypothesis_models import CandidateParameter, TuningHypothesis


@dataclass(frozen=True)
class PreApplyValidationOutcome:
    allowed: bool
    reason: str


@dataclass
class PreApplyValidator:
    def validate(
        self,
        candidate: CandidateParameter,
        hypothesis: TuningHypothesis,
    ) -> PreApplyValidationOutcome:
        proposed_value = hypothesis.proposed_value
        if candidate.current_value is not None and proposed_value == candidate.current_value:
            return PreApplyValidationOutcome(
                allowed=False,
                reason=(
                    f"proposed value {proposed_value!r} is a no-op for "
                    f"{candidate.parameter_key}"
                ),
            )
        if proposed_value in candidate.forbidden_values:
            return PreApplyValidationOutcome(
                allowed=False,
                reason=(
                    f"proposed value {proposed_value!r} is forbidden for "
                    f"{candidate.parameter_key}"
                ),
            )
        if candidate.allowed_values and proposed_value not in candidate.allowed_values:
            return PreApplyValidationOutcome(
                allowed=False,
                reason=(
                    f"proposed value {proposed_value!r} is not in allowed values for "
                    f"{candidate.parameter_key}"
                ),
            )
        if candidate.min_value is not None or candidate.max_value is not None:
            try:
                numeric_value = int(proposed_value)
            except ValueError:
                return PreApplyValidationOutcome(
                    allowed=False,
                    reason=(
                        f"proposed value {proposed_value!r} is not an integer for "
                        f"{candidate.parameter_key}"
                    ),
                )
            if candidate.min_value is not None and numeric_value < candidate.min_value:
                return PreApplyValidationOutcome(
                    allowed=False,
                    reason=(
                        f"proposed value {numeric_value} is below minimum for "
                        f"{candidate.parameter_key}"
                    ),
                )
            if candidate.max_value is not None and numeric_value > candidate.max_value:
                return PreApplyValidationOutcome(
                    allowed=False,
                    reason=(
                        f"proposed value {numeric_value} is above maximum for "
                        f"{candidate.parameter_key}"
                    ),
                )
        return PreApplyValidationOutcome(allowed=True, reason="accepted")
