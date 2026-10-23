"""Educational ICR/ISF calculator with hard limits.

EDUCATIONAL ONLY. This is not medical advice and never instructs anyone to dose.
Parameters must come from the user (set with their doctor); nothing is auto-filled.
"""
import math
from dataclasses import dataclass, field

DISCLAIMER = (
    "Educational estimate only. Not medical advice. Confirm every dose with your "
    "care plan or care team. Insulin on board is NOT considered."
)


class CalculatorError(ValueError):
    pass


@dataclass(frozen=True)
class DoseEstimate:
    meal_units: float
    correction_units: float
    total_units: float
    requires_confirmation: bool
    notes: tuple = field(default_factory=tuple)
    disclaimer: str = DISCLAIMER


def _round_step(x, step):
    return math.floor(x / step + 0.5) * step


def estimate(carbs_g, current_mgdl, icr, isf, rules):
    """Educational estimate: meal = carbs/ICR, correction = (BG - target)/ISF."""
    lim = rules["calculator"]
    if icr is None or isf is None:
        raise CalculatorError("ICR and ISF must be entered by the user; they are never auto-filled.")
    if not lim["icr_min"] <= icr <= lim["icr_max"]:
        raise CalculatorError(f"ICR {icr} outside allowed range {lim['icr_min']}-{lim['icr_max']} g/unit.")
    if not lim["isf_min"] <= isf <= lim["isf_max"]:
        raise CalculatorError(f"ISF {isf} outside allowed range {lim['isf_min']}-{lim['isf_max']} mg/dL/unit.")
    if not 0 <= carbs_g <= lim["max_carbs_g"]:
        raise CalculatorError(f"Carbs must be between 0 and {lim['max_carbs_g']} g.")
    if current_mgdl is None or current_mgdl <= 0:
        raise CalculatorError("A current glucose value is required.")

    if current_mgdl < rules["low_below"]:
        return DoseEstimate(
            0.0, 0.0, 0.0, True,
            ("Glucose is low. No dose is calculated. Follow your personal low-glucose plan first.",),
        )

    step = lim["round_to_units"]
    meal = _round_step(carbs_g / icr, step)
    over = current_mgdl - lim["target_mgdl"]
    correction = _round_step(over / isf, step) if over > 0 else 0.0
    total = meal + correction
    needs_confirm = total > lim["confirm_above_units"]
    notes = []
    if needs_confirm:
        notes.append(f"Total exceeds {lim['confirm_above_units']} units: confirm with your care team before acting.")
    if over <= 0:
        notes.append("Glucose is at or below target: no correction added.")
    return DoseEstimate(meal, correction, total, needs_confirm, tuple(notes))
