"""Rule-based safety engine. No AI, no learning: fixed thresholds from a config file."""
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "safety_rules.json"

MESSAGES = {
    "urgent_low": "URGENT LOW glucose. Follow your personal diabetes-care plan now.",
    "low": "Low glucose. Follow your personal diabetes-care plan.",
    "high": "High glucose.",
    "very_high": "VERY HIGH glucose. Consider contacting your care team.",
    "rapid_fall": "Glucose is falling rapidly.",
    "rapid_rise": "Glucose is rising rapidly.",
}


@dataclass(frozen=True)
class Alert:
    t: int
    kind: str
    value: float
    message: str


def load_rules(path=None):
    with open(path or DEFAULT_RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


def classify(value, rules):
    """Map one glucose value to a state. Boundaries: below the limit triggers."""
    if value < rules["urgent_low_below"]:
        return "urgent_low"
    if value < rules["low_below"]:
        return "low"
    if value > rules["very_high_above"]:
        return "very_high"
    if value > rules["high_above"]:
        return "high"
    return "in_range"


def rate_per_min(readings, i, window_min):
    """Average change in mg/dL per minute over the window ending at index i."""
    if i < 1:
        return 0.0
    step = readings[i].t - readings[i - 1].t
    back = max(1, int(round(window_min / step)))
    j = max(0, i - back)
    dt = readings[i].t - readings[j].t
    return 0.0 if dt == 0 else (readings[i].mgdl - readings[j].mgdl) / dt


def scan(readings, rules):
    """Walk a series and emit alerts on entering a worse state or a rapid change."""
    alerts = []
    prev_state = "in_range"
    last_fired = {}
    cooldown = rules["alert_cooldown_min"]

    def fire(kind, r):
        if kind in last_fired and r.t - last_fired[kind] < cooldown:
            return
        last_fired[kind] = r.t
        alerts.append(Alert(t=r.t, kind=kind, value=r.mgdl, message=MESSAGES[kind]))

    for i, r in enumerate(readings):
        state = classify(r.mgdl, rules)
        if state != prev_state and state != "in_range":
            fire(state, r)
        prev_state = state
        slope = rate_per_min(readings, i, rules["rate_window_min"])
        if slope <= rules["rapid_fall_per_min"]:
            fire("rapid_fall", r)
        elif slope >= rules["rapid_rise_per_min"]:
            fire("rapid_rise", r)
    return alerts


def time_in_range(readings, rules):
    """Fractions below / in / above the standard range (low_below to high_above)."""
    n = len(readings)
    if n == 0:
        return {"below": 0.0, "in_range": 0.0, "above": 0.0}
    below = sum(1 for r in readings if r.mgdl < rules["low_below"])
    above = sum(1 for r in readings if r.mgdl > rules["high_above"])
    return {"below": below / n, "in_range": (n - below - above) / n, "above": above / n}
