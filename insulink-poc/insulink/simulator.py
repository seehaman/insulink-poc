"""Synthetic CGM generator.

The simulated person has a hidden, PERSONAL sensitivity per food (for example
rice raises their glucose much more than pasta). The PoC must rediscover this
from the data alone, which is exactly the "personal food response" idea.

All data is synthetic and for demonstration only.
"""
import math
import random

from .models import FOODS, Meal, Reading

# Hidden ground truth: mg/dL peak rise per gram of carbohydrate, per food.
PERSONAL_SENSITIVITY = {
    "rice": 1.4, "bread": 0.85, "pasta": 0.5, "soft_drink": 1.7, "juice": 1.5,
    "beans": 0.6, "fruit": 0.8, "yogurt": 0.5, "vegetables": 0.3, "salad": 0.3,
    "chicken": 0.0, "eggs": 0.0, "fish": 0.0, "water": 0.0,
}

TEMPLATES = {
    "breakfast": [("bread", "eggs"), ("yogurt", "fruit"), ("bread", "juice", "eggs")],
    "lunch": [
        ("rice", "chicken", "vegetables"),
        ("rice", "chicken", "soft_drink"),
        ("pasta", "chicken", "salad"),
        ("beans", "bread", "salad"),
        ("fish", "vegetables", "water"),
    ],
    "dinner": [
        ("rice", "fish", "salad"),
        ("pasta", "chicken", "vegetables"),
        ("bread", "beans", "yogurt"),
        ("chicken", "salad", "water"),
        ("rice", "chicken", "soft_drink"),
    ],
}
MEAL_CLOCK = {"breakfast": (8 * 60, 20), "lunch": (13 * 60 + 30, 30), "dinner": (20 * 60, 30)}
EVENING_MULTIPLIER = 1.25
PEAK_MIN = 75
BASE = 105.0


def _bump(minutes_since_meal, amplitude):
    """Gamma-like response: zero at the meal, peak = amplitude at PEAK_MIN."""
    if minutes_since_meal <= 0 or minutes_since_meal > 400:
        return 0.0
    x = minutes_since_meal / PEAK_MIN
    return amplitude * x * x * math.exp(2.0 * (1.0 - x))


def generate(days=14, seed=7, inject_low_day=5):
    """Return (readings, meals). Readings every 5 minutes. Deterministic per seed."""
    rng = random.Random(seed)
    meals = []
    amplitudes = []
    for day in range(days):
        for kind, (centre, jitter) in MEAL_CLOCK.items():
            items = rng.choice(TEMPLATES[kind])
            portion = {i: rng.uniform(0.85, 1.15) for i in items}
            carbs = sum(FOODS[i][1] * portion[i] for i in items)
            amp = sum(FOODS[i][1] * portion[i] * PERSONAL_SENSITIVITY[i] for i in items)
            if kind == "dinner":
                amp *= EVENING_MULTIPLIER
            t = day * 1440 + centre + rng.randint(-jitter, jitter)
            meals.append(Meal(t=t, items=tuple(items), carbs=round(carbs, 1)))
            amplitudes.append(amp)

    dip_centre = inject_low_day * 1440 + 17 * 60 + 30 if inject_low_day is not None else None
    readings = []
    noise = 0.0
    for t in range(0, days * 1440, 5):
        noise = 0.9 * noise + rng.gauss(0, 1.5)
        value = BASE + 4.0 * math.sin(2 * math.pi * (t % 1440) / 1440) + noise
        for meal, amp in zip(meals, amplitudes):
            value += _bump(t - meal.t, amp)
        if dip_centre is not None:
            value -= 75.0 * math.exp(-(((t - dip_centre) / 25.0) ** 2))
        readings.append(Reading(t=t, mgdl=round(min(max(value, 40.0), 400.0), 1)))
    return readings, meals
