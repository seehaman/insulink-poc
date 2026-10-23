"""Shared data types and a tiny food catalogue (carbs per serving in grams)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    t: int          # minutes since start of the data set
    mgdl: float     # glucose in mg/dL


@dataclass(frozen=True)
class Meal:
    t: int                  # minutes since start of the data set
    items: tuple            # food names, e.g. ("rice", "chicken")
    carbs: float            # total carbohydrates in grams


# name -> (category, carbs per serving in grams). Illustrative values only.
FOODS = {
    "rice": ("starch", 45),
    "bread": ("starch", 30),
    "pasta": ("starch", 50),
    "soft_drink": ("sugary", 35),
    "juice": ("sugary", 25),
    "beans": ("legume", 25),
    "fruit": ("fruit", 20),
    "yogurt": ("dairy", 12),
    "vegetables": ("veg", 8),
    "salad": ("veg", 6),
    "chicken": ("protein", 0),
    "eggs": ("protein", 1),
    "fish": ("protein", 0),
    "water": ("none", 0),
}


def make_meal(t, items):
    """Build a Meal from food names using catalogue carbs."""
    carbs = sum(FOODS[i][1] for i in items)
    return Meal(t=t, items=tuple(items), carbs=float(carbs))


def fmt_time(t):
    """Minutes since start -> 'Day N HH:MM'."""
    day, minute = divmod(int(t), 1440)
    return f"Day {day + 1} {minute // 60:02d}:{minute % 60:02d}"
