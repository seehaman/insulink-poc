"""Personal food response engine (statistics + nearest neighbours, fully explainable)."""
from dataclasses import dataclass

from .models import fmt_time


@dataclass(frozen=True)
class MealResponse:
    meal: object
    baseline: float
    peak: float
    rise: float
    time_to_peak_min: int
    iauc: float          # incremental area above baseline, mg/dL * min, over the window


@dataclass(frozen=True)
class Prediction:
    rise: float
    neighbors: tuple     # ((MealResponse, similarity), ...)


def measure(meal, by_t, next_meal_t=None, window_min=180, pre_min=15, step=5):
    """Measure one meal's response. Returns None if there is not enough clean data."""
    start = step * round(meal.t / step)          # snap meal time to the reading grid
    pre = [by_t[t] for t in range(start - pre_min, start + 1, step) if t in by_t]
    if not pre:
        return None
    baseline = sum(pre) / len(pre)
    end = start + window_min
    if next_meal_t is not None:
        end = min(end, step * round(next_meal_t / step))
    if end - start < 90:
        return None
    pts = [(t, by_t[t]) for t in range(start, end + 1, step) if t in by_t]
    if len(pts) < 10:
        return None
    peak_t, peak = max(pts, key=lambda p: p[1])
    iauc = 0.0
    for (t1, v1), (t2, v2) in zip(pts, pts[1:]):
        iauc += ((max(v1 - baseline, 0.0) + max(v2 - baseline, 0.0)) / 2.0) * (t2 - t1)
    return MealResponse(meal, baseline, peak, peak - baseline, peak_t - start, iauc)


def measure_all(meals, readings):
    by_t = {r.t: r.mgdl for r in readings}
    ordered = sorted(meals, key=lambda m: m.t)
    out = []
    for i, m in enumerate(ordered):
        nxt = ordered[i + 1].t if i + 1 < len(ordered) else None
        resp = measure(m, by_t, nxt)
        if resp is not None:
            out.append(resp)
    return out


def similarity(a, b):
    """0..1. Shared foods (60%) plus closeness of carbohydrate load (40%)."""
    sa, sb = set(a.items), set(b.items)
    jaccard = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    top = max(a.carbs, b.carbs)
    closeness = 1.0 if top == 0 else 1.0 - abs(a.carbs - b.carbs) / top
    return 0.6 * jaccard + 0.4 * closeness


def predict_rise(meal, history, k=3, sim=None):
    """Predict the glucose rise for a (possibly hypothetical) meal from past meals.

    By default `meal` is a Meal and past meals are compared with `similarity`
    (shared foods + carbohydrate load). Data sets without food names pass their
    own `sim(query, past_response)`; `meal` is then whatever query that function
    expects (for HUPA-UCM, a MealResponse so starting glucose can be compared).
    """
    if not history:
        return Prediction(0.0, ())
    if sim is None:
        scored = ((h, similarity(meal, h.meal)) for h in history)
    else:
        scored = ((h, sim(meal, h)) for h in history)
    scored = sorted(scored, key=lambda p: -p[1])[:k]
    total = sum(s for _, s in scored)
    if total == 0:
        rise = sum(h.rise for h in history) / len(history)
    else:
        rise = sum(h.rise * s for h, s in scored) / total
    return Prediction(rise, tuple(scored))


def explain(pred):
    """Human-readable evidence behind a prediction."""
    lines = [f"Estimated rise: about +{pred.rise:.0f} mg/dL. Based on your {len(pred.neighbors)} most similar past meals:"]
    for h, s in pred.neighbors:
        lines.append(
            f"  - {fmt_time(h.meal.t)}: {' + '.join(h.meal.items)} "
            f"({h.meal.carbs:.0f} g carbs) -> +{h.rise:.0f} mg/dL (similarity {s:.2f})"
        )
    return "\n".join(lines)


def rise_by_food(responses):
    """Average rise of meals containing each food. Rough, confounded by co-eaten foods."""
    acc = {}
    for r in responses:
        for item in r.meal.items:
            acc.setdefault(item, []).append(r.rise)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def evaluate(train, test, k=3):
    """Mean absolute error (mg/dL) of three predictors on held-out meals."""
    if not train or not test:
        raise ValueError("need both train and test responses")
    global_mean = sum(r.rise for r in train) / len(train)
    per_gram = sum(r.rise / r.meal.carbs for r in train if r.meal.carbs > 0) / max(
        1, sum(1 for r in train if r.meal.carbs > 0)
    )
    err_knn = err_global = err_carb = 0.0
    for r in test:
        err_knn += abs(predict_rise(r.meal, train, k).rise - r.rise)
        err_global += abs(global_mean - r.rise)
        err_carb += abs(per_gram * r.meal.carbs - r.rise)
    n = len(test)
    return {
        "n_train": len(train),
        "n_test": n,
        "mae_personal_knn": err_knn / n,
        "mae_global_mean": err_global / n,
        "mae_carb_ratio": err_carb / n,
    }
