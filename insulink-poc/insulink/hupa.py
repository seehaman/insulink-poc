"""HUPA-UCM real-data adapter: load, clean and evaluate with the same engine.

HUPA-UCM (Hidalgo et al., Mendeley Data, 2024, CC BY 4.0) holds 25 people with
type 1 diabetes: FreeStyle Libre 2 glucose every 5 minutes, insulin, carbohydrate
entries and Fitbit activity. We read the files in `Preprocessed/` (one CSV per
patient, `;`-separated). The data is not stored in this repository.

Everything here is pure Python. The food-response predictor is the same
`food_response.predict_rise`; only the similarity function changes, because
HUPA-UCM records carbohydrates but no food names.
"""
import csv
import glob
import os
import statistics
from dataclasses import dataclass
from datetime import datetime

from . import food_response as fr
from .models import Meal, Reading

STEP_MIN = 5
WINDOW_MIN = 120          # response window after a meal
PRE_MIN = 15              # baseline window before a meal
EXCHANGE_GRAMS = 10.0     # one carbohydrate exchange
EXCHANGE_MEDIAN_MAX = 12  # a patient whose median entry is <= this logs exchanges

# Similarity weights for meals without food names. Fixed before testing.
W_CARBS, W_TIME, W_START = 0.5, 0.25, 0.25
CARB_SCALE_G, TIME_SCALE_H, START_SCALE_MGDL = 60.0, 6.0, 100.0


@dataclass(frozen=True)
class Patient:
    pid: str
    start: datetime        # midnight of the first day; Reading.t and Meal.t count from here
    readings: tuple        # Reading, every 5 minutes
    meals: tuple           # Meal with items=() and carbs in grams
    carb_factor: float     # 10.0 if the patient logged exchanges, else 1.0


def carb_factor(entries):
    """Grams per logged unit. The documentation says grams, but most patients
    log 10 g exchanges (typical meal entries of 1 to 7)."""
    if not entries:
        return 1.0
    return EXCHANGE_GRAMS if statistics.median(entries) <= EXCHANGE_MEDIAN_MAX else 1.0


def load_patient(path):
    pid = os.path.basename(path)[4:8]
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append((datetime.fromisoformat(r["time"]), float(r["glucose"]), float(r["carb_input"])))
    if not rows:
        raise ValueError(f"{path}: no rows")
    first = rows[0][0]
    start = first.replace(hour=0, minute=0, second=0, microsecond=0)

    def minutes(ts):
        return int((ts - start).total_seconds() // 60)

    factor = carb_factor([c for _, _, c in rows if c > 0])
    readings = tuple(Reading(minutes(ts), g) for ts, g, _ in rows)
    meals = tuple(Meal(minutes(ts), (), c * factor) for ts, _, c in rows if c > 0)
    return Patient(pid, start, readings, meals, factor)


def load_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "HUPA*.csv")))
    if not files:
        raise FileNotFoundError(f"No HUPA*.csv files in {folder!r}. Point to the dataset's Preprocessed folder.")
    return [load_patient(f) for f in files]


def measure_isolated(patient):
    """MealResponse for every meal with no other carb entry in the next 2 hours,
    at least 2 readings in the 15 minutes before and 24 of 25 readings in the
    2 hours after. Rise = highest glucose in the window minus the baseline."""
    by_t = {r.t: r.mgdl for r in patient.readings}
    meals = sorted(patient.meals, key=lambda m: m.t)
    out = []
    for i, m in enumerate(meals):
        if i + 1 < len(meals) and meals[i + 1].t - m.t < WINDOW_MIN:
            continue
        pre = [by_t[t] for t in range(m.t - PRE_MIN, m.t + 1, STEP_MIN) if t in by_t]
        pts = [(t, by_t[t]) for t in range(m.t, m.t + WINDOW_MIN + 1, STEP_MIN) if t in by_t]
        if len(pre) < 2 or len(pts) < WINDOW_MIN // STEP_MIN:
            continue
        baseline = sum(pre) / len(pre)
        peak_t, peak = max(pts, key=lambda p: p[1])
        iauc = sum(((max(v1 - baseline, 0) + max(v2 - baseline, 0)) / 2) * (t2 - t1)
                   for (t1, v1), (t2, v2) in zip(pts, pts[1:]))
        out.append(fr.MealResponse(m, baseline, peak, peak - baseline, peak_t - m.t, iauc))
    return out


def context_similarity(query, past):
    """0..1 between two MealResponses: carbohydrate load (50%), time of day (25%)
    and starting glucose (25%). `query` only needs .meal and .baseline, so a
    hypothetical meal at the current glucose works too."""
    a, b = query.meal, past.meal
    s_carbs = 1 - min(abs(a.carbs - b.carbs) / CARB_SCALE_G, 1)
    dh = abs((a.t % 1440) - (b.t % 1440)) / 60
    dh = min(dh, 24 - dh)
    s_time = 1 - min(dh / TIME_SCALE_H, 1)
    s_start = 1 - min(abs(query.baseline - past.baseline) / START_SCALE_MGDL, 1)
    return W_CARBS * s_carbs + W_TIME * s_time + W_START * s_start


def split(responses, train_frac=0.7):
    """Time-ordered split: the model never sees meals from the future."""
    ordered = sorted(responses, key=lambda r: r.meal.t)
    cut = int(len(ordered) * train_frac)
    return ordered[:cut], ordered[cut:]


def carb_ratio(train):
    """Patient's own average rise per gram (ratio of totals)."""
    grams = sum(r.meal.carbs for r in train)
    return sum(r.rise for r in train) / grams if grams else 0.0


def errors(train, test, global_mean, k=3):
    """Mean absolute error of the three predictors on one patient's test meals."""
    ratio = carb_ratio(train)
    e = {"personal": 0.0, "ratio": 0.0, "global": 0.0}
    for r in test:
        e["personal"] += abs(fr.predict_rise(r, train, k, sim=context_similarity).rise - r.rise)
        e["ratio"] += abs(ratio * r.meal.carbs - r.rise)
        e["global"] += abs(global_mean - r.rise)
    return {key: v / len(test) for key, v in e.items()}


def evaluate(patients, k=3, train_frac=0.7, min_meals=10):
    """Per-patient MAE plus the average over patients (each patient counts once)."""
    splits = {}
    for p in patients:
        responses = measure_isolated(p)
        if len(responses) >= min_meals:
            splits[p.pid] = split(responses, train_frac)
    if not splits:
        raise ValueError("no patient has enough isolated meals")
    all_train = [r for tr, _ in splits.values() for r in tr]
    global_mean = sum(r.rise for r in all_train) / len(all_train)
    per_patient = {}
    for pid, (tr, te) in splits.items():
        per_patient[pid] = dict(errors(tr, te, global_mean, k), n_train=len(tr), n_test=len(te))
    summary = {key: sum(v[key] for v in per_patient.values()) / len(per_patient)
               for key in ("personal", "ratio", "global")}
    summary["personal_best"] = sum(1 for v in per_patient.values()
                                   if v["personal"] < min(v["ratio"], v["global"]))
    return {"global_mean": global_mean, "splits": splits, "per_patient": per_patient, "summary": summary}


def learning_curve(train, test, global_mean, sizes=(10, 20, 30, 60, 120), k=3):
    """Error on fixed test meals when the model only sees the most recent n learning meals."""
    out = []
    for n in sizes:
        if n <= len(train):
            out.append((n, errors(train[-n:], test, global_mean, k)))
    return out


def low_episodes(readings, below=54, min_minutes=15):
    """Count runs of consecutive readings under `below` lasting at least `min_minutes`."""
    need = max(1, min_minutes // STEP_MIN)
    count = run = 0
    for r in readings:
        if r.mgdl < below:
            run += 1
        else:
            count += run >= need
            run = 0
    return count + (run >= need)
