"""Tests for the HUPA-UCM adapter. They build tiny CSV files, so the real dataset is not needed."""
import os
import tempfile
import unittest
from datetime import datetime, timedelta

from insulink import food_response as fr, hupa
from insulink.models import Meal, Reading

HEADER = "time;glucose;calories;heart_rate;steps;basal_rate;bolus_volume_delivered;carb_input\n"


def write_csv(folder, pid, rows):
    """rows: list of (datetime, glucose, carbs)."""
    path = os.path.join(folder, f"HUPA{pid}P.csv")
    with open(path, "w") as f:
        f.write(HEADER)
        for ts, g, c in rows:
            f.write(f"{ts.isoformat()};{g};1.0;80.0;0.0;0.1;0.0;{c}\n")
    return path


def day_of_readings(start, meals, base=110.0, rise=60.0):
    """One reading every 5 minutes for 24 h. Each meal (minute, carbs) adds a
    triangle peaking 60 minutes later."""
    rows = []
    for i in range(288):
        minute = i * 5
        g = base
        carbs = 0.0
        for mt, mc in meals:
            if minute == mt:
                carbs = mc
            dt = minute - mt
            if 0 <= dt <= 120:
                g += rise * (1 - abs(dt - 60) / 60)
        rows.append((start + timedelta(minutes=minute), round(g, 1), carbs))
    return rows


class HupaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.start = datetime(2020, 1, 1, 6, 0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_carb_units_detected(self):
        self.assertEqual(hupa.carb_factor([2, 3, 4, 5]), 10.0)    # 10 g exchanges
        self.assertEqual(hupa.carb_factor([45, 60, 80]), 1.0)     # grams
        self.assertEqual(hupa.carb_factor([]), 1.0)

    def test_load_normalises_time_and_carbs(self):
        path = write_csv(self.tmp.name, "0099", day_of_readings(self.start, [(60, 4.0)]))
        p = hupa.load_patient(path)
        self.assertEqual(p.pid, "0099")
        self.assertEqual(p.readings[0].t, 6 * 60)        # minutes since midnight of day 1
        self.assertEqual(len(p.meals), 1)
        self.assertEqual(p.meals[0].carbs, 40.0)         # 4 exchanges -> 40 g
        self.assertEqual(p.meals[0].items, ())

    def test_isolated_meal_measured_and_crowded_meal_skipped(self):
        rows = day_of_readings(self.start, [(60, 4.0), (300, 3.0), (360, 2.0)])
        p = hupa.load_patient(write_csv(self.tmp.name, "0098", rows))
        responses = hupa.measure_isolated(p)
        # the meal at +300 min has another meal 60 min later, so it is skipped
        self.assertEqual([r.meal.carbs for r in responses], [40.0, 20.0])
        first = responses[0]
        self.assertAlmostEqual(first.baseline, 110.0)
        self.assertAlmostEqual(first.rise, 60.0)
        self.assertEqual(first.time_to_peak_min, 60)

    def test_context_similarity(self):
        def resp(t, carbs, base):
            return fr.MealResponse(Meal(t, (), carbs), base, base, 0.0, 0, 0.0)
        a = resp(13 * 60, 50, 120)
        self.assertAlmostEqual(hupa.context_similarity(a, a), 1.0)
        far = resp(1 * 60, 110, 300)      # 60 g more, 12 h apart, 180 mg/dL higher
        self.assertAlmostEqual(hupa.context_similarity(a, far), 0.0)
        late, early = resp(23 * 60 + 30, 50, 120), resp(24 * 60 + 30, 50, 120)   # across midnight
        self.assertGreater(hupa.context_similarity(late, early), 0.95)

    def test_evaluate_uses_time_ordered_split(self):
        meals = [(60, 2.0), (360, 4.0), (720, 6.0)]
        rows = []
        for d in range(6):
            rows += day_of_readings(self.start + timedelta(days=d), meals)
        p = hupa.load_patient(write_csv(self.tmp.name, "0097", rows))
        res = hupa.evaluate([p], min_meals=10)
        tr, te = res["splits"]["0097"]
        self.assertLess(max(r.meal.t for r in tr), min(r.meal.t for r in te))
        self.assertEqual(set(res["summary"]), {"personal", "ratio", "global", "personal_best"})
        self.assertGreaterEqual(res["per_patient"]["0097"]["personal"], 0.0)

    def test_low_episodes(self):
        vals = [100, 50, 50, 50, 100, 50, 50, 100, 40, 40, 40, 40]
        readings = [Reading(i * 5, v) for i, v in enumerate(vals)]
        self.assertEqual(hupa.low_episodes(readings, below=54, min_minutes=15), 2)


if __name__ == "__main__":
    unittest.main()
