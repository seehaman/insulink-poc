import unittest

from insulink import calculator, escalation, food_response as fr, safety, simulator
from insulink.models import Meal, Reading, make_meal

RULES = safety.load_rules()


class SimulatorTests(unittest.TestCase):
    def test_deterministic_and_sized(self):
        a, ma = simulator.generate(days=3, seed=1, inject_low_day=None)
        b, mb = simulator.generate(days=3, seed=1, inject_low_day=None)
        self.assertEqual(a, b)
        self.assertEqual(ma, mb)
        self.assertEqual(len(a), 3 * 288)
        self.assertEqual(len(ma), 9)

    def test_injected_low_exists(self):
        readings, _ = simulator.generate(days=7, seed=7, inject_low_day=5)
        self.assertTrue(any(r.mgdl < 54 for r in readings))


class SafetyTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(safety.classify(53.9, RULES), "urgent_low")
        self.assertEqual(safety.classify(54, RULES), "low")
        self.assertEqual(safety.classify(69.9, RULES), "low")
        self.assertEqual(safety.classify(70, RULES), "in_range")
        self.assertEqual(safety.classify(180, RULES), "in_range")
        self.assertEqual(safety.classify(180.1, RULES), "high")
        self.assertEqual(safety.classify(250, RULES), "high")
        self.assertEqual(safety.classify(250.1, RULES), "very_high")

    def test_scan_finds_urgent_low(self):
        readings, _ = simulator.generate(days=7, seed=7, inject_low_day=5)
        kinds = {a.kind for a in safety.scan(readings, RULES)}
        self.assertIn("urgent_low", kinds)

    def test_rapid_fall_detected(self):
        series = [Reading(t=i * 5, mgdl=150 - i * 15) for i in range(6)]  # -3 mg/dL/min
        kinds = {a.kind for a in safety.scan(series, RULES)}
        self.assertIn("rapid_fall", kinds)

    def test_cooldown_prevents_spam(self):
        series = [Reading(t=i * 5, mgdl=40) for i in range(10)]
        alerts = [a for a in safety.scan(series, RULES) if a.kind == "urgent_low"]
        self.assertEqual(len(alerts), 1)

    def test_time_in_range_sums_to_one(self):
        readings, _ = simulator.generate(days=2, seed=3)
        tir = safety.time_in_range(readings, RULES)
        self.assertAlmostEqual(sum(tir.values()), 1.0)


class EscalationTests(unittest.TestCase):
    def make(self):
        return escalation.Escalation(
            [escalation.Contact("B", 2), escalation.Contact("A", 1)], ack_timeout_min=10, step_min=10
        )

    def test_user_first_then_contacts_by_priority(self):
        e = self.make()
        first = e.trigger(0, "low")
        self.assertEqual([n.recipient for n in first], ["user"])
        self.assertEqual(e.tick(9), [])
        self.assertEqual([n.recipient for n in e.tick(10)], ["A"])
        self.assertEqual(e.tick(15), [])
        self.assertEqual([n.recipient for n in e.tick(20)], ["B"])
        self.assertEqual(e.tick(60), [])

    def test_acknowledge_stops_escalation(self):
        e = self.make()
        e.trigger(0, "low")
        e.acknowledge(5)
        self.assertEqual(e.tick(30), [])

    def test_resolve_stops_escalation(self):
        e = self.make()
        e.trigger(0, "low")
        e.resolve(5)
        self.assertEqual(e.tick(30), [])

    def test_not_okay_notifies_everyone_once(self):
        e = self.make()
        self.assertEqual({n.recipient for n in e.not_okay(0)}, {"A", "B"})
        self.assertEqual(e.not_okay(1), [])


class CalculatorTests(unittest.TestCase):
    def test_basic_math(self):
        est = calculator.estimate(60, 200, icr=10, isf=45, rules=RULES)
        self.assertEqual(est.meal_units, 6.0)
        self.assertEqual(est.correction_units, 2.0)
        self.assertEqual(est.total_units, 8.0)
        self.assertFalse(est.requires_confirmation)

    def test_no_correction_at_or_below_target(self):
        est = calculator.estimate(30, 100, icr=10, isf=45, rules=RULES)
        self.assertEqual(est.correction_units, 0.0)

    def test_rounding_to_half_unit(self):
        est = calculator.estimate(25, 110, icr=10, isf=45, rules=RULES)  # 2.5 units
        self.assertEqual(est.meal_units, 2.5)

    def test_missing_parameters_rejected(self):
        with self.assertRaises(calculator.CalculatorError):
            calculator.estimate(60, 200, icr=None, isf=45, rules=RULES)
        with self.assertRaises(calculator.CalculatorError):
            calculator.estimate(60, 200, icr=10, isf=None, rules=RULES)

    def test_out_of_range_rejected(self):
        for kwargs in (dict(icr=1, isf=45), dict(icr=10, isf=500)):
            with self.assertRaises(calculator.CalculatorError):
                calculator.estimate(60, 200, rules=RULES, **kwargs)
        with self.assertRaises(calculator.CalculatorError):
            calculator.estimate(999, 200, icr=10, isf=45, rules=RULES)

    def test_low_glucose_gives_no_dose(self):
        est = calculator.estimate(60, 60, icr=10, isf=45, rules=RULES)
        self.assertEqual(est.total_units, 0.0)
        self.assertTrue(est.requires_confirmation)

    def test_large_total_requires_confirmation(self):
        est = calculator.estimate(150, 300, icr=8, isf=30, rules=RULES)
        self.assertTrue(est.requires_confirmation)

    def test_disclaimer_always_present(self):
        est = calculator.estimate(60, 200, icr=10, isf=45, rules=RULES)
        self.assertIn("Not medical advice", est.disclaimer)


class FoodResponseTests(unittest.TestCase):
    def test_measure_known_curve(self):
        # flat 100 until t=60, then rises to 160 at t=120, back to 100 by t=180
        series = {}
        for t in range(0, 301, 5):
            if t < 60:
                series[t] = 100.0
            elif t <= 120:
                series[t] = 100.0 + (t - 60)
            elif t <= 180:
                series[t] = 160.0 - (t - 120)
            else:
                series[t] = 100.0
        meal = Meal(t=62, items=("rice",), carbs=45)  # off-grid on purpose
        r = fr.measure(meal, series)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.baseline, 100.0)
        self.assertAlmostEqual(r.peak, 160.0)
        self.assertAlmostEqual(r.rise, 60.0)

    def test_similarity_properties(self):
        a = make_meal(0, ("rice", "chicken"))
        self.assertAlmostEqual(fr.similarity(a, a), 1.0)
        b = make_meal(0, ("fish", "water"))
        self.assertLess(fr.similarity(a, b), 0.5)

    def test_all_meals_measured_on_simulated_data(self):
        readings, meals = simulator.generate(days=14, seed=7)
        self.assertEqual(len(fr.measure_all(meals, readings)), len(meals))

    def test_personal_model_beats_global_average(self):
        readings, meals = simulator.generate(days=14, seed=7)
        responses = fr.measure_all(meals, readings)
        train = [r for r in responses if r.meal.t < 10 * 1440]
        test = [r for r in responses if r.meal.t >= 10 * 1440]
        ev = fr.evaluate(train, test)
        self.assertLess(ev["mae_personal_knn"], ev["mae_global_mean"])

    def test_explanation_lists_evidence(self):
        readings, meals = simulator.generate(days=5, seed=2)
        history = fr.measure_all(meals, readings)
        text = fr.explain(fr.predict_rise(make_meal(0, ("rice", "chicken")), history))
        self.assertIn("similar past meals", text)
        self.assertIn("mg/dL", text)


if __name__ == "__main__":
    unittest.main()
