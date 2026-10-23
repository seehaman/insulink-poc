"""End-to-end proof of concept. Run:  python3 demo.py"""
from insulink import calculator, escalation, food_response as fr, safety, simulator
from insulink.models import fmt_time, make_meal


def header(title):
    print(f"\n=== {title} ===")


def main():
    rules = safety.load_rules()
    readings, meals = simulator.generate(days=14, seed=7)
    print(f"Simulated {len(readings)} CGM readings and {len(meals)} meals over 14 days (synthetic data).")

    header("1. Safety rules (fixed thresholds, no AI)")
    tir = safety.time_in_range(readings, rules)
    print(f"Time below / in / above range: {tir['below']:.1%} / {tir['in_range']:.1%} / {tir['above']:.1%}")
    alerts = safety.scan(readings, rules)
    counts = {}
    for a in alerts:
        counts[a.kind] = counts.get(a.kind, 0) + 1
    print("Alerts raised:", counts)
    urgent = [a for a in alerts if a.kind == "urgent_low"]
    for a in urgent[:1]:
        print(f"First urgent low: {fmt_time(a.t)}  {a.value} mg/dL  -> {a.message}")

    header("2. Escalation if the user does not respond")
    if urgent:
        esc = escalation.Escalation(
            [escalation.Contact("Mother", 1), escalation.Contact("Friend", 2), escalation.Contact("Brother", 3)],
            **rules["escalation"],
        )
        t0 = urgent[0].t
        events = list(esc.trigger(t0, "urgent_low"))
        for minute in range(t0 + 5, t0 + 41, 5):
            events += esc.tick(minute)
        for n in events:
            print(f"  +{n.t - t0:>2} min  notify {n.recipient:<7} {n.message}")
        print("Manual 'I'm Not Okay' notifies everyone at once:")
        esc2 = escalation.Escalation([escalation.Contact("Mother", 1), escalation.Contact("Friend", 2)])
        for n in esc2.not_okay(0):
            print(f"  +0 min  notify {n.recipient:<7} {n.message}")

    header("3. Personal food response (learned only from the logged data)")
    responses = fr.measure_all(meals, readings)
    cutoff = 10 * 1440
    train = [r for r in responses if r.meal.t < cutoff]
    test = [r for r in responses if r.meal.t >= cutoff]
    print(f"Measured {len(responses)} meal responses. Learn on days 1-10 ({len(train)}), test on days 11-14 ({len(test)}).")
    ranking = sorted(fr.rise_by_food(train).items(), key=lambda kv: -kv[1])
    print("Average rise of meals containing each food (mg/dL):")
    for food, rise in ranking[:6]:
        print(f"  {food:<11} +{rise:.0f}")

    header("4. Does personalisation beat generic methods? (held-out meals)")
    ev = fr.evaluate(train, test, k=3)
    print(f"Mean absolute error of predicted rise, lower is better (mg/dL):")
    print(f"  Personal nearest-neighbour : {ev['mae_personal_knn']:.1f}")
    print(f"  Carb-count ratio           : {ev['mae_carb_ratio']:.1f}")
    print(f"  Global average             : {ev['mae_global_mean']:.1f}")

    header("5. What-if meal simulator, explained")
    a = make_meal(20 * 60, ("rice", "chicken", "soft_drink"))
    b = make_meal(20 * 60, ("chicken", "vegetables", "water"))
    for label, meal in (("Meal A", a), ("Meal B", b)):
        print(f"{label}: {' + '.join(meal.items)} ({meal.carbs:.0f} g carbs)")
        print(fr.explain(fr.predict_rise(meal, train)))

    header("6. Educational ICR/ISF calculator with hard limits")
    est = calculator.estimate(carbs_g=60, current_mgdl=200, icr=10, isf=45, rules=rules)
    print(f"60 g carbs, glucose 200, ICR 10, ISF 45 -> meal {est.meal_units} + correction "
          f"{est.correction_units} = {est.total_units} units (estimate)")
    print(est.disclaimer)
    low = calculator.estimate(carbs_g=60, current_mgdl=60, icr=10, isf=45, rules=rules)
    print("Glucose 60 ->", low.notes[0])
    try:
        calculator.estimate(60, 200, icr=None, isf=45, rules=rules)
    except calculator.CalculatorError as exc:
        print("Missing ICR ->", exc)


if __name__ == "__main__":
    main()
