"""Run the InsuLink engine on the HUPA-UCM real dataset.

    python3 hupa_eval.py [path/to/HUPA-UCM Diabetes Dataset/Preprocessed]

Default path: data/HUPA-UCM/Preprocessed. Download the dataset from
https://data.mendeley.com/datasets/3hbcscwz44 (CC BY 4.0). Pure Python, no installs.
"""
import sys

from insulink import hupa, safety


def main(folder):
    rules = safety.load_rules()
    patients = hupa.load_folder(folder)
    n_read = sum(len(p.readings) for p in patients)
    n_meals = sum(len(p.meals) for p in patients)
    print(f"HUPA-UCM: {len(patients)} patients, {n_read:,} glucose readings, {n_meals:,} carb entries")
    exch = [p.pid for p in patients if p.meals and p.carb_factor > 1]
    grams = [p.pid for p in patients if p.meals and p.carb_factor == 1]
    print(f"Carb units: {len(exch)} patients log 10 g exchanges, {len(grams)} log grams ({', '.join(grams)})")

    print("\n=== 1. Safety engine on real data (same config/safety_rules.json) ===")
    tirs = {p.pid: safety.time_in_range(p.readings, rules) for p in patients}
    meet = sum(1 for t in tirs.values() if t["in_range"] >= 0.70)
    med_tir = sorted(t["in_range"] for t in tirs.values())[len(tirs) // 2]
    med_below = sorted(t["below"] for t in tirs.values())[len(tirs) // 2]
    print(f"Patients reaching the 70% time-in-range target: {meet} of {len(patients)}")
    print(f"Median time in range {med_tir:.1%}, median time below 70 mg/dL {med_below:.1%} (target under 4%)")
    with_l2 = sum(1 for p in patients if hupa.low_episodes(p.readings, 54, 15) > 0)
    print(f"Patients with a low under 54 mg/dL lasting 15+ minutes: {with_l2} of {len(patients)}")
    urgent = sum(sum(1 for a in safety.scan(p.readings, rules) if a.kind == "urgent_low") for p in patients)
    print(f"Urgent-low alerts the rule engine would raise: {urgent:,}")

    print("\n=== 2. Personal food response vs generic methods (held-out meals) ===")
    res = hupa.evaluate(patients)
    n_train = sum(v["n_train"] for v in res["per_patient"].values())
    n_test = sum(v["n_test"] for v in res["per_patient"].values())
    print(f"{len(res['per_patient'])} patients with 10+ isolated meals; learn on first 70% ({n_train}), "
          f"test on last 30% ({n_test}). Global average rise {res['global_mean']:.1f} mg/dL.")
    print(f"{'patient':>7} {'learn':>5} {'test':>5} {'personal':>9} {'ratio':>7} {'global':>7}")
    for pid, v in res["per_patient"].items():
        print(f"{pid:>7} {v['n_train']:>5} {v['n_test']:>5} {v['personal']:>9.1f} {v['ratio']:>7.1f} {v['global']:>7.1f}")
    s = res["summary"]
    print("Mean absolute error of peak rise, average over patients (mg/dL):")
    print(f"  Personal nearest-neighbour : {s['personal']:.1f}")
    print(f"  Carb-count ratio           : {s['ratio']:.1f}")
    print(f"  Global average             : {s['global']:.1f}")
    print(f"Personal model had the lowest error for {s['personal_best']} of {len(res['per_patient'])} patients.")

    print("\n=== 3. How much history does personalisation need? ===")
    for pid, (tr, te) in res["splits"].items():
        if len(tr) < 120:
            continue
        print(f"Patient {pid} ({len(te)} fixed test meals): MAE with the most recent n learning meals")
        for n, e in hupa.learning_curve(tr, te, res["global_mean"]):
            print(f"  n={n:>4}  personal {e['personal']:5.1f}  ratio {e['ratio']:5.1f}  global {e['global']:5.1f}")

    print("\nFirst-pass academic analysis on real data. Not clinical evidence.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/HUPA-UCM/Preprocessed")
