# InsuLink: Proof of Concept

Academic graduation project, School of Computing, ESLSCA University. This repository holds the first ~10% of the implementation: the core logic that carries the most technical risk, written as dependency-free Python (3.10+) so it can be tested in isolation, and evaluated on real patient data. The mobile app (Flutter + Firebase) will port this logic.

> **Educational prototype. Not a medical device. Not medical advice.**
> All thresholds in `config/safety_rules.json` are placeholders that a clinician must review.

## What it demonstrates

| Module | Purpose |
|---|---|
| `insulink/safety.py` | Rule-based alerts (low, urgent low, high, very high, rapid change), time in range |
| `insulink/escalation.py` | User alert, acknowledgement timeout, contacts in priority order, "I'm Not Okay" |
| `insulink/calculator.py` | Educational ICR/ISF calculator with hard limits, never auto-filled |
| `insulink/food_response.py` | Per-meal response, similar-meal matching, explainable prediction |
| `insulink/simulator.py` | Synthetic CGM data with a hidden *personal* sensitivity per food |
| `insulink/hupa.py` | HUPA-UCM real-data adapter: loading, carb-unit normalisation, evaluation |

## Run

```bash
python3 demo.py                                  # end-to-end demo (synthetic data)
python3 -m unittest discover -s tests -v         # 30 tests, no dataset needed
python3 hupa_eval.py "path/to/Preprocessed"      # real-data evaluation (HUPA-UCM)
```

No installs needed.

## Real data: HUPA-UCM

[HUPA-UCM Diabetes Dataset](https://data.mendeley.com/datasets/3hbcscwz44) (Hidalgo et al., Mendeley Data, 2024, CC BY 4.0): 25 people with type 1 diabetes, FreeStyle Libre 2 glucose every 5 minutes, insulin, carbohydrates and Fitbit activity. Download it and pass its `Preprocessed` folder to `hupa_eval.py` (default `data/HUPA-UCM/Preprocessed`, which is git-ignored). The data is not redistributed here.

Results of `hupa_eval.py` (first-pass academic analysis):

- Safety engine on real readings: 8 of 25 patients reach the 70% time-in-range target; 22 of 25 had a low under 54 mg/dL lasting 15+ minutes.
- Held-out meals (22 patients, time-ordered 70/30 split, 662 test meals), mean absolute error of peak rise averaged over patients:

| Predictor | MAE (mg/dL) |
|---|---|
| Personal nearest neighbours | 36.4 |
| Carb-count ratio | 37.5 |
| Global average | **32.4** |

- With ~2 weeks (~30 meals) of history, personalisation does not beat a global average yet. With months of history it does: for patient 28 the personal error falls from 29.9 to 20.9 mg/dL as history grows from 10 to 120 meals, 23% below the global average.
- Design consequence: start new users from population estimates and shift to personal ones as meals accumulate.

Data note: the documentation says carbs are in grams, but most patients log 10 g exchanges. `hupa.carb_factor` detects this per patient.

The synthetic demo (`demo.py`) still shows the personal predictor winning (13.7 vs 22.8 vs 31.1 mg/dL), because personal food sensitivities exist there by construction. It checks the method, not real-world performance.

## Design rules

1. Safety logic is fixed rules in a config file. No AI decides alerts or treatment.
2. The calculator needs user-entered ICR/ISF, enforces ranges, refuses to dose when glucose is low, and requires confirmation above a limit.
3. Every recommendation shows the past meals it is based on.

## Roadmap

Blended population-to-personal estimates, insulin and activity features, Flutter app, Firebase, server-side escalation timers, forecasting model. See the proposal chapters in `docs/overleaf/`.

## Team

Ahmed Abdelsalam, Youssef Gharib, Mohamed Emad, Omar Wael. Supervisor: Dr. Raghda.
