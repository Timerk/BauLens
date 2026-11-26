# Model card for concrete compressive strength

## Model and data

Histogram gradient boosting regression, scikit-learn. Model `9a6c8beadf10` uses the eight raw
inputs plus eight engineered features. The final fit used 616 measurements across 256 exact mix
designs. A separate 201 measurements across 86 mixes supplied split-conformal calibration. The
90% intervals have a fixed half-width of 10.86 MPa for this artifact. See the
[evaluation protocol](../adr/0003-strength-evaluation-protocol.md) for the split and selection.

The dataset is I-Cheng Yeh's Concrete Compressive Strength workbook from UCI, licensed CC BY
4.0. The original file has 1,030 measurements. Removing 25 exact duplicates left 1,005
measurements from 428 ingredient combinations. The raw inputs describe seven ingredient masses
in kg/m³ and testing age in days. Strength is in MPa. The source omits material and batch
identifiers, cement type, curing conditions, and detailed specimen metadata.
[UCI dataset](https://doi.org/10.24432/C5PK67)

## Intended use and limitations

Use for research demonstrations and preliminary decision support for formulations comparable to
the historical dataset. Use standard specimen tests for material acceptance and structural
decisions, and have a qualified engineer interpret the results. EN 12390-3 specifies compressive
strength testing of hardened concrete specimens.
[EN 12390-3 standard record](https://landingpage.bsigroup.com/LandingPage/Undated?UPI=000000000030015544)

Different curing regimes, constituent sources, specimen geometries and modern blended cements
can change strength and interval coverage. The input-range flag checks each raw feature separately.
An in-range formulation can still combine ingredients in an unfamiliar way. The interval describes
empirical prediction uncertainty under a comparable sampling process. It provides no lower
characteristic strength or structural design value. Report an out-of-range flag alongside estimates,
present intervals with point predictions, and preserve the original measurement units in downstream
outputs. Measure external performance before applying the model in a different setting.

## Evaluation

| Metric | Result on 188 measurements from 86 unseen mixes |
| --- | ---: |
| MAE (MPa) | 3.83 (95% group bootstrap CI 3.00–4.68) |
| RMSE (MPa) | 5.17 (4.06–6.21) |
| R² | 0.895 (0.864–0.923) |
| Coverage per measurement | 95.7% |
| Coverage for every measurement of a mix | 93.0% |
| Mean interval width (MPa) | 21.71 |

MAE rose to 7.24 MPa at observed strengths of 60 MPa and above, with a mean underprediction of
7.24 MPa. That slice contains 11 measurements from four mixes. At ages 1–7 days, the MAE was
4.59 MPa and coverage was 91.1%. See the [full results](../results/strength.md) for the random-row
leakage comparison, group bootstrap intervals, and error slices.

## Integration

`uv run --locked baulens strength train --config configs/strength.yaml` produces
`artifacts/strength/model/pipeline.joblib` (about 67 KiB) and `metadata.json`. The build needs
the UCI workbook at the configured path. It does not need test evaluation to make a usable model.
Load joblib files from a trusted build with the same locked environment.

```python
from pathlib import Path

from baulens.strength.inference import StrengthPredictor

predictor = StrengthPredictor.load(Path("artifacts/strength/model"))
# Pass a validated baulens.schemas.ConcreteMix to predictor.predict(mix).
```

Load once at service startup and reuse for predictions. `predict` returns a `StrengthPrediction`
with the point estimate, interval, input-range flag, five signed SHAP contributions in MPa ordered by
absolute magnitude, model name, and version. On the Windows CPU laptop under shared load, the
initial imports took about 5.0 s, `load` including first-time SHAP import took 1.8 s, and median
warm latency was 22 ms over 30 single-mix calls. Full training took 177 s under shared load.
