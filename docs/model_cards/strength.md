# Model card for concrete compressive strength

## Model and data

Histogram gradient boosting regression, scikit-learn. Model `9a6c8beadf10` uses the eight raw
inputs plus eight engineered features. The final fit used 616 measurements across 256 exact mix
designs. A separate 201 measurements across 86 mixes supplied split-conformal calibration. The
nominal 90% target is joint across a mix's measurements, with one maximum absolute residual
per calibration mix. The 79th of 86 calibration scores gives a fixed half-width of 10.86 MPa
and a rank probability of 79/87 (90.8%) under exchangeability with no ties. The holdout uses
strata based on observed mean strength, so exchangeability with calibration is unestablished. See the
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

| Metric | Result on 188 measurements from 86 unseen exact ingredient tuples |
| --- | ---: |
| MAE (MPa) | 3.83 (95% stratified mix bootstrap CI 3.05–4.61) |
| RMSE (MPa) | 5.17 (4.12–6.12) |
| R² | 0.895 (0.868–0.922) |
| Empirical coverage per measurement | 180/188 (95.7%) |
| Empirical joint coverage per mix | 80/86 (93.0%) |
| Mean interval width (MPa) | 21.71 |

MAE rose to 7.24 MPa at observed strengths of 60 MPa and above, with a mean underprediction of
7.24 MPa. That slice contains 11 measurements from four mixes. At ages 1–7 days, the MAE was
4.59 MPa and coverage was 91.1%. See the [full results](../results/strength.md) for the random-row
leakage comparison, stratified whole-mix bootstrap intervals, and error slices.

In the tolerance sensitivity analysis, 22 test tuples were within 0.5 kg/m³ per ingredient
of a fitting tuple. Those 29 measurements had 4.48 MPa MAE. The other 64 tuples had
3.71 MPa MAE over 159 measurements. Batch provenance is unknown.

## Integration

`uv run --locked baulens strength train --config configs/strength.yaml` produces
`pipeline.joblib` (about 67 KiB), `metadata.json`, and `manifest.json` under
`artifacts/strength/model/`. Ship all three files for inference. The manifest binds the files to the
same fitted artifact and the loader verifies them. Use a trusted build with the same locked
environment. Training needs the UCI workbook at the configured path, and inference can start as
soon as training publishes the artifact.

```python
from pathlib import Path

from baulens.strength.inference import StrengthPredictor

predictor = StrengthPredictor.load(Path("artifacts/strength/model"))
# Pass a validated baulens.schemas.ConcreteMix to predictor.predict(mix).
```

Load once at service startup and reuse for predictions. `predict` returns a `StrengthPrediction`
with the point estimate, interval, input-range flag, five signed SHAP contributions in MPa ordered by
absolute magnitude, model name, and version. In the initial Windows CPU benchmark, imports took
about 5.0 s, `load` including first-time SHAP import took 1.8 s, and median warm latency was 22 ms
over 30 single-mix calls. The full review rerun took 21.5 min under shared CPU load.
