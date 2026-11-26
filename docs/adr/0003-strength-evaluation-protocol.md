# 0003 Evaluate strength on unseen mix designs

Accepted on 2026-11-26.

The UCI concrete strength workbook includes the same formulation at several testing ages. A row
split can give the model information about a validation mix from another age. The deployment
question is how well the model estimates strength for a previously unseen formulation.

## Data and splits

- Drop exact duplicates over the eight inputs and target once. With no specimen or batch IDs,
  identical entries add no identifiable independent evidence. Keep rows with the same formulation
  and age if their measured strengths differ.
- Identify a mix by the exact tuple of seven ingredient amounts. Testing age is excluded from the
  key. The workbook has limited metadata for identifying near-duplicates or shared source batches.
- Assign 20% of mix IDs to the frozen test partition. Stratify by quantile bins of mean strength per
  mix. Begin with five bins, reduce the count for small samples or tied values, and use an
  unstratified split when feasible bins cannot be formed. This is the only operation that uses
  labels outside development to make a split. Row counts and individual strength distributions
  remain approximate because mix sizes and ages vary.
- Assign 25% of the remaining mix IDs uniformly at random to calibration. The rest are available
  for fitting and model selection. Save partition indices and verify the data checksum before
  evaluation.

## Selection and evaluation

All model and feature combinations share five shuffled group folds repeated three times with
seeds `seed + repeat` (zero-based). Each fold's MAE-based grid search uses three independently
constructed group folds from that fold's fitting rows. The lowest mean outer MAE determines the
winner. Evaluate that family and feature set in an otherwise matched nested random-row CV as a
leakage diagnostic. Refit the winner with grouped tuning on all fitting rows, including rows used
as outer validation. Feature transformations and standardisation are fitted inside the pipelines.

Mean, sample standard deviation and repeat-mean 95% Student t intervals describe outer fold
scores. The interval is `mean ± t_(0.975, R-1) × SD(repeat means) / sqrt(R)` for `R` repeats. Repeats
reuse the same observations, so the interval measures sensitivity to the splits and is too limited
for formal claims about differences between models. The smoke config uses a single repeat and
reports no interval. Test metric intervals use the 2.5th and 97.5th percentiles of at least 2,000
bootstrap samples of test mixes. Draw as many mixes as are in the holdout, sampling with
replacement and including every measurement of each draw. Point estimates and each bootstrap
replicate weight individual measurements equally.

Compute one calibration score per mix, the greatest absolute residual across its ages. For `n`
calibration mixes, use the `ceil((n + 1) × 0.90)`-th smallest score as the symmetric radius. Under
group exchangeability this targets joint coverage over the measured ages of a new mix. The
strength-stratified test partition and new deployment populations can depart from exchangeability.
Report coverage both by measurement and by whole mix, plus empirical age-specific coverage and
mean width. Fixed-width intervals are a starting point for error characterisation. [Conformal
method reference](https://arxiv.org/abs/2107.07511)

Test labels never guide features, model selection, tuning or interval calibration. Only the frozen
winner scores the test partition, through `baulens strength evaluate`. It writes holdout predictions
once per fitted artifact and reuses them when rebuilding reports. Feature permutation happens on
grouped outer validation folds. SHAP plots use calibration inputs. Future changes informed by these
holdout results require a fresh external holdout for another independent performance claim.
