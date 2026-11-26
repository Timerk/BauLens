# 0003 Evaluate strength on unseen exact ingredient tuples

Accepted on 2026-11-26.

The UCI concrete strength workbook includes the same formulation at several testing ages. A row
split can give the model information about a validation mix from another age. The deployment
benchmark estimates error for ingredient tuples absent from fitting. The workbook cannot identify
independent production batches.

## Data and splits

- Drop exact duplicates over the eight inputs and target once. With no specimen or batch IDs,
  identical entries add no identifiable independent evidence. Keep rows with the same formulation
  and age if their measured strengths differ.
- Keep the original exact tuple of seven ingredient amounts as the primary grouping key. It is
  reproducible, keeps every repeated observation of the same recorded composition together, and
  preserves the original holdout. The workbook provides no batching or assay precision metadata to
  justify a universal clustering tolerance. Testing age is excluded from the key.
- Assign 20% of mix IDs to the frozen test partition. Stratify by quantile bins of mean strength per
  mix. Begin with five bins, reduce the count for small samples or tied values, and use an
  unstratified split when feasible bins cannot be formed. This is the only operation that uses
  labels outside development to make a split. Row counts and individual strength distributions
  remain approximate because mix sizes and ages vary. Save the original stratum for each test mix.
- Assign 25% of the remaining mix IDs uniformly at random to calibration. The rest are available
  for fitting and model selection. Save partition indices and verify the data checksum before
  evaluation.

### Sensitivity to near-duplicate ingredients

Fix the audit rule from the review before rerunning the benchmark. A test tuple has a nearby fitting
tuple if all seven absolute ingredient differences are at most 0.5 kg/m³. Compare it only with the
final fitting partition, ignoring age and strength. Keep the split, model selection, and calibration
fixed. From the frozen winner's existing test predictions, report the counts and row-weighted MAE
for tuples with and without a nearby fitting tuple. This threshold probes similarity in recorded
inputs. Batch provenance remains unknown and the subset differences are descriptive.

## Selection and evaluation

All model and feature combinations share five shuffled group folds repeated three times with
seeds `seed + repeat` (zero-based). Each fold's MAE-based grid search uses three independently
constructed group folds from that fold's fitting rows. The lowest mean outer MAE determines the
winner. Evaluate that family and feature set in an otherwise matched nested random-row CV as a
leakage diagnostic. Refit the winner with grouped tuning on all fitting rows, including rows used
as outer validation. Feature transformations and standardisation are fitted inside the pipelines.
Use equal fold weights for selection. Also report pooled out-of-fold MAE by weighting each validation
prediction equally across folds and repeats. Grouped and random-row folds differ in size, so the
comparison describes a split-protocol effect conditional on the selected model and feature set.

Mean, sample standard deviation and repeat-mean 95% Student t intervals describe outer fold
scores. The interval is `mean ± t_(0.975, R-1) × SD(repeat means) / sqrt(R)` for `R` repeats. Repeats
reuse the same observations, so the interval measures sensitivity to the splits and is too limited
for formal claims about differences between models. The smoke config uses a single repeat and
reports no interval. Test metric intervals use the 2.5th and 97.5th percentiles of at least 2,000
bootstrap samples of test mixes. Sample complete mixes with replacement within each of the
holdout's original mean-strength strata and preserve the mix count in every stratum. Include every
measurement of each draw. Point estimates and each replicate weight measurements equally.

Compute one calibration score per mix, the greatest absolute residual across its ages. For `n`
calibration mixes, use the `ceil((n + 1) × 0.90)`-th smallest score as the symmetric radius. The
nominal 90% target is joint across the recorded measurements of a new mix. With 86 calibration
mixes, the 79th score corresponds to a rank probability of 79/87 (90.8%) under group score
exchangeability and no ties. Test membership is stratified on observed mean strength, while
calibration is drawn from the remainder. Exchangeability is an assumption for the rank argument
and this split gives no guarantee. Report empirical counts and coverage by measurement and by
whole mix, plus empirical age-specific coverage and mean width. [Conformal method reference](https://arxiv.org/abs/2107.07511)

Test labels never guide features, model selection, tuning or interval calibration. Only the frozen
winner scores the test partition, through `baulens strength evaluate`. Training stages the pipeline
and diagnostics together and publishes their checksums in a manifest. Evaluation verifies them and
uses the serialized fit's SHA-256 to identify holdout predictions. It scores each fitted artifact
once and reuses its predictions when rebuilding reports. Feature permutation happens on
grouped outer validation folds. SHAP plots use calibration inputs. Future changes informed by these
holdout results require a fresh external holdout for another independent performance claim.
