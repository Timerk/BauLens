import json
from pathlib import Path
from shutil import copytree

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from typer.testing import CliRunner

from baulens.cli import app
from baulens.config import StrengthConfig, load_config
from baulens.paths import project_path
from baulens.schemas import ConcreteMix, StrengthPrediction
from baulens.strength import evaluate, train
from baulens.strength.artifacts import CV_SUMMARY_PATH, FIT_FILES, MANIFEST_PATH
from baulens.strength.features import RAW_FEATURES, MixFeatures
from baulens.strength.inference import FittedStrength, StrengthPredictor
from baulens.strength.metrics import bootstrap_metrics, conformal_radius
from baulens.strength.modeling import Candidate
from baulens.strength.splits import (
    fold_indices,
    mix_groups,
    near_fitting_mix,
    prepare_data,
    split_partitions,
)


@pytest.fixture
def settings() -> StrengthConfig:
    return load_config(Path("configs/smoke/strength.yaml"), StrengthConfig)


@pytest.fixture
def isolated_settings(tmp_path: Path, settings: StrengthConfig) -> StrengthConfig:
    return settings.model_copy(
        update={
            "artifacts_dir": tmp_path,
            "results_path": tmp_path / "results.md",
            "figures_dir": tmp_path / "figures",
        }
    )


def test_engineered_values_and_feature_names() -> None:
    frame = pd.DataFrame([[200, 50, 50, 150, 6, 1000, 500, 28]], columns=list(RAW_FEATURES))
    transformer = MixFeatures().fit(frame)
    features = transformer.transform(frame)
    assert list(features.columns) == transformer.get_feature_names_out().tolist()
    pd.testing.assert_frame_equal(features.loc[:, list(RAW_FEATURES)], frame)
    assert features.iloc[0, 8:].to_numpy() == pytest.approx(
        [0.75, 0.5, np.log(28), 1 / 3, 0.02, 2.0, 5.0, 1956.0]
    )
    pd.testing.assert_frame_equal(MixFeatures(engineered=False).fit_transform(frame), frame)


def test_engineering_handles_zero_denominators() -> None:
    frame = pd.DataFrame([[0, 0, 0, 150, 6, 1000, 0, 0]], columns=list(RAW_FEATURES))
    features = MixFeatures().fit_transform(frame)
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, ["water_cement", "water_binder", "coarse_fine_aggregate"]].tolist() == [
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize("cement", [0.0, -1.0])
def test_prepare_data_rejects_nonpositive_cement(
    tmp_path: Path, settings: StrengthConfig, cement: float
) -> None:
    rows = pd.read_csv(project_path(settings.data_path))
    rows.loc[0, "cement"] = cement
    csv = tmp_path / "nonpositive.csv"
    rows.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="require cement > 0"):
        prepare_data(csv)


def test_groups_ignore_age_and_target_and_use_every_ingredient() -> None:
    frame = pd.DataFrame([[200, 50, 50, 150, 6, 1000, 500, 28]], columns=list(RAW_FEATURES))
    changes = [
        frame.assign(age_days=56),
        *(frame.assign(**{name: frame[name] + 1}) for name in RAW_FEATURES[:-1]),
    ]
    rows = pd.concat([frame, *changes], ignore_index=True)
    rows["compressive_strength_mpa"] = np.arange(len(rows))
    groups = mix_groups(rows)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.nunique() == 8
    pd.testing.assert_series_equal(mix_groups(rows.iloc[::-1]).sort_index(), groups)


def test_near_mix_uses_inclusive_per_ingredient_tolerance_and_ignores_age() -> None:
    fit = pd.DataFrame([[200, 50, 50, 150, 6, 1000, 500, 28]], columns=list(RAW_FEATURES))
    test = pd.DataFrame(
        [[200.5, 50, 50, 150, 5.5, 1000.5, 499.5, 80], [200, 50, 50, 150, 5.49, 1000, 500, 28]],
        columns=list(RAW_FEATURES),
        index=[9, 20],
    )
    pd.testing.assert_series_equal(
        near_fitting_mix(fit, test), pd.Series([True, False], index=test.index)
    )


def test_split_and_grouped_folds_are_disjoint(settings: StrengthConfig) -> None:
    frame, _ = prepare_data(project_path(settings.data_path))
    groups = mix_groups(frame)
    partitions = split_partitions(frame, groups, settings)
    fit, cal, test = (
        set(groups.iloc[idx]) for idx in (partitions.fit, partitions.calibration, partitions.test)
    )
    assert fit.isdisjoint(cal) and fit.isdisjoint(test) and cal.isdisjoint(test)
    assert fit | cal | test == set(groups)
    assert len(test) == int(np.ceil(settings.test_fraction * groups.nunique()))
    assert partitions.strength_bins == 5
    strata = pd.Series(partitions.test_strata, index=partitions.test)
    assert strata.nunique() == partitions.strength_bins
    assert strata.groupby(groups.iloc[partitions.test]).nunique().eq(1).all()
    fit_groups = groups.iloc[partitions.fit]
    for fit_idx, validation_idx in fold_indices(fit_groups, 2, settings.seed, grouped=True):
        assert set(fit_groups.iloc[fit_idx]).isdisjoint(fit_groups.iloc[validation_idx])
    np.testing.assert_array_equal(split_partitions(frame, groups, settings).test, partitions.test)


def test_bootstrap_preserves_strata_and_resamples_whole_mixes() -> None:
    # Each stratum has one mix, so preserving counts makes every replicate identical.
    actual = np.array([4.0, 9.0, 50.0])
    predicted = np.array([3.0, 7.0, 40.0])
    result = bootstrap_metrics(
        actual, predicted, pd.Series([0, 0, 1]), pd.Series([0, 0, 1]), 2000, 42
    )
    assert result["mae"] == pytest.approx((13 / 3, 13 / 3))
    assert result["rmse"] == pytest.approx((np.sqrt(35), np.sqrt(35)))


def test_conformal_uses_mix_maxima_and_finite_sample_rank() -> None:
    groups = pd.Series(np.repeat(np.arange(12), 2))
    errors = np.column_stack((np.zeros(12), -np.arange(1, 13))).ravel()
    assert conformal_radius(errors, groups, 0.90) == 12
    with pytest.raises(ValueError, match="Too few calibration mixes"):
        conformal_radius(errors[:16], groups.iloc[:16], 0.90)


def test_conformal_joint_coverage_with_dependent_ages() -> None:
    rng = np.random.default_rng(27)
    errors = rng.normal(size=(5000, 1)) + rng.normal(scale=0.4, size=(5000, 2))
    calibration = errors[:1000].ravel()
    radius = conformal_radius(calibration, pd.Series(np.repeat(np.arange(1000), 2)), 0.90)
    joint_coverage = (np.abs(errors[1000:]).max(axis=1) <= radius).mean()
    assert joint_coverage == pytest.approx(0.90, abs=0.035)


def test_hgb_predictor_maps_features_and_shap_adds_up_in_mpa(settings: StrengthConfig) -> None:
    frame, _ = prepare_data(project_path(settings.data_path))
    inputs = frame.loc[:, list(RAW_FEATURES)]
    pipeline = Pipeline(
        [
            ("features", MixFeatures()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=12, max_leaf_nodes=3, min_samples_leaf=5, random_state=42
                ),
            ),
        ]
    )
    with threadpool_limits(limits=1):
        pipeline.fit(inputs, frame["compressive_strength_mpa"])
        artifact = FittedStrength(
            pipeline,
            Candidate("Histogram GBM", True),
            "example",
            "checksum",
            1.0,
            {name: (inputs[name].min(), inputs[name].max()) for name in RAW_FEATURES},
            pipeline.named_steps["features"].transform(inputs).mean().to_numpy(),
            split_partitions(frame, mix_groups(frame), settings),
        )
        predictor = StrengthPredictor(artifact)
        features, shap = predictor.explain(inputs.iloc[:12])
        reference_mpa = pipeline.predict(inputs).mean()
        np.testing.assert_allclose(
            reference_mpa + shap.sum(axis=1), pipeline.predict(inputs.iloc[:12])
        )
        assert np.ptp(shap.sum(axis=1)) > 0
        mix = ConcreteMix.model_validate(inputs.iloc[0].to_dict())
        prediction = predictor.predict(mix)
    assert features.iloc[0]["water_cement"] == pytest.approx(mix.water / mix.cement)
    assert prediction.point_mpa == pytest.approx(reference_mpa + shap[0].sum())
    expected = [str(features.columns[i]) for i in np.argsort(-np.abs(shap[0]), kind="stable")[:5]]
    assert [feature.feature for feature in prediction.top_features] == expected
    for feature in prediction.top_features:
        assert feature.value == pytest.approx(features.iloc[0][feature.feature])
        assert feature.contribution_mpa == pytest.approx(
            shap[0, features.columns.get_loc(feature.feature)]
        )


def test_predictor_rejects_overflowing_ratio_before_estimator(
    settings: StrengthConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, _ = prepare_data(project_path(settings.data_path))
    inputs = frame.loc[:, list(RAW_FEATURES)]

    def unexpected_predict(inputs: pd.DataFrame) -> None:
        pytest.fail("Overflowing ratios must be rejected before calling the estimator")

    pipeline = Pipeline([("features", MixFeatures()), ("regressor", DummyRegressor())])
    pipeline.fit(inputs, frame["compressive_strength_mpa"])
    monkeypatch.setattr(pipeline.named_steps["regressor"], "predict", unexpected_predict)
    artifact = FittedStrength(
        pipeline,
        Candidate("Mean", True),
        "example",
        "checksum",
        1.0,
        {},
        np.zeros(16),
        split_partitions(frame, mix_groups(frame), settings),
    )
    values = inputs.iloc[0].to_dict()
    values.update(cement=1e-310, blast_furnace_slag=0, fly_ash=0)
    with pytest.raises(ValueError, match="non-finite engineered features: water_cement"):
        StrengthPredictor(artifact).predict(ConcreteMix.model_validate(values))


def test_failed_refit_preserves_old_bundle_and_mixed_sidecars_are_rejected(
    isolated_settings: StrengthConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = isolated_settings.artifacts_dir
    monkeypatch.setattr("baulens.strength.training.FAMILIES", ("Mean",))
    original = train(isolated_settings)
    published = {name: (root / name).read_bytes() for name in (*FIT_FILES, MANIFEST_PATH)}
    monkeypatch.setattr("baulens.strength.training.FAMILIES", ("Linear",))

    def failed_dump(artifact: FittedStrength, destination: Path, **kwargs: object) -> None:
        assert artifact.candidate.family == "Linear"
        assert artifact.version == original.model_version
        destination.write_bytes(b"partial dump")
        raise OSError("simulated joblib dump failure")

    with monkeypatch.context() as patch:
        patch.setattr("baulens.strength.training.joblib.dump", failed_dump)
        with pytest.raises(OSError, match="simulated joblib dump failure"):
            train(isolated_settings)
    assert published == {name: (root / name).read_bytes() for name in published}
    assert evaluate(isolated_settings).model_name == "Mean"

    # Changing a sidecar with the same config/data version must fail closed.
    cv_path = root / CV_SUMMARY_PATH
    cv = json.loads(published[CV_SUMMARY_PATH])
    cv["grouped"][0]["model"] = "Linear"
    cv_path.write_text(json.dumps(cv), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match the manifest"):
        evaluate(isolated_settings)


@pytest.mark.smoke
def test_strength_cli_smoke_and_predictor_round_trip(
    tmp_path: Path, isolated_settings: StrengthConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = isolated_settings
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
    runner = CliRunner()
    for command in ("train", "evaluate"):
        result = runner.invoke(app, ["strength", command, "--config", str(config)])
        assert result.exit_code == 0, result.exception or result.output

    model_dir = tmp_path / "model"
    assert (model_dir / "metadata.json").is_file()
    assert (tmp_path / "test_metrics.json").is_file()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3
    assert "Random row" in settings.results_path.read_text(encoding="utf-8")
    folds = pd.read_csv(tmp_path / "cv_folds.csv")
    assert len(folds[folds.protocol == "grouped"]) == 24
    assert (folds.loc[folds.protocol == "grouped", "overlapping_validation_rows"] == 0).all()

    artifact = FittedStrength.load(model_dir)
    raw, _ = prepare_data(project_path(settings.data_path))
    mix = ConcreteMix.model_validate(
        raw.iloc[artifact.partitions.fit[0]][list(RAW_FEATURES)].to_dict()
    )
    predictor = StrengthPredictor.load(model_dir)
    output = predictor.predict(mix)
    assert output == StrengthPredictor.load(model_dir).predict(mix)
    deployment_dir = tmp_path / "deployed-strength"
    copytree(model_dir, deployment_dir)
    assert output == StrengthPredictor.load(deployment_dir).predict(mix)
    assert output.in_training_domain
    assert not predictor.predict(mix.model_copy(update={"age_days": 3650})).in_training_domain
    assert output.interval_coverage == 0.90
    assert output.upper_mpa - output.lower_mpa == pytest.approx(2 * artifact.radius_mpa)
    assert len(output.top_features) == 5
    magnitudes = [abs(feature.contribution_mpa) for feature in output.top_features]
    assert sorted(magnitudes, reverse=True) == magnitudes
    mix_json = tmp_path / "mix.json"
    mix_json.write_text(mix.model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app, ["strength", "predict", "--config", str(config), "--mix-json", str(mix_json)]
    )
    assert result.exit_code == 0, result.exception or result.output
    assert StrengthPrediction.model_validate_json(result.stdout) == output

    def unexpected_rescore(*args: object) -> None:
        pytest.fail("Repeated evaluation should reuse the frozen holdout predictions")

    monkeypatch.setattr("baulens.strength.evaluation.predictions", unexpected_rescore)
    assert evaluate(settings).model_version == output.model_version
    with pytest.raises(ValueError, match="differs from the fitted artifact"):
        evaluate(settings.model_copy(update={"seed": 43}))
