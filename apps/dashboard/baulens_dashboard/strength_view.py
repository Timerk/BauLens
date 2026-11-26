"""Concrete mix form and strength prediction."""

import streamlit as st

from baulens_dashboard.charts import contributions_chart, interval_chart
from baulens_dashboard.client import ApiClient, ApiError, FeatureRange, StrengthModel
from baulens_dashboard.labels import DEFAULT_MIX, DOCS_URL, MIX_FIELDS, number

# Physical bounds of the shared ConcreteMix schema; the training ranges are narrower.
_LIMITS: dict[str, float] = {
    "cement": 1000.0,
    "blast_furnace_slag": 600.0,
    "fly_ash": 400.0,
    "water": 400.0,
    "superplasticizer": 60.0,
    "coarse_aggregate": 1600.0,
    "fine_aggregate": 1300.0,
    "age_days": 3650.0,
}


def _unit(name: str) -> str:
    return "d" if name == "age_days" else "kg/m³"


def outside_training_range(mix: dict[str, float], ranges: list[FeatureRange]) -> list[str]:
    return [
        MIX_FIELDS[item["feature"]]
        for item in ranges
        if not item["minimum"] <= mix[item["feature"]] <= item["maximum"]
    ]


def _mix_form(ranges: dict[str, FeatureRange]) -> dict[str, float] | None:
    current: dict[str, float] = st.session_state.get("mix", DEFAULT_MIX)
    with st.form("mix_form"):
        columns = st.columns(2)
        values: dict[str, float] = {}
        for index, (name, label) in enumerate(MIX_FIELDS.items()):
            known = ranges.get(name)
            hint = (
                f"Trainingsbereich {number(known['minimum'])}–{number(known['maximum'])} "
                f"{_unit(name)}"
                if known
                else None
            )
            with columns[index % 2]:
                if name == "age_days":
                    values[name] = st.number_input(
                        f"{label} (d)",
                        min_value=1,
                        max_value=int(_LIMITS[name]),
                        value=int(current[name]),
                        step=1,
                        help=hint,
                    )
                else:
                    values[name] = st.number_input(
                        f"{label} ({_unit(name)})",
                        min_value=0.0,
                        max_value=_LIMITS[name],
                        value=float(current[name]),
                        step=5.0,
                        format="%.1f",
                        help=hint,
                    )
        submitted = st.form_submit_button("Festigkeit prognostizieren", type="primary")
    return values if submitted else None


def render(client: ApiClient, model: StrengthModel | None) -> None:
    if model is None:
        st.info(
            "Das Festigkeitsmodell ist nicht geladen. Mit `make strength` trainieren und die "
            "API neu starten (Details in der Seitenleiste)."
        )
        return
    ranges = {item["feature"]: item for item in model["training_ranges"]}
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### Rezeptur je m³ Beton")
        submitted = _mix_form(ranges)
        with st.expander("Trainingsbereich des Modells"):
            st.dataframe(
                [
                    {
                        "Merkmal": MIX_FIELDS[item["feature"]],
                        "Minimum": item["minimum"],
                        "Maximum": item["maximum"],
                        "Einheit": item["unit"],
                    }
                    for item in model["training_ranges"]
                ],
                hide_index=True,
                width="stretch",
            )
    if submitted is not None:
        try:
            st.session_state["strength"] = client.predict_strength(submitted)
            st.session_state["mix"] = submitted
        except ApiError as error:
            st.session_state.pop("strength", None)
            left.error(f"Prognose fehlgeschlagen: {error.detail}")
    with right:
        _result(model)


def _result(model: StrengthModel) -> None:
    prediction = st.session_state.get("strength")
    if prediction is None:
        st.markdown("##### Prognose")
        st.caption("Rezeptur eingeben und Prognose starten.")
        return
    mix = st.session_state["mix"]
    st.markdown("##### Prognose")
    st.metric(
        "Druckfestigkeit (Modellschätzung)",
        f"{number(prediction['point_mpa'])} MPa",
        help="Punktschätzung des Modells für das angegebene Prüfalter",
    )
    st.altair_chart(interval_chart(prediction), width="stretch")
    coverage = round(prediction["interval_coverage"] * 100)
    st.caption(
        f"{coverage} %-Prognoseintervall: {number(prediction['lower_mpa'])} bis "
        f"{number(prediction['upper_mpa'])} MPa (Methode: {prediction['interval_method']}). "
        "Die Abdeckung gilt im Mittel über Rezepturen wie in den Trainingsdaten, nicht für "
        "jede einzelne Mischung."
    )
    if not prediction["in_training_domain"]:
        outside = ", ".join(outside_training_range(mix, model["training_ranges"]))
        st.warning(
            f"Außerhalb des Trainingsbereichs: {outside or 'mindestens ein Wert'}. "
            "Die Prognose ist hier eine Extrapolation; das Intervall ist nicht abgesichert.",
            icon=":material/warning:",
        )
    st.markdown("##### Wichtigste Einflüsse (SHAP)")
    st.altair_chart(contributions_chart(prediction["top_features"]), width="stretch")
    st.caption(
        "Beiträge relativ zur mittleren Prognose; sie erklären das Modell, nicht die "
        "Betontechnologie."
    )
    metrics = model["test_metrics"]
    quality = (
        f" · Test-MAE {number(metrics['mae'], 2)} MPa auf ungesehenen Rezepturen"
        if "mae" in metrics
        else ""
    )
    st.caption(
        f"Modell {model['model_name']} · Version `{model['model_version']}`{quality} · "
        f"[Model Card]({DOCS_URL}{model['model_card']})"
    )
