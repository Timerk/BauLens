"""Altair charts with the dashboard palette: one message per chart, no chart junk."""

from typing import cast

import altair as alt
import pandas as pd

from baulens_dashboard.client import Contribution, StrengthPrediction
from baulens_dashboard.labels import ACCENT, ACCENT_SOFT, FEATURES, NEGATIVE, NEUTRAL

_FONT = "sans-serif"


def _style[C: (alt.Chart, alt.LayerChart)](chart: C) -> C:
    # Altair's configure_* methods are untyped; they return a copy of the same chart type.
    styled = chart.configure_view(strokeWidth=0).configure_axis(
        labelFont=_FONT,
        titleFont=_FONT,
        labelColor=NEUTRAL,
        titleColor=NEUTRAL,
        gridColor="#E3E1DB",
        domain=False,
        tickColor="#E3E1DB",
    )
    return cast(C, styled)


def interval_chart(prediction: StrengthPrediction, axis_max: float = 100.0) -> alt.LayerChart:
    """Point estimate on a strength axis with the conformal interval as a band."""
    lower, upper = prediction["lower_mpa"], prediction["upper_mpa"]
    frame = pd.DataFrame(
        [{"y": "Prognose", "lower": lower, "upper": upper, "point": prediction["point_mpa"]}]
    )
    scale = alt.Scale(domain=[0, max(axis_max, upper * 1.1)], nice=False)
    base = alt.Chart(frame).encode(y=alt.Y("y:N", axis=None))
    band = base.mark_bar(height=20, color=ACCENT_SOFT, cornerRadius=10).encode(
        x=alt.X("lower:Q", scale=scale, title="Druckfestigkeit in MPa"), x2="upper:Q"
    )
    point = base.mark_point(filled=True, size=300, color=ACCENT, opacity=1).encode(
        x=alt.X("point:Q", scale=scale)
    )
    chart = alt.LayerChart(layer=[band, point]).properties(height=80)
    return _style(chart)


def contributions_chart(contributions: list[Contribution]) -> alt.Chart:
    """Signed SHAP contributions in MPa, largest magnitude first."""
    frame = pd.DataFrame(
        [
            {
                "feature": FEATURES.get(item["feature"], item["feature"]),
                "mpa": item["contribution_mpa"],
                "direction": "erhöht" if item["contribution_mpa"] >= 0 else "senkt",
            }
            for item in contributions
        ]
    )
    order = frame.reindex(frame["mpa"].abs().sort_values(ascending=False).index)["feature"]
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("mpa:Q", title="Beitrag zur Prognose in MPa"),
            y=alt.Y(
                "feature:N",
                sort=list(order),
                title=None,
                axis=alt.Axis(labelLimit=240, labelOverlap=False, labelPadding=6),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["erhöht", "senkt"], range=[ACCENT, NEGATIVE]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Merkmal"),
                alt.Tooltip("mpa:Q", title="MPa", format="+.2f"),
            ],
        )
        .properties(height=alt.Step(30))
    )
    return _style(cast(alt.Chart, chart))


def probability_chart(probability: float, threshold: float) -> alt.LayerChart:
    """Crack probability on [0, 1] with the validation threshold as a rule."""
    frame = pd.DataFrame([{"y": "P(Riss)", "p": probability, "t": threshold}])
    scale = alt.Scale(domain=[0, 1])
    base = alt.Chart(frame).encode(y=alt.Y("y:N", axis=None))
    track = (
        alt.Chart(pd.DataFrame([{"y": "P(Riss)", "a": 0.0, "b": 1.0}]))
        .mark_bar(height=14, color="#E3E1DB", cornerRadius=7)
        .encode(y=alt.Y("y:N", axis=None), x=alt.X("a:Q", scale=scale), x2="b:Q")
    )
    bar = base.mark_bar(height=14, cornerRadius=7, color=ACCENT).encode(
        x=alt.X("p:Q", scale=scale, title="Risswahrscheinlichkeit", axis=alt.Axis(format="%")),
    )
    tick = base.mark_tick(color=NEGATIVE, thickness=3, size=30).encode(x="t:Q")
    chart = alt.LayerChart(layer=[track, bar, tick]).properties(height=70)
    return _style(chart)
