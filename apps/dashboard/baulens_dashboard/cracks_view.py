"""Crack image upload or example selection, probability against threshold, Grad-CAM."""

import base64
from pathlib import Path

import streamlit as st

from baulens_dashboard.charts import probability_chart
from baulens_dashboard.client import ApiClient, ApiError, CrackModel
from baulens_dashboard.labels import DATASET_ATTRIBUTION, DOCS_URL, number

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = {
    "riss_16450_1.jpg": "Breiter Riss",
    "riss_06611.jpg": "Feiner Riss",
    "ohne_riss_00346.jpg": "Raue Oberfläche",
    "ohne_riss_16584.jpg": "Helle Einschlüsse",
}
_STRATEGIES = {"f1": "maximales F1", "youden": "Youden-Index"}


def _choose_image() -> tuple[bytes, str, str] | None:
    source = st.radio(
        "Bildquelle",
        ["Beispielbild", "Eigenes Bild"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if source == "Eigenes Bild":
        upload = st.file_uploader(
            "JPEG oder PNG, höchstens 10 MB",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        if upload is None:
            return None
        return upload.getvalue(), upload.name, upload.type or "application/octet-stream"
    columns = st.columns(len(EXAMPLES))
    for column, (name, caption) in zip(columns, EXAMPLES.items(), strict=True):
        column.image(str(EXAMPLES_DIR / name), caption=caption, width="stretch")
    choice = st.selectbox(
        "Beispiel", list(EXAMPLES), format_func=lambda name: EXAMPLES[name], key="example"
    )
    st.caption(DATASET_ATTRIBUTION)
    return (EXAMPLES_DIR / choice).read_bytes(), choice, "image/jpeg"


def render(client: ApiClient, model: CrackModel | None) -> None:
    if model is None:
        st.info(
            "Das Rissmodell ist nicht geladen. Mit `make cracks` trainieren und exportieren "
            "und die API neu starten (Details in der Seitenleiste)."
        )
        return
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### Oberflächenbild")
        chosen = _choose_image()
        element = st.text_input(
            "Bauteil (optional, für den Bericht)",
            max_chars=120,
            placeholder="z. B. Stütze B2, Achse 4",
        )
        analyse = st.button("Bild analysieren", type="primary", disabled=chosen is None)
    if analyse and chosen is not None:
        content, filename, content_type = chosen
        try:
            result = client.predict_cracks(content, filename, content_type, gradcam=True)
        except ApiError as error:
            st.session_state.pop("crack", None)
            left.error(f"Analyse fehlgeschlagen: {error.detail}")
        else:
            st.session_state["crack"] = result
            st.session_state["crack_image"] = content
            st.session_state["crack_element"] = element.strip() or None
    with right:
        _result(model)


def _result(model: CrackModel) -> None:
    st.markdown("##### Befund")
    analysis = st.session_state.get("crack")
    if analysis is None:
        st.caption("Bild wählen und analysieren.")
        return
    prediction = analysis["prediction"]
    crack = prediction["label"] == "crack"
    probability = prediction["crack_probability"]
    if crack:
        st.markdown(f":red-badge[:material/report: Riss erkannt] P = {number(probability, 3)}")
    else:
        st.markdown(
            f":green-badge[:material/check: Kein Riss erkannt] P = {number(probability, 3)}"
        )
    st.altair_chart(probability_chart(probability, prediction["threshold"]), width="stretch")
    strategy = _STRATEGIES.get(model["threshold_strategy"] or "", "auf der Validierung")
    st.caption(
        f"Markierung: Entscheidungsschwelle {number(prediction['threshold'], 3)}, auf den "
        f"Validierungsdaten gewählt ({strategy})."
    )
    original, overlay = st.columns(2)
    original.image(st.session_state["crack_image"], caption="Original", width="stretch")
    if analysis["gradcam_png_base64"]:
        overlay.image(
            base64.b64decode(analysis["gradcam_png_base64"]),
            caption="Grad-CAM: entscheidungsrelevante Bereiche",
            width="stretch",
        )
    image = analysis["image"]
    size = analysis["input_size_px"]
    st.caption(
        f"{image['width_px']} × {image['height_px']} px, für das Modell auf {size} × {size} px "
        "skaliert. Grad-CAM zeigt, worauf das Modell reagiert, und ist keine Rissvermessung. "
        "Das Modell klassifiziert Bildausschnitte; Rissbreite und -tiefe werden nicht bestimmt."
    )
    metrics = model["test_metrics"]
    quality = f" · Test-F1 {number(metrics['f1'], 3)}" if "f1" in metrics else ""
    st.caption(
        f"Modell {model['model_name']} ({model['backend']}) · Version "
        f"`{model['model_version']}`{quality} · [Model Card]({DOCS_URL}{model['model_card']})"
    )
