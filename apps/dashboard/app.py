"""BauLens dashboard: run with ``streamlit run app.py`` from this directory."""

import streamlit as st

from baulens_dashboard import cracks_view, report_view, strength_view
from baulens_dashboard.client import ApiClient, ApiError, Health, Models, default_client
from baulens_dashboard.labels import ACCENT, DANGER, DISCLAIMER, DOCS_URL

st.set_page_config(page_title="BauLens", page_icon=":material/foundation:", layout="wide")

st.html(
    f"""
    <style>
      .block-container {{ max-width: 1240px; padding-top: 2.2rem; }}
      h1 {{ font-weight: 650; letter-spacing: -0.01em; }}
      .bl-disclaimer {{ border-left: 3px solid {ACCENT}; padding: .55rem .9rem;
        background: #ECEAE4; border-radius: .3rem; font-size: .9rem; color: #3A4043; }}
      .bl-report {{ line-height: 1.6; margin-bottom: .8rem; }}
      mark.bl-ok {{ background: #DCEBE7; border-bottom: 2px solid {ACCENT}; padding: 0 .1em; }}
      mark.bl-bad {{ background: #F4DCDC; border-bottom: 2px solid {DANGER}; padding: 0 .1em; }}
    </style>
    """
)


def _status(client: ApiClient) -> tuple[Health | None, Models | None, str | None]:
    try:
        return client.health(), client.models(), None
    except ApiError as error:
        return None, None, error.detail


def _sidebar(client: ApiClient, health: Health | None, problem: str | None) -> None:
    with st.sidebar:
        st.markdown("#### Systemstatus")
        if health is None:
            st.error(problem or "API nicht erreichbar", icon=":material/cloud_off:")
            st.caption(f"Erwartet unter `{client.base_url}` (Variable `BAULENS_API_URL`).")
            return
        names = {"strength": "Festigkeit", "cracks": "Risserkennung", "report": "Berichte"}
        for key, label in names.items():
            component = health["components"][key]
            icon = ":material/check_circle:" if component["ready"] else ":material/block:"
            st.markdown(f"{icon} **{label}**")
            if component["ready"] and component["model_version"]:
                st.caption(f"{component['model_name']} · `{component['model_version']}`")
            elif not component["ready"]:
                st.caption(component["detail"])
        mode = "Demo (nur Replay)" if health["llm_mode"] == "replay" else "Live erlaubt"
        st.caption(f"Sprachmodell: {mode} · API {health['version']}")
        st.divider()
        st.markdown(
            f"[Architektur]({DOCS_URL}docs/architecture.md) · "
            f"[API-Dokumentation]({client.public_url}/api/v1/docs)"
        )


client = default_client()
health, models, problem = _status(client)

st.title("BauLens")
st.caption("KI-gestützte Auswertung von Betonrezepturen und Oberflächenbildern")
st.html(f'<div class="bl-disclaimer">{DISCLAIMER}</div>')
_sidebar(client, health, problem)

if health is None or models is None:
    st.error(
        "Die BauLens-API ist nicht erreichbar. Starten mit `uv run baulens api serve` oder "
        "`docker compose up`.",
        icon=":material/cloud_off:",
    )
    st.stop()

strength_tab, cracks_tab, report_tab = st.tabs(["Betonrezeptur", "Rissbild", "Prüfbericht"])
with strength_tab:
    strength_view.render(client, models["strength"])
with cracks_tab:
    cracks_view.render(client, models["cracks"])
with report_tab:
    report_view.render(client, health["components"].get("report"), health["llm_mode"])
