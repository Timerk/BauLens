"""Inspection report: combine the current results, pick a source, show the checked draft."""

import html
from datetime import date
from typing import Any

import streamlit as st

from baulens_dashboard.client import (
    ApiClient,
    ApiError,
    ComponentStatus,
    NumberVerdict,
    RecordedCase,
    Report,
)
from baulens_dashboard.labels import REPLAY_LABEL, SOURCES, STATUS, VERDICTS, number


def inspection_input(case_id: str) -> dict[str, Any] | None:
    """The ``InspectionInput`` built from the results currently shown in the other tabs."""
    case: dict[str, Any] = {"case_id": case_id}
    if "strength" in st.session_state:
        case["mix"] = st.session_state["mix"]
        case["strength"] = st.session_state["strength"]
    if "crack" in st.session_state:
        analysis = st.session_state["crack"]
        case["image"] = analysis["image"] | {"element": st.session_state.get("crack_element")}
        case["crack"] = analysis["prediction"]
    return case if len(case) > 1 else None


def highlighted(text: str, section: str, checks: list[NumberVerdict]) -> str:
    """HTML for a section with every checked number marked as verified or violating."""
    spans = sorted((c for c in checks if c["section"] == section), key=lambda c: c["start"])
    parts: list[str] = []
    cursor = 0
    for check in spans:
        if check["start"] < cursor or check["end"] > len(text):
            continue
        css = "bl-ok" if check["status"] == "ok" else "bl-bad"
        title = html.escape(f"{VERDICTS[check['status']]}: {check['reason']}", quote=True)
        parts.append(html.escape(text[cursor : check["start"]]))
        parts.append(
            f'<mark class="{css}" title="{title}">'
            f"{html.escape(text[check['start'] : check['end']])}</mark>"
        )
        cursor = check["end"]
    parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def _sources(llm_mode: str) -> list[str]:
    return ["template", "replay"] if llm_mode == "replay" else ["template", "replay", "live"]


def render(client: ApiClient, status: ComponentStatus | None, llm_mode: str) -> None:
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### Eingaben")
        case_id = st.text_input(
            "Vorgangsnummer", value=f"PB-{date.today():%Y%m%d}", max_chars=64
        ).strip()
        case = inspection_input(case_id)
        st.markdown(
            f"- Festigkeitsprognose: {'übernommen' if 'strength' in st.session_state else '—'}\n"
            f"- Rissbefund: {'übernommen' if 'crack' in st.session_state else '—'}"
        )
        source = st.radio(
            "Quelle des Berichtstexts",
            _sources(llm_mode),
            format_func=lambda key: SOURCES[key],
            horizontal=False,
        )
        if llm_mode == "replay":
            st.caption("Live-Generierung ist in dieser Demo deaktiviert.")
        recorded: RecordedCase | None = None
        if source == "replay":
            st.warning(REPLAY_LABEL, icon=":material/history:")
            recorded = _recorded_case(client, status)
        elif source == "live":
            st.caption("Ein lokales 3-4B-Modell braucht auf einer CPU 30–70 s pro Bericht.")
        payload = recorded["case"] if recorded is not None else case
        ready = status is not None and status["ready"]
        if not ready:
            st.info(
                "Das Berichtsmodul ist in dieser Version noch nicht verfügbar. "
                "Die Eingaben, die übergeben würden, stehen unten.",
                icon=":material/construction:",
            )
        elif payload is None:
            st.caption("Zuerst eine Festigkeitsprognose oder einen Rissbefund erstellen.")
        create = st.button(
            "Bericht erstellen", type="primary", disabled=not ready or payload is None
        )
        if payload is not None:
            with st.expander("Übergebene Daten (InspectionInput)"):
                st.json(payload)
    if create and payload is not None:
        model = recorded["models"][0] if recorded is not None and recorded["models"] else None
        with right, st.spinner("Bericht wird erstellt …"):
            try:
                st.session_state["report"] = client.report(payload, source, model)
            except ApiError as error:
                st.session_state.pop("report", None)
                right.error(f"Bericht fehlgeschlagen: {error.detail}")
    with right:
        _result()


def _recorded_case(client: ApiClient, status: ComponentStatus | None) -> RecordedCase | None:
    if status is None or not status["ready"]:
        return None
    try:
        cases = client.recorded_cases()
    except ApiError as error:
        st.error(f"Aufzeichnungen nicht abrufbar: {error.detail}")
        return None
    if not cases:
        st.caption("Keine Aufzeichnungen vorhanden.")
        return None
    st.caption(
        "Replays gibt es nur für aufgezeichnete Fälle; die eigenen Ergebnisse werden dafür "
        "nicht verwendet."
    )
    index = st.selectbox(
        "Aufgezeichneter Fall",
        range(len(cases)),
        format_func=lambda i: f"{cases[i]['case']['case_id']} ({', '.join(cases[i]['models'])})",
    )
    return cases[index]


def _result() -> None:
    st.markdown("##### Prüfberichtsentwurf")
    report: Report | None = st.session_state.get("report")
    if report is None:
        st.caption("Noch kein Bericht erstellt.")
        return
    if report["source"] == "replay":
        st.warning(REPLAY_LABEL, icon=":material/history:")
    status = STATUS.get(report["status"], report["status"])
    badge = "green" if report["status"] in {"accepted", "template"} else "orange"
    st.markdown(
        f":{badge}-badge[{status}] :gray-badge[{SOURCES[report['source']]}]"
        + (f" :gray-badge[{report['model']}]" if report["model"] else "")
        + f" :gray-badge[{number(report['latency_s'])} s]"
    )
    for section in report["sections"]:
        st.markdown(f"**{section['title']}**")
        st.markdown(
            f'<div class="bl-report">'
            f"{highlighted(section['text'], section['name'], report['checks'])}</div>",
            unsafe_allow_html=True,
        )
    st.caption(report["disclaimer"])
    st.caption(f"Herkunft: {report['provenance']}")
    if report["note"]:
        st.caption(report["note"])
    _checks(report)
    st.download_button(
        "Als Markdown herunterladen",
        report["markdown"],
        file_name=f"{report['case_id']}.md",
        mime="text/markdown",
    )


def _checks(report: Report) -> None:
    checks = report["checks"]
    verified = sum(check["status"] == "ok" for check in checks)
    with st.expander(f"Zahlenprüfung: {verified} von {len(checks)} Zahlen belegt"):
        st.dataframe(
            [
                {
                    "Zahl": check["text"],
                    "Abschnitt": check["section"],
                    "Ergebnis": VERDICTS[check["status"]],
                    "Belegt durch": ", ".join(check["matched_fields"]) or "—",
                }
                for check in checks
            ],
            hide_index=True,
            width="stretch",
        )
    rejected = report["rejected_violations"]
    if rejected:
        st.error(
            "Verworfener Modellentwurf – nicht belegte Zahlen:\n\n"
            + "\n".join(
                f"- „{item['text']}“ ({VERDICTS[item['status']]}): {item['reason']}"
                for item in rejected
            )
        )
