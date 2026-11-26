"""German UI labels and the shared palette. Code identifiers stay English."""

DISCLAIMER = (
    "Entscheidungsunterstützung, kein Prüfergebnis: Die Modelle ersetzen weder die "
    "Druckfestigkeitsprüfung nach DIN EN 12390-3 noch die Begutachtung durch eine "
    "fachkundige Person."
)
REPLAY_LABEL = "Aufgezeichnete Ausgabe – keine Live-Generierung"
DOCS_URL = "https://github.com/Timerk/BauLens/blob/main/"
DATASET_ATTRIBUTION = (
    "Beispielbilder: Ç. F. Özgenel (2019), *Concrete Crack Images for Classification*, "
    "Mendeley Data, [doi:10.17632/5y9wdsg2zt.2](https://doi.org/10.17632/5y9wdsg2zt.2), "
    "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Alle vier stammen aus dem "
    "zurückgehaltenen Testsplit."
)

MIX_FIELDS: dict[str, str] = {
    "cement": "Zement",
    "blast_furnace_slag": "Hüttensand",
    "fly_ash": "Flugasche",
    "water": "Wasser",
    "superplasticizer": "Fließmittel",
    "coarse_aggregate": "Grobe Körnung",
    "fine_aggregate": "Feine Körnung",
    "age_days": "Prüfalter",
}
FEATURES: dict[str, str] = MIX_FIELDS | {
    "water_cement": "w/z-Wert",
    "water_binder": "w/b-Wert",
    "log_age": "log(Prüfalter)",
    "scm_fraction": "Anteil Zusatzstoffe",
    "superplasticizer_binder": "Fließmittel/Bindemittel",
    "coarse_fine_aggregate": "Grob-/Feinanteil",
    "aggregate_binder": "Gesteinskörnung/Bindemittel",
    "total_mass": "Gesamtmasse",
}
# A typical C30/37-like mix inside the UCI training ranges.
DEFAULT_MIX: dict[str, float] = {
    "cement": 380.0,
    "blast_furnace_slag": 95.0,
    "fly_ash": 0.0,
    "water": 185.0,
    "superplasticizer": 4.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 760.0,
    "age_days": 28,
}
SOURCES: dict[str, str] = {
    "live": "Live (lokales Sprachmodell)",
    "replay": "Replay (aufgezeichnet)",
    "template": "Vorlage (ohne Sprachmodell)",
}
STATUS: dict[str, str] = {
    "accepted": "Modellentwurf geprüft und übernommen",
    "rejected": "Modellentwurf verworfen (Zahlen nicht belegt) – Vorlage ausgeliefert",
    "invalid": "Modellantwort ungültig – Vorlage ausgeliefert",
    "error": "Sprachmodell nicht erreichbar – Vorlage ausgeliefert",
    "replay_miss": "Keine Aufzeichnung für diese Eingabe – Vorlage ausgeliefert",
    "template": "Vorlagenbericht",
}
VERDICTS: dict[str, str] = {
    "ok": "belegt",
    "invented": "erfunden",
    "misattributed": "falsch zugeordnet",
    "inexact": "ungenau",
    "unbound": "nicht zuordenbar",
}

# Palette: one calm accent, a warm neutral and two signal colours.
ACCENT = "#1F5F6B"
ACCENT_SOFT = "#A9C8CD"
NEGATIVE = "#B5653A"
WARNING = "#B7791F"
DANGER = "#A63D3D"
NEUTRAL = "#6B6F72"


def number(value: float, digits: int = 1) -> str:
    """German decimal comma, e.g. 51,2."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",").replace(" ", ".")
