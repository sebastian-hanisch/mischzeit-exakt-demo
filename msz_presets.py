"""Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) und Zufalls-Seed-Button."""

import random
from dataclasses import dataclass

import streamlit as st

import msz_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    kind: str                       # "int", "float", "text", "wahl", "liste"
    default: object
    lo: float | None = None
    hi: float | None = None
    erlaubt: tuple = ()             # erlaubte Werte bei „wahl“ und „liste“


def _specs() -> dict:
    d = {
        "seed_input": SettingSpec("seed", "int", C.SIM_SEED_DEFAULT, 0, C.SIM_SEED_MAX),
        "e1_n": SettingSpec("e1n", "int", C.DEFAULT_N, C.RING_N_MIN, C.RING_N_MAX_LIVE),
        "e1_lam": SettingSpec("e1l", "float", C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
        "e1_kette": SettingSpec("e1k", "wahl", "Heat-bath", erlaubt=C.KETTEN),
        "e1_start": SettingSpec("e1s", "text", C.START_SCHLECHTESTER),
        "e1_eps": SettingSpec("e1e", "wahl", C.SCHWELLE_STANDARD, erlaubt=tuple(C.SCHWELLEN)),
        "e1_unit": SettingSpec("e1u", "wahl", "Schritte", erlaubt=C.EINHEITEN),
        "e2_n": SettingSpec("e2n", "int", C.E2_N_MAX, C.E2_N_MIN, C.E2_N_MAX),
        "e2_lam": SettingSpec("e2l", "float", C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
        "e3_n": SettingSpec("e3n", "int", C.DEFAULT_N, C.RING_N_MIN, C.RING_N_MAX_LIVE),
        "e3_lam": SettingSpec("e3l", "float", C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
        "e3_kette": SettingSpec("e3k", "wahl", "Heat-bath", erlaubt=C.KETTEN),
        "e3_kreis": SettingSpec("e3kr", "int", C.KREIS_DEFAULT, C.KREIS_N_MIN, C.KREIS_N_MAX),
        "e3_traege": SettingSpec("e3t", "wahl", "träge", erlaubt=("träge", "nicht träge")),
        "e3_wuerfel": SettingSpec("e3w", "int", C.WUERFEL_DEFAULT, C.WUERFEL_N_MIN, C.WUERFEL_N_MAX),
        "e4_kreis": SettingSpec("e4kr", "liste", tuple(C.KREIS_N_STANDARD), erlaubt=C.KREIS_N_AUSWAHL),
        "e4_wuerfel": SettingSpec("e4w", "liste", tuple(C.WUERFEL_N_STANDARD), erlaubt=C.WUERFEL_N_AUSWAHL),
        "e4_ring": SettingSpec("e4r", "liste", tuple(C.RING_FENSTER_N_STANDARD), erlaubt=C.RING_FENSTER_N),
        "e4_karten": SettingSpec("e4ka", "int", C.KARTEN_DEFAULT, C.KARTEN_N_MIN, C.KARTEN_N_MAX),
        "e5_r": SettingSpec("e5r", "wahl", C.SIM_R_DEFAULT, erlaubt=C.SIM_R_LIVE),
        "e5_kennzahl": SettingSpec("e5k", "wahl", "ganze Verteilung", erlaubt=C.KENNZAHLEN),
        "e5_reps": SettingSpec("e5w", "int", C.SIM_REPS_DEFAULT, C.SIM_REPS_MIN, C.SIM_REPS_MAX),
    }
    return d


SETTING_SPECS = _specs()


def init_session_state_defaults():
    for key, spec in SETTING_SPECS.items():
        if key not in st.session_state:
            st.session_state[key] = list(spec.default) if spec.kind == "liste" else spec.default


def bounds(key):
    spec = SETTING_SPECS[key]
    return spec.lo, spec.hi


def _lesen(spec: SettingSpec, roh: str):
    """Wert aus der Adresszeile, begrenzt; None, wenn er nicht lesbar ist."""
    try:
        if spec.kind == "int":
            return min(spec.hi, max(spec.lo, int(roh)))
        if spec.kind == "float":
            return min(spec.hi, max(spec.lo, round(float(roh) * 2) / 2))
        if spec.kind == "wahl":
            wert = type(spec.erlaubt[0])(roh)
            return wert if wert in spec.erlaubt else None
        if spec.kind == "liste":
            werte = [int(x) for x in roh.split(",") if x.strip()]
            return [w for w in werte if w in spec.erlaubt] or None
        return str(roh)
    except (ValueError, TypeError):
        return None


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            wert = _lesen(spec, qp[spec.url_param])
            if wert is not None:
                st.session_state[key] = wert
    st.session_state["permalink_loaded"] = True


def sync_query_params(werte: dict):
    """`werte`: {Schlüssel: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for key, wert in werte.items():
            spec = SETTING_SPECS[key]
            st.query_params[spec.url_param] = ",".join(str(w) for w in wert) if isinstance(wert, (list, tuple)) else str(wert)
    except Exception:
        pass


def _norm(wert):
    return tuple(wert) if isinstance(wert, (list, tuple)) else wert


def apply_preset(name):
    """Setzt jeden Regler auf den Wert des Presets, nicht genannte Regler auf ihren Standard (so bleibt die Hervorhebung eindeutig)."""
    preset = C.PRESETS[name]
    for key, spec in SETTING_SPECS.items():
        if key == "seed_input":
            continue
        wert = preset.get(key, spec.default)
        st.session_state[key] = list(wert) if spec.kind == "liste" else wert


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Der Zufalls-Seed gehört nicht dazu (er gilt für alle Presets). Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(
            _norm(st.session_state.get(key, spec.default)) == _norm(preset.get(key, spec.default))
            for key, spec in SETTING_SPECS.items() if key != "seed_input"
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SIM_SEED_MAX)
