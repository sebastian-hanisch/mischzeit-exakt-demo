"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen
und prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig
identischen Inhalt rendern und kollidieren) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen.

Eine einzige App-Instanz für das ganze Modul (die Zwischenspeicher der Rechnungen bleiben erhalten); nach jedem Fall wird das Standard-Preset geklickt."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import msz_constants as C
import msz_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _zurueck(at):
    at.sidebar.button(key=f"preset_{C.STANDARD_PRESET}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_app_loads_without_exception(app):
    assert not app.exception
    assert len(app.subheader) >= 6 and not app.error


def test_every_preset_click_does_not_raise(app):
    for name in C.PRESETS:
        app.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not app.exception, (name, [str(e) for e in app.exception])
    _zurueck(app)


def test_seed_button_does_not_raise(app):
    app.sidebar.button[-1].click().run(timeout=TIMEOUT)      # der letzte Button der Seitenleiste würfelt den Zufalls-Seed
    assert not app.exception, [str(e) for e in app.exception]
    assert 0 <= app.sidebar.number_input(key="seed_input").value <= C.SIM_SEED_MAX
    _zurueck(app)


@pytest.mark.parametrize("key", [k for k, s in P.SETTING_SPECS.items() if s.kind in ("int", "float") and k != "seed_input"])
def test_every_slider_at_its_min_and_max_does_not_raise(app, key):
    spec = P.SETTING_SPECS[key]
    for kante in (spec.lo, spec.hi):
        app.slider(key=key).set_value(kante).run(timeout=TIMEOUT)
        assert not app.exception, (key, kante, [str(e) for e in app.exception])
    _zurueck(app)


@pytest.mark.parametrize("key", [k for k, s in P.SETTING_SPECS.items() if s.kind == "wahl"])
def test_every_option_of_every_choice_widget_does_not_raise(app, key):
    spec = P.SETTING_SPECS[key]
    for option in spec.erlaubt:
        for art in ("radio", "select_slider", "selectbox"):
            try:
                widget = getattr(app, art)(key=key)
            except KeyError:
                continue
            widget.set_value(option).run(timeout=TIMEOUT)
            assert not app.exception, (key, option, [str(e) for e in app.exception])
            break
    _zurueck(app)


@pytest.mark.parametrize("key", [k for k, s in P.SETTING_SPECS.items() if s.kind == "liste"])
def test_multiselect_empty_and_full_does_not_raise(app, key):
    spec = P.SETTING_SPECS[key]
    for werte in ([], list(spec.erlaubt), list(spec.erlaubt)[:1]):
        app.multiselect(key=key).set_value(werte).run(timeout=TIMEOUT)
        assert not app.exception, (key, werte, [str(e) for e in app.exception])
    _zurueck(app)


def test_start_selectbox_every_state_of_a_small_ring_does_not_raise(app):
    app.slider(key="e1_n").set_value(4).run(timeout=TIMEOUT)
    for option in app.selectbox(key="e1_start").options:
        app.selectbox(key="e1_start").set_value(option).run(timeout=TIMEOUT)
        assert not app.exception, (option, [str(e) for e in app.exception])
    _zurueck(app)


def test_start_wird_zurueckgesetzt_wenn_n_die_auswahl_ungueltig_macht(app):
    app.slider(key="e1_n").set_value(6).run(timeout=TIMEOUT)
    app.selectbox(key="e1_start").set_value("{0, 2, 4}").run(timeout=TIMEOUT)
    app.slider(key="e1_n").set_value(5).run(timeout=TIMEOUT)         # {0, 2, 4} gibt es beim 5er-Ring nicht (0 und 4 sind Nachbarn)
    assert not app.exception and app.selectbox(key="e1_start").value == C.START_SCHLECHTESTER
    _zurueck(app)


def test_kreis_nicht_traege_gerades_n_zeigt_warnung_statt_fehler(app):
    app.slider(key="e3_kreis").set_value(100).run(timeout=TIMEOUT)
    app.radio(key="e3_traege").set_value("nicht träge").run(timeout=TIMEOUT)
    assert not app.exception and any("periodisch" in w.value for w in app.warning)
    _zurueck(app)
