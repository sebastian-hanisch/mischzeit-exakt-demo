"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben, Regler der Experimente im Hauptbereich.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset; der Zufalls-Seed gilt für alle Presets und gehört nicht dazu).
Der Test unten stellt sicher, dass wirklich JEDER Regler dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import msz_constants as C
import msz_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input", "multiselect")


def _neu() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


@pytest.fixture(scope="module")
def app():
    return _neu()


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht auf der Seite")


def _standard(at):
    at.sidebar.button(key=f"preset_{C.STANDARD_PRESET}").click().run()
    assert not at.exception, [str(e) for e in at.exception]


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich(app):
    assert [b.label for b in app.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in app.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf(app):
    assert "st-key-kopf" in " ".join(h.proto.body for h in app.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert app.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben(app):
    typen = _typen(app)
    assert [name for name, typ in typen.items() if typ == "primary"] == [C.STANDARD_PRESET]


def test_regler_der_experimente_stehen_im_hauptbereich_die_seitenleiste_hat_nur_den_seed(app):
    assert [w.key for art in REGLER_ARTEN for w in getattr(app.sidebar, art)] == ["seed_input"]
    main_keys = {w.key for art in REGLER_ARTEN for w in getattr(app.main, art) if w.key}
    assert main_keys == set(P.SETTING_SPECS) - {"seed_input"}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(app, name):
    app.sidebar.button(key=f"preset_{name}").click().run()
    assert not app.exception, [str(e) for e in app.exception]
    assert [n for n, typ in _typen(app).items() if typ == "primary"] == [name]
    app.run()
    assert [n for n, typ in _typen(app).items() if typ == "primary"] == [name]
    _standard(app)


def test_der_zufalls_seed_hebt_die_hervorhebung_nicht_auf(app):
    _standard(app)
    app.sidebar.number_input(key="seed_input").set_value(777).run()
    assert [n for n, typ in _typen(app).items() if typ == "primary"] == [C.STANDARD_PRESET]
    app.sidebar.number_input(key="seed_input").set_value(C.SIM_SEED_DEFAULT).run()


def _anderer_wert(key, regler, spec):
    """Ein Wert, der vom aktuellen abweicht und in keinem Preset für diesen Regler vorkommt (sonst träfe der geänderte Zustand zufällig ein anderes Preset)."""
    belegt = {P._norm(p[key]) for p in C.PRESETS.values() if key in p} | {P._norm(regler.value)}
    if spec.kind in ("int", "float"):
        kandidaten = [spec.hi, spec.lo, (spec.lo + spec.hi) // 2 if spec.kind == "int" else spec.lo + 1.0]
    elif spec.kind == "wahl":
        kandidaten = list(spec.erlaubt)
    else:
        kandidaten = [list(regler.value)[:-1], list(regler.value) + [o for o in spec.erlaubt if o not in regler.value][:1], [spec.erlaubt[0]]]
    return next(k for k in kandidaten if P._norm(k) not in belegt)


@pytest.mark.parametrize("key", [k for k in P.SETTING_SPECS if k not in ("seed_input", "e1_start")])
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(app, key):
    _standard(app)
    spec = P.SETTING_SPECS[key]
    regler = _regler(app, key)
    regler.set_value(_anderer_wert(key, regler, spec)).run()
    assert not app.exception, [str(e) for e in app.exception]
    assert "primary" not in _typen(app).values(), key
    _standard(app)                                       # erneutes Anklicken markiert wieder
    assert _typen(app)[C.STANDARD_PRESET] == "primary"


def test_aenderung_des_starts_hebt_die_hervorhebung_auf(app):
    _standard(app)
    start = app.selectbox(key="e1_start")
    start.set_value(C.START_LEER).run()
    assert not app.exception and "primary" not in _typen(app).values()
    _standard(app)
