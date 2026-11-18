"""Presets: Vollständigkeit, gültige Werte, Standard-Preset gleich Standardeinstellungen; Permalink begrenzt und liest nur erlaubte Werte."""

import msz_constants as C
import msz_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name] and set(preset) <= set(P.SETTING_SPECS), name
        assert "seed_input" not in preset                                   # der Zufalls-Seed gilt für alle Presets


def test_standard_preset_equals_the_default_settings():
    standard = C.PRESETS[C.STANDARD_PRESET]
    for key, wert in standard.items():
        assert wert == P.SETTING_SPECS[key].default, key
    assert C.STANDARD_PRESET in C.PRESETS


def test_setting_specs_cover_every_widget_key_of_the_app():
    erwartet = {"seed_input", "e1_n", "e1_lam", "e1_kette", "e1_start", "e1_eps", "e1_unit", "e2_n", "e2_lam", "e3_n", "e3_lam", "e3_kette", "e3_kreis", "e3_traege", "e3_wuerfel",
                "e4_kreis", "e4_wuerfel", "e4_ring", "e4_karten", "e5_r", "e5_kennzahl", "e5_reps"}
    assert set(P.SETTING_SPECS) == erwartet
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)        # keine doppelten Adressparameter


def test_defaults_liegen_innerhalb_der_grenzen():
    for key, spec in P.SETTING_SPECS.items():
        if spec.kind in ("int", "float"):
            assert spec.lo <= spec.default <= spec.hi, key
        elif spec.kind == "wahl":
            assert spec.default in spec.erlaubt, key
        elif spec.kind == "liste":
            assert set(spec.default) <= set(spec.erlaubt), key


def test_permalink_wert_wird_begrenzt_und_nur_erlaubtes_gelesen():
    spec_n, spec_l, spec_w, spec_liste = (P.SETTING_SPECS[k] for k in ("e1_n", "e1_lam", "e1_kette", "e4_kreis"))
    assert P._lesen(spec_n, "99") == C.RING_N_MAX_LIVE and P._lesen(spec_n, "-4") == C.RING_N_MIN and P._lesen(spec_n, "abc") is None
    assert P._lesen(spec_l, "2.26") == 2.5 and P._lesen(spec_l, "500") == C.LAM_MAX and P._lesen(spec_l, "0") == C.LAM_MIN
    assert P._lesen(spec_w, "Metropolis") == "Metropolis" and P._lesen(spec_w, "Kawasaki") is None
    assert P._lesen(spec_liste, "25,100,7") == [25, 100] and P._lesen(spec_liste, "7") is None
    assert P._lesen(P.SETTING_SPECS["e5_r"], "300") == 300 and P._lesen(P.SETTING_SPECS["e5_r"], "301") is None


def test_listenwerte_werden_beim_vergleich_gleich_behandelt():
    assert P._norm([1, 2]) == P._norm((1, 2)) == (1, 2) and P._norm("x") == "x"
