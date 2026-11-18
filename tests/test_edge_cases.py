"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Ringe, Grenzen der Regler, ungültige Eingaben, periodische Ketten."""

import math

import numpy as np
import pytest

import msz_chain as K
import msz_constants as C
import msz_evaluation as ev
import msz_families as F
import msz_mixing as M
import msz_simulation as S


def test_ring_mit_drei_plaetzen_vier_zustaende_drei_maximalbelegungen():
    """n = 3: vier Zustände (leer und drei Einzelplätze); jeder Einzelplatz ist eine Maximalbelegung (drei gleichwertige); d(0) = 3/4 vom Einzelplatz aus."""
    ch = K.ring_chain(3, 1.0, "Heat-bath")
    assert ch.N == 4 and sorted(ch.teilchen) == [0, 1, 1, 1] and np.allclose(ch.pi, 1 / 4)
    sp = M.ring_spectrum(3, 1.0, "Heat-bath")
    assert M.d_t(sp, 0) == pytest.approx(3 / 4, abs=1e-13)
    tx = M.t_alle_starts(sp)
    assert len(set(tx[ch.teilchen == 1])) == 1 and tx.max() == M.t_mix(sp)


def test_ring_mit_weniger_als_drei_plaetzen_gibt_eine_meldung():
    for n in (0, 1, 2):
        with pytest.raises(ValueError, match="mindestens drei Plätze"):
            K.ring_chain(n, 1.0, "Heat-bath")


def test_ring_n12_ist_die_live_grenze_und_rechnet_in_zeit():
    assert K.lucas(12) == 322 and C.RING_N_MAX_LIVE == 12 and C.RING_N_MAX_TABLE == 16
    r = ev.exp1(ev.E1Settings(12, 20.0, "Heat-bath"))
    assert r.t_dbar is None and r.T_e is None and r.dbar_kurve is None            # d̄ nur bis n = 10 live
    assert r.t_d == 2276 and r.N == 322


def test_reglergrenzen_der_experimente_rechnen_ohne_fehler():
    for n in (C.RING_N_MIN, C.E1_DBAR_N_MAX_LIVE):
        for lam in (C.LAM_MIN, C.LAM_MAX):
            for kette in C.KETTEN:
                r = ev.exp1(ev.E1Settings(n, lam, kette))
                assert 1 <= r.t_d <= r.t_dbar <= r.t_d_halb
    r2 = ev.exp2(C.E2_N_MIN, C.LAM_MAX)
    assert r2.N == K.lucas(4) and r2.t_mix >= r2.t_leer


def test_schwelle_ein_halb_bis_null_komma_null_eins_rechnet():
    for name, eps in C.SCHWELLEN.items():
        r = ev.exp1(ev.E1Settings(6, 2.0, "Heat-bath", eps_name=name))
        assert r.t_eps_d == M.t_mix(M.ring_spectrum(6, 2.0, "Heat-bath"), eps) and r.t_eps_dbar is not None


def test_start_auswahl_leer_schlechtester_und_ein_zustand():
    for start in (C.START_LEER, C.START_SCHLECHTESTER, "{0, 2}"):
        r = ev.exp1(ev.E1Settings(5, 2.0, "Heat-bath", start))
        assert r.t_start_14 is not None
    assert ev.start_liste(4)[:2] == [C.START_SCHLECHTESTER, C.START_LEER] and len(ev.start_liste(4)) == 2 + 6
    with pytest.raises(ValueError):
        ev.exp1(ev.E1Settings(5, 2.0, "Heat-bath", "{0, 1}"))                     # keine unabhängige Menge: nicht in der Auswahl


def test_schwelle_eins_ist_sofort_erreicht():
    """ε = 1: jeder Abstand ist höchstens 1, also t_mix(1) = 0 und t_x = 0 für jeden Start."""
    sp = M.ring_spectrum(5, 1.0, "Heat-bath")
    assert M.t_mix(sp, 1.0) == 0 and M.t_start(sp, 0, 1.0) == 0 and (M.t_alle_starts(sp, 1.0) == 0).all()


def test_kreis_nicht_traege_gerades_n_hat_keine_mischzeit_und_keinen_einschluss():
    e = ev.einschluss_kreis(4, traege=False)
    assert e.t_mix is None and math.isinf(e.t_rel) and ev.verdict_einschluss_raum(e)[0] == "warning"
    assert ev.einschluss_kreis(5, traege=False).t_mix == 5


def test_kreis_kleinster_und_groesster_regler():
    assert F.kreis_t(C.KREIS_N_MIN) == 1 and F.kreis_t(C.KREIS_N_MAX) == 60718


def test_wuerfel_kleinster_und_groesster_regler():
    assert F.wuerfel_ts(C.WUERFEL_N_MIN, (0.25,))[0.25] is not None
    assert F.wuerfel_ts(C.WUERFEL_N_MAX, (0.25,))[0.25] == 4009


def test_wuerfel_n1_hat_zwei_zustaende():
    kurve = F.wuerfel_kurve(1, 10)
    assert kurve[0] == pytest.approx(0.5) and kurve[1] == pytest.approx(0.0, abs=1e-13)        # träge: halbe Masse bleibt, halbe geht über: nach einem Schritt gleich π = (1/2, 1/2)


def test_karten_ein_zwei_und_grosse_zahl():
    assert F.riffle_tv(1, 3) == 0.0 and F.riffle_t(1) == 0
    assert F.riffle_tv(2, 1) == pytest.approx(0.25) and F.riffle_t(2) == 1                    # zwei Karten: nach einer Mischung genau 1/4
    assert F.riffle_t(C.KARTEN_N_MAX) == 10
    assert len(ev.karten_tabelle(C.KARTEN_N_MAX)) == C.KARTEN_K_MAX
    with pytest.raises(ValueError):
        F.riffle_tv_bruch(0, 1)


def test_simulation_r_gleich_1_und_grosses_r_und_seed_rand():
    assert len(S.lauf(1, 0).tv) == C.SIM_T_MAX + 1
    r = ev.exp5(ev.E5Settings(C.SIM_R_LIVE[-1], "Teilchenzahl", C.SIM_REPS_MIN, C.SIM_SEED_MAX))
    assert len(r.schaetzungen) == C.SIM_REPS_MIN and r.zusammen["von"] == C.SIM_REPS_MIN


def test_einheiten_fuer_nicht_berechnete_werte():
    assert M.in_einheit(None, "Abfertigungsdauern", 5, 1.0, "Heat-bath") is None


def test_ungueltige_eingaben_geben_meldungen():
    with pytest.raises(ValueError):
        K.ring_chain(5, -1.0, "Heat-bath")
    with pytest.raises(ValueError):
        K.ring_chain(5, 1.0, "Kawasaki")
    with pytest.raises(ValueError):
        F.mischer_ts(4, "Faro")


def test_t_mix_bis_zur_rechengrenze_gibt_none_statt_unendlich_laufen():
    sp = M.ring_spectrum(4, 1.0, "Heat-bath")
    assert M._erstes_t(lambda t: 1.0, 0.25, grenze=64) is None and M.t_mix(sp, 0.25) == 12
