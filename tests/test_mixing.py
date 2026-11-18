"""Abstand, Mischzeiten, Schranken: Handrechnung für den Ring mit 4 Plätzen, λ = 1 (7 Zustände) und Verhalten an den Rändern."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import msz_chain as K
import msz_mixing as M

E = 1.0 / math.e


def test_abstand_d_von_hand(ring4_spektrum):
    sp = ring4_spektrum
    # d(0) = 1 − 1/7; aus {0}: ½[(3/4 − 1/7) + 2·(1/8 − 1/7) in Beträgen + 4·(1/7)] = 17/28
    assert M.d_t(sp, 0) == pytest.approx(6 / 7, abs=1e-13)
    assert M.d_t(sp, 1) == pytest.approx(17 / 28, abs=1e-13)
    assert M.d_t(sp, 2) == pytest.approx(121 / 224, abs=1e-13)
    assert M.d_t(sp, 3) == pytest.approx(449 / 896, abs=1e-13)
    assert M.d_t(sp, 11) == pytest.approx(0.2631, abs=5e-5) and M.d_t(sp, 12) == pytest.approx(0.2435, abs=5e-5)


def test_aus_einzelplatz_von_hand():
    """TV aus {0} nach einem Schritt: Masse 3/4 auf {0}, 1/8 auf ∅ und 1/8 auf {0, 2}, gegen 1/7 je Zustand."""
    manuell = Fr(1, 2) * ((Fr(3, 4) - Fr(1, 7)) + 2 * (Fr(1, 8) - Fr(1, 7)).__abs__() + 4 * Fr(1, 7))
    assert manuell == Fr(17, 28)


def test_drei_mischzeiten_der_einen_kette(ring4_spektrum):
    sp = ring4_spektrum
    assert M.t_mix(sp, 0.25) == 12                       # d, 1/4
    assert M.t_mix_dbar(sp, 0.25) == 21                  # d̄, 1/4
    assert M.t_mix_dbar(sp, E) == 16                     # d̄, 1/e (Propp/Wilson)
    assert M.t_mix(sp, E) == 7                           # d, 1/e
    assert M.d_t(sp, 11) > 0.25 >= M.d_t(sp, 12)
    assert M.dbar_t(sp, 20) > 0.25 >= M.dbar_t(sp, 21)
    assert M.dbar_t(sp, 15) == pytest.approx(0.3858, abs=5e-5) and M.dbar_t(sp, 16) == pytest.approx(0.3575, abs=5e-5)


def test_schwelle_umrechnung_von_hand(ring4_spektrum):
    ts = [M.t_mix(ring4_spektrum, 2.0 ** -k) for k in range(1, 6)]
    assert ts == [4, 12, 21, 30, 39]
    assert all(t <= k * 12 for k, t in zip(range(1, 6), ts))       # ⌈log2(1/ε)⌉·t_mix(1/4)


def test_bester_und_schlechtester_start_von_hand(ring4_spektrum):
    sp = ring4_spektrum
    ch = K.ring_chain(4, 1.0, "Heat-bath")
    assert M.d_start_t(sp, ch.index(0), 1) == pytest.approx(5 / 14, abs=1e-13)        # leer ist der beste Start
    tx = M.t_alle_starts(sp)
    assert tx[ch.index(0)] == min(tx) and tx[ch.index(0b0101)] == tx[ch.index(0b1010)] == max(tx) == 12


def test_d_bar_gegen_zwei_d(ring4_spektrum):
    sp = ring4_spektrum
    for t in range(0, 61):
        d, db = M.d_t(sp, t), M.dbar_t(sp, t)
        assert d - 1e-12 <= db <= 2 * d + 1e-12
    assert M.dbar_t(sp, 60) / M.d_t(sp, 60) > 1.99


def test_spektrum_von_hand(ring4_spektrum):
    sp = ring4_spektrum
    soll = sorted([1.0, 0.75, 0.75, 0.75 + math.sqrt(2) / 8, 0.75 - math.sqrt(2) / 8, 0.5 + math.sqrt(2) / 8, 0.5 - math.sqrt(2) / 8], reverse=True)
    assert np.allclose(sp.mu, soll, atol=1e-12)
    # 2×2-Matrix der langsamen Mode bei λ = 1 (p = 1/2): Spur 2p − 3 = −2, Determinante 2q² = 1/2
    p = Fr(1, 2)
    spur, det = (2 * p - 1) + (-2), (2 * p - 1) * (-2) - (-p) * (2 * p)
    assert spur == -2 and det == Fr(1, 2)
    g = 1 - 1 / math.sqrt(2)
    assert M.g_gerades_n(1.0) == pytest.approx(g, abs=1e-15) and sp.mu[1] == pytest.approx(1 - g / 4, abs=1e-12)


def test_t_rel_und_satz_12_4_von_hand(ring4_spektrum):
    sp = ring4_spektrum
    trel = M.t_rel(sp.mu)
    assert trel == pytest.approx(4 / (1 - 1 / math.sqrt(2)), abs=1e-9) and trel == pytest.approx(13.657, abs=5e-4)
    unten, oben = M.schranken(trel, float(sp.pi.min()))
    assert unten == pytest.approx(8.7727, abs=1e-3) and oben == pytest.approx(45.507, abs=2e-3)
    assert unten <= 12 <= oben and oben / 12 == pytest.approx(3.79, abs=0.01) and 12 / unten == pytest.approx(1.368, abs=0.002)
    assert M.absolute_luecke(sp.mu) == pytest.approx(1 - sp.mu[1], abs=1e-15)         # kleinster Eigenwert 0,3232 > 0: absolute Lücke = Lücke


def test_einheiten():
    assert M.in_einheit(12, "Schritte", 4, 1.0, "Heat-bath") == 12.0
    assert M.in_einheit(12, "Durchläufe (n Schritte)", 4, 1.0, "Heat-bath") == 3.0
    assert M.in_einheit(12, "Abfertigungsdauern", 4, 1.0, "Heat-bath") == 1.5
    assert M.in_einheit(None, "Schritte", 4, 1.0, "Heat-bath") is None
    # n = 10, λ = 2: 73 Schritte (Heat-bath) = 2,433, 49 Schritte (Metropolis) = 2,450 Abfertigungsdauern
    assert M.in_einheit(73, "Abfertigungsdauern", 10, 2.0, "Heat-bath") == pytest.approx(73 / 30)
    assert M.in_einheit(49, "Abfertigungsdauern", 10, 2.0, "Metropolis") == pytest.approx(2.45)


def test_zeitraster_ist_aufsteigend_und_enthaelt_die_enden():
    ts = M.zeitraster(500, 40)
    assert ts[0] == 0 and ts[-1] == 500 and (np.diff(ts) > 0).all() and 1 in ts and 12 in ts
    assert list(M.zeitraster(1, 10)) == [0, 1]


def test_halbierungssuche_findet_das_erste_t():
    f = lambda t: 1.0 / (t + 1)
    assert M._erstes_t(f, 0.25) == 3 and M._erstes_t(f, 1.0) == 0 and M._erstes_t(f, 1e-9, grenze=1000) is None


def test_reversibilitaet_wird_geprueft():
    P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])        # gerichteter Kreis, π gleichverteilt, nicht reversibel
    with pytest.raises(ValueError):
        M.spectrum_of(P, np.full(3, 1 / 3))
