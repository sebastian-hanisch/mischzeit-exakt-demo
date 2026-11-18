"""Kreis, Hyperwürfel, Riffle-Mischen und Mischer auf Karten: Handrechnung (Kreis n = 5, 3 Karten) und bekannte Werte."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import msz_families as F


def test_kreis_n5_traege_von_hand():
    # (1/2 bleiben, je 1/4 links und rechts) gegen 1/5: d(0) = 4/5, d(1) = ½(|1/2 − 1/5| + 2·|1/4 − 1/5| + 2·(1/5)) = 2/5
    assert F.kreis_d(5, 0, True) == pytest.approx(4 / 5, abs=1e-14)
    assert F.kreis_d(5, 1, True) == pytest.approx(2 / 5, abs=1e-14)
    assert F.kreis_d(5, 2, True) == pytest.approx(11 / 40, abs=1e-14) and F.kreis_d(5, 3, True) == pytest.approx(29 / 160, abs=1e-14)
    assert F.kreis_t(5) == 3
    assert F.kreis_luecke(5, True) == pytest.approx((1 - math.cos(2 * math.pi / 5)) / 2) and F.kreis_luecke(5, True) == pytest.approx(0.3455, abs=5e-5)
    assert 5 ** 2 / 32 <= 3 <= 5 ** 2


def test_kreis_nicht_traege_gerades_n_ist_periodisch():
    assert [F.kreis_d(4, t, False) for t in range(0, 8)] == pytest.approx([3 / 4] + [1 / 2] * 7, abs=1e-13)
    assert F.kreis_t(4, 0.25, False) is None and F.kreis_ist_periodisch(4, False) and not F.kreis_ist_periodisch(5, False)
    assert F.kreis_luecke(4, False) == pytest.approx(0.0, abs=1e-15)
    assert F.kreis_t(5, 0.25, False) == 5


def test_kreis_bekannte_werte():
    assert [F.kreis_t(n) for n in (5, 25, 100, 400, 800)] == [3, 59, 949, 15180, 60718]
    assert all(n ** 2 / 32 <= F.kreis_t(n) <= n ** 2 for n in (5, 25, 100, 400, 800))


def test_wuerfel_bekannte_werte():
    assert [F.wuerfel_ts(n, (0.25,))[0.25] for n in (12, 50, 100, 400)] == [20, 120, 275, 1378]
    assert F.wuerfel_luecke(50) == 1 / 50


def test_wuerfel_pi_ist_binomial():
    pi = F.wuerfel_pi(6)
    assert pi == pytest.approx([math.comb(6, k) / 64 for k in range(7)], abs=1e-14)


def test_riffle_drei_karten_von_hand():
    assert [F.riffle_tv_bruch(3, k) for k in (1, 2, 3, 4)] == [Fr(1, 3), Fr(7, 48), Fr(13, 192), Fr(25, 768)]
    assert F.euler_zeile(3) == (1, 4, 1) and F.euler_zeile(4) == (1, 11, 11, 1) and sum(F.euler_zeile(5)) == 120
    assert F.riffle_t(3) == 2 and F.riffle_tv_bruch(3, 2) <= Fr(1, 4) < F.riffle_tv_bruch(3, 1)
    assert F.riffle_tv_bruch(2, 1) == Fr(1, 4) and F.riffle_t(2) == 1                    # zwei Karten: nach einer Mischung genau 1/4
    assert F.riffle_t(1) == 0 and F.riffle_tv_bruch(1, 3) == 0


def test_riffle_52_karten_neun_werte():
    soll = {5: 0.9237, 6: 0.6135, 7: 0.3341, 8: 0.1672, 9: 0.0854, 10: 0.0429, 11: 0.0215, 12: 0.0108, 13: 0.0054}
    for k, v in soll.items():
        assert round(F.riffle_tv(52, k), 4) == v
    assert all(F.riffle_tv(52, k) > 0.99 for k in (1, 2, 3, 4))
    assert F.riffle_t(52) == 8 and F.riffle_t(52, 0.01) == 13 and F.riffle_tv(52, 12) > 0.01 >= F.riffle_tv(52, 13)
    assert all(0.49 < F.riffle_tv(52, k + 1) / F.riffle_tv(52, k) < 0.51 for k in range(9, 13))


def test_riffle_t_mix_fuer_viele_kartenzahlen():
    assert [F.riffle_t(n) for n in (4, 8, 16, 32, 52, 64, 128)] == [3, 4, 5, 7, 8, 8, 10]


def test_riffle_kurve_laenge_und_start():
    kurve = F.riffle_kurve(52, 14)
    assert len(kurve) == 15 and kurve[0] == pytest.approx(1.0) or kurve[0] > 0.99


def test_mischer_kleine_kartenzahlen():
    """Drei Karten, zufällige Transpositionen (gleiche Position: nichts, 1/3; je Paar 2/9): d(1) = 1/3, d(2) = 5/54 = 0,093 (Eigenwerte 1, −1/3, 1/3 der drei Darstellungen von S_3),
    also t(0,5) = 1, t(1/4) = 2 und t(0,1) = 2."""
    assert F.mischer_ts(3, "Transpositionen") == {0.9: 0, 0.5: 1, 0.25: 2, 0.1: 2}
    assert F.mischer_ts(3, "Top-zu-zufällig")[0.25] == 2
    with pytest.raises(ValueError):
        F.mischer_ts(3, "Riffle")


def test_mischer_acht_karten_gegen_frueher_gemessene_einheiten():
    # Transpositionen 9 Schritte, Top-zu-zufällig 13 Schritte, Riffle 4 Mischungen (n = 8; vorgerechnet und hier neu)
    assert F.mischer_ts(8, "Transpositionen")[0.25] == 9
    assert F.mischer_ts(8, "Top-zu-zufällig")[0.25] == 13
    assert F.riffle_t(8) == 4
