"""Literaturwerte (nur aus gelesenen Quellen; Stand der Einsicht: Quellenbericht 2026-11-17 und 2026-11-18, Lesestufe W = Volltext gelesen).

* Levin/Peres/Wilmer, Markov Chains and Mixing Times, 2. Aufl., AMS 2017 (Volltext gelesen): Abschnitt 8.3 mit Bemerkung 8.14, S. 113 und 114: nach 7 Riffle-Mischungen von
  52 Karten beträgt der Abstand „approximately 0.3341“; Abschnitt 5.3.2: n²/32 ≤ t_mix ≤ n² für die träge Irrfahrt auf dem n-Kreis; Beispiel 12.17: Eigenwerte
  (1 + cos(2πj/n))/2, Lücke (1 − cos(2π/n))/2; Beispiel 12.16: Eigenwerte (n − |J|)/n, Lücke 1/n des Hyperwürfels; (4.34): t_mix(ε) ≤ ⌈log₂(1/ε)⌉ t_mix(1/4).
* Bayer/Diaconis 1992, Trailing the Dovetail Shuffle to its Lair, Tabelle 1 (OCR, Zeichenverlust in Satz 2; Tabelle gelesen): 52 Karten, 5 bis 10 Mischungen:
  0,924 0,614 0,334 0,167 0,085 0,043.
Nicht benutzt, weil nicht gelesen: Norris, Handbook (Buch), Robert/Casella, Sinclair/Jerrum 1989, Besag 1974, Gelman/Rubin 1992 (nur zwei Seiten)."""

import math

import numpy as np
import pytest

import msz_families as F
import msz_mixing as M


def test_lpw_s_114_52_karten_nach_7_mischungen():
    assert F.riffle_tv(52, 7) == pytest.approx(0.3341, abs=5e-5)


def test_bayer_diaconis_tabelle_1_52_karten():
    tabelle = {5: 0.924, 6: 0.614, 7: 0.334, 8: 0.167, 9: 0.085, 10: 0.043}
    for k, wert in tabelle.items():
        assert round(F.riffle_tv(52, k), 3) == wert


def test_lpw_5_3_2_kreis_n_quadrat_durch_32_bis_n_quadrat():
    for n in (5, 8, 25, 50, 100, 400):
        assert n ** 2 / 32 <= F.kreis_t(n) <= n ** 2


def test_lpw_12_17_kreis_eigenwerte_und_luecke():
    for n in (5, 12, 40):
        ev = np.sort(F.kreis_eigenwerte(n, True))[::-1]
        assert ev == pytest.approx(np.sort((1 + np.cos(2 * np.pi * np.arange(n) / n)) / 2)[::-1], abs=1e-14)
        assert F.kreis_luecke(n, True) == pytest.approx((1 - math.cos(2 * math.pi / n)) / 2, abs=1e-14)


def test_lpw_12_16_hyperwuerfel_luecke_ein_durch_n():
    """Die Zählkette hat dieselben Eigenwerte (n − j)/n, j = 0 … n (Eigenfunktionen der vollen Kette mit |J| = j): zweitgrößter 1 − 1/n."""
    for n in (4, 9):
        w = np.arange(n + 1)
        P = np.zeros((n + 1, n + 1))
        for k in w:
            P[k, k] += 0.5
            if k < n:
                P[k, k + 1] += (n - k) / (2 * n)
            if k > 0:
                P[k, k - 1] += k / (2 * n)
        ev = np.sort(np.linalg.eigvals(P).real)[::-1]
        assert ev == pytest.approx(1 - np.arange(n + 1) / n, abs=1e-12)
    assert F.wuerfel_luecke(9) == pytest.approx(1 / 9)


def test_lpw_4_34_umrechnung_der_schwelle():
    sp = M.ring_spectrum(8, 2.0, "Heat-bath")
    t4 = M.t_mix(sp, 0.25)
    for eps in (0.5, 1 / math.e, 0.1, 0.05, 0.01, 0.001):
        assert M.t_mix(sp, eps) <= math.ceil(math.log2(1 / eps)) * t4
    assert F.kreis_t(50, 0.01) <= math.ceil(math.log2(100)) * F.kreis_t(50)
