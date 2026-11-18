"""Simulation: der vektorisierte SplitMix64 gleicht dem Skalar-Generator, die Rauschgrenze ist exakt und stimmt mit der Simulation überein, Randfälle."""

import math
from fractions import Fraction as Fr
from math import factorial

import numpy as np
import pytest

import msz_chain as K
import msz_simulation as S
from msz_rng import SplitMix64


def test_vektorstrom_gleich_skalarstrom():
    for seed in (0, 1, 123456789, 2 ** 63 + 5, 2 ** 64 - 1):
        skalar = SplitMix64(seed)
        erwartet = [skalar.next_u64() for _ in range(300)]
        vek = S.VektorSplitMix64(seed)
        gezogen = vek.block(100).tolist() + vek.block(1).tolist() + vek.block(199).tolist()
        assert gezogen == erwartet
        assert vek.state == skalar.state


def test_zufall_und_unterhalb_gleich_skalar():
    a, b = SplitMix64(77), S.VektorSplitMix64(77)
    assert [a.random() for _ in range(50)] == b.zufall(50).tolist()
    assert [a.below(10) for _ in range(50)] == b.unterhalb(50, 10).tolist()


def test_handwerte_rauschgrenze_in_brueche_aufgezaehlt():
    """Erwarteter Abstand von R perfekten Ziehungen, über alle Zählvektoren in Brüchen aufgezählt (anderer Weg als die Binomialsumme)."""
    def kompositionen(R, Nz):
        if Nz == 1:
            yield (R,)
            return
        for a in range(R + 1):
            for rest in kompositionen(R - a, Nz - 1):
                yield (a,) + rest

    def exakt(R, Nz):
        gesamt = Fr(0)
        for c in kompositionen(R, Nz):
            mult = factorial(R)
            for a in c:
                mult //= factorial(a)
            gesamt += Fr(mult, Nz ** R) * (sum(abs(Fr(a, R) - Fr(1, Nz)) for a in c) / 2)
        return gesamt

    assert exakt(2, 2) == Fr(1, 4) and exakt(4, 2) == Fr(3, 16)
    assert S.rauschgrenze_exakt(np.full(2, 0.5), 2) == pytest.approx(0.25, abs=1e-14) and S.rauschgrenze_exakt(np.full(2, 0.5), 4) == pytest.approx(3 / 16, abs=1e-14)
    e7 = Fr(279936, 823543)
    assert exakt(7, 7) == e7 and S.rauschgrenze_exakt(np.full(7, 1 / 7), 7) == pytest.approx(float(e7), abs=1e-13) and float(e7) == pytest.approx(0.3399, abs=5e-5)
    e14 = Fr(169789022208, 678223072849)
    assert S.rauschgrenze_exakt(np.full(7, 1 / 7), 14) == pytest.approx(float(e14), abs=1e-13) and float(e14) > 0.25 and float(e14) == pytest.approx(0.2503, abs=5e-5)


def test_rauschgrenze_fuer_ring_gegen_simulierte_ziehungen():
    """Exakte Rauschgrenze gegen das Mittel simulierter Ziehungen (SplitMix64, 60 Wiederholungen): innerhalb von 4 Standardfehlern."""
    ch = K.ring_chain(10, 2.0, "Heat-bath")
    cdf = np.cumsum(ch.pi)
    cdf[-1] = 1.0
    rng = S.VektorSplitMix64(31)
    for R in (100, 300, 3000):
        werte = []
        for _ in range(60):
            idx = np.searchsorted(cdf, rng.zufall(R), side="right")
            werte.append(0.5 * np.abs(np.bincount(idx, minlength=ch.N) / R - ch.pi).sum())
        se = np.std(werte, ddof=1) / math.sqrt(len(werte))
        assert abs(np.mean(werte) - S.rauschgrenzen(R)[0]) < 4 * se + 1e-9


def test_rauschgrenze_faellt_mit_r_und_naeherung():
    gz = [S.rauschgrenzen(R)[0] for R in (100, 300, 1000, 3000, 10000, 100000)]
    assert all(a > b for a, b in zip(gz, gz[1:]))
    assert gz[0] > 0.25 > gz[2] and 0.20 < gz[1] < 0.27               # bei R = 100 über der Schwelle 1/4
    for R in (100, 300, 1000, 3000):
        assert abs(S.rauschgrenze_naeherung(R) / S.rauschgrenzen(R)[0] - 1) < 0.03      # Näherung bis 3 % daneben
    assert S.rauschgrenzen(100)[1] < S.rauschgrenzen(100)[0]            # die Teilchenzahl hat weniger Werte, also weniger Rauschen


def test_referenz_exakt_73_und_11():
    ref = S.referenz()
    assert (ref.t_exakt, ref.t_exakt_zahl, ref.N) == (73, 11, 123)
    ch = K.ring_chain(10, 2.0, "Heat-bath")
    assert ref.kurve[0] == pytest.approx(1.0 - ch.pi[K.maximalstart(ch)], abs=1e-13)      # t = 0: Punktmasse im Start
    assert (np.diff(ref.kurve) <= 1e-13).all() and ref.kurve[73] <= 0.25 < ref.kurve[72]


def test_ein_lauf_hat_richtige_laenge_und_startabstand():
    r = S.lauf(50, 3)
    assert len(r.tv) == len(r.tv_zahl) == C_T_MAX + 1
    ch = K.ring_chain(10, 2.0, "Heat-bath")
    # Zeitpunkt 0: alle Ketten im Maximalstart, empirische Verteilung = Punktmasse: Abstand 1 − π(Start)
    assert r.tv[0] == pytest.approx(1.0 - ch.pi[K.maximalstart(ch)], abs=1e-13)


def test_lauf_ist_deterministisch_im_zufalls_seed():
    a, b, c = S.lauf(80, 5), S.lauf(80, 5), S.lauf(80, 6)
    assert np.array_equal(a.tv, b.tv) and not np.array_equal(a.tv, c.tv)


def test_viele_ketten_treffen_die_exakte_mischzeit_als_band():
    """R = 20 000 Ketten, zwei Zufalls-Seeds: geschätzte Mischzeit innerhalb von ±3 Schritten um 73, die der Teilchenzahl um ±2 Schritte um 11."""
    for seed in (101, 202):
        r = S.lauf(20000, seed)
        assert r.t_schaetz is not None and abs(r.t_schaetz - 73) <= 3
        assert r.t_schaetz_zahl is not None and abs(r.t_schaetz_zahl - 11) <= 2


def test_empirischer_abstand_ueber_der_exakten_kurve_minus_rauschen():
    """Der Mittelwert des empirischen Abstands über 10 Läufe liegt nie unter dem exakten Abstand minus 3 Standardfehler (Rauschen macht den Abstand nur größer)."""
    ref = S.referenz()
    laeufe = [S.lauf(1000, 40 + i) for i in range(10)]
    tv = np.array([l.tv for l in laeufe])
    mittel, se = tv.mean(axis=0), tv.std(axis=0, ddof=1) / math.sqrt(10)
    assert (mittel >= ref.kurve - 3 * se - 1e-9).all()


def test_randfaelle():
    with pytest.raises(ValueError):
        S.lauf(0, 1)
    assert S.zusammenfassung([None, None]) == {"mittel": None, "min": None, "max": None, "nie": 2, "von": 2}
    z = S.zusammenfassung([10, None, 14])
    assert z == {"mittel": 12.0, "min": 10, "max": 14, "nie": 1, "von": 3}
    ch = K.ring_chain(10, 2.0, "Heat-bath")
    r1 = S.lauf(1, 9)                                    # eine einzige Kette: der empirische Abstand ist nie kleiner als 1 − (größter Eintrag von π)
    assert len(r1.tv) == C_T_MAX + 1 and (r1.tv >= 1.0 - ch.pi.max() - 1e-12).all()


C_T_MAX = 160
