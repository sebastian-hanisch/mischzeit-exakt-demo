"""Hard-Core-Ring: Zustände und Übergangsmatrix an der Handrechnung (n = 4, λ = 1: 7 Zustände) und Struktur für viele (n, λ, Kette)."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import msz_chain as K

KETTEN = ("Heat-bath", "Metropolis")


def test_zustaende_sind_die_lucas_zahlen():
    assert [K.lucas(n) for n in (3, 4, 5, 6, 10, 12, 16, 24)] == [4, 7, 11, 18, 123, 322, 2207, 103682]
    assert [len(K.ring_states(n)) for n in range(3, 15)] == [K.lucas(n) for n in range(3, 15)]


def test_zustaende_n4_von_hand(ring4):
    # leer, vier Einzelplätze, die zwei Paare {0, 2} und {1, 3}
    assert list(ring4.states) == [0, 1, 2, 4, 5, 8, 10]
    assert ring4.N == 7 and np.allclose(ring4.pi, 1 / 7)


def test_zeilen_von_hand_n4_lambda1(ring4):
    P = ring4.dense()
    i = ring4.index
    leer, a0, p02 = i(0), i(1), i(0b0101)
    assert P[leer][leer] == pytest.approx(1 / 2) and all(P[leer][i(1 << v)] == pytest.approx(1 / 8) for v in range(4))
    assert P[a0][a0] == pytest.approx(3 / 4) and P[a0][leer] == pytest.approx(1 / 8) and P[a0][p02] == pytest.approx(1 / 8)
    assert P[p02][p02] == pytest.approx(3 / 4) and P[p02][a0] == pytest.approx(1 / 8) and P[p02][i(4)] == pytest.approx(1 / 8)
    assert int((P > 0).sum()) == 7 + 4 + 4 * 2 + 2 * 2   # 7 Diagonale, 4 (leer → Einzel), 4·2 (Einzel → leer, Paar), 2·2 (Paar → Einzel)


@pytest.mark.parametrize("kette", KETTEN)
@pytest.mark.parametrize("n", [3, 4, 5, 6, 9])
@pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 5.0, 20.0])
def test_struktur_zeilensumme_gleichgewicht_detailed_balance(n, lam, kette):
    ch = K.build_ring_chain(n, lam, kette)
    P = ch.dense()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-13) and (P >= 0).all()
    assert np.allclose(ch.pi @ P, ch.pi, atol=1e-14)
    assert np.allclose(ch.pi[:, None] * P, (ch.pi[:, None] * P).T, atol=1e-15)       # Detailed Balance
    assert ch.pi.sum() == pytest.approx(1.0)


def test_gleichgewicht_ist_lambda_hoch_teilchenzahl():
    ch = K.ring_chain(6, 3.0, "Heat-bath")
    w = 3.0 ** ch.teilchen
    assert np.allclose(ch.pi, w / w.sum())


def test_metropolis_zeile_von_hand_n4_lambda2():
    """Metropolis, λ = 2: aus ∅ wird jeder gezogene Platz mit min(1, 2) = 1 belegt (1/4 je Platz), aus {0} wird Platz 0 mit min(1, 1/2) = 1/2 geleert (1/8),
    Platz 2 mit 1 belegt (1/4), die Plätze 1 und 3 haben einen belegten Nachbarn und werden abgelehnt (bleiben): 1/4 + 1/4 + 1/8 (Platz 0 nicht geleert) = 5/8."""
    ch = K.ring_chain(4, 2.0, "Metropolis")
    P = ch.dense()
    i = ch.index
    assert all(P[i(0)][i(1 << v)] == pytest.approx(1 / 4) for v in range(4)) and P[i(0)][i(0)] == pytest.approx(0.0)
    assert P[i(1)][i(0)] == pytest.approx(1 / 8) and P[i(1)][i(0b0101)] == pytest.approx(1 / 4) and P[i(1)][i(1)] == pytest.approx(1 / 4 + 1 / 4 + 1 / 8)


def test_maximalstart_hat_n_halbe_teilchen():
    for n in range(3, 13):
        ch = K.ring_chain(n, 1.0, "Heat-bath")
        assert ch.teilchen[K.maximalstart(ch)] == n // 2 == ch.teilchen.max()


def test_takt_und_einheiten():
    assert K.takt(4, 1.0, "Heat-bath") == 8.0 and K.takt(10, 2.0, "Heat-bath") == 30.0
    assert K.takt(10, 2.0, "Metropolis") == 20.0 and K.takt(10, 0.5, "Metropolis") == 10.0
    assert Fr(12, int(K.takt(4, 1.0, "Heat-bath"))) == Fr(3, 2)       # 12 Schritte = 3/2 Abfertigungsdauern


def test_label_und_index():
    ch = K.ring_chain(4, 1.0, "Heat-bath")
    assert K.label(0, 4) == "∅ (leer)" and K.label(0b0101, 4) == "{0, 2}"
    with pytest.raises(ValueError):
        ch.index(0b0011)          # zwei Nachbarn belegt


def test_ring_braucht_drei_plaetze_und_positives_lambda():
    with pytest.raises(ValueError):
        K.ring_states(2)
    with pytest.raises(ValueError):
        K.build_ring_chain(5, 0.0)
    with pytest.raises(ValueError):
        K.build_ring_chain(5, 1.0, "Gibbs")


def test_torus_zustaende_klein_gegen_aufzaehlung():
    from itertools import product
    for l1, l2 in [(3, 3), (3, 4)]:
        zaehlung = 0
        for bits in product((0, 1), repeat=l1 * l2):
            g = [[bits[a * l2 + b] for b in range(l2)] for a in range(l1)]
            if all(not (g[a][b] and (g[(a + 1) % l1][b] or g[a][(b + 1) % l2])) for a in range(l1) for b in range(l2)):
                zaehlung += 1
        assert K.torus_zustaende(l1, l2) == zaehlung
    assert K.torus_zustaende(8, 8) == 213256442503


@pytest.mark.parametrize("kette", KETTEN)
@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_irreduzibel_und_aperiodisch_eine_potenz_hat_nur_positive_eintraege(n, kette):
    """Eine Potenz P^k mit lauter positiven Einträgen heißt: irreduzibel und aperiodisch (primitiv); k = Zahl der Zustände reicht hier."""
    for lam in (0.5, 1.0, 5.0, 20.0):
        P = K.build_ring_chain(n, lam, kette).dense()
        assert (np.linalg.matrix_power(P, len(P)) > 0).all(), (n, lam, kette)
