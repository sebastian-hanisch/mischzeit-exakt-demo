"""Orakel-Tests: Matrix, Abstände, Mischzeiten, Eigenwerte und Räume gegen einen anderen Rechenweg.

(a) die Übergangsmatrix der App gegen die aus der Definition aufgezählte Bruch-Matrix (tests/oracle_helfer.py), n ≤ 8, beide Ketten;
(b) d(t) und d̄(t) der App (Eigenzerlegung, float) gegen die Potenz der Bruch-Matrix (ganze Zahlen), n = 4 bis 6, bis t = 40;
(c) t_mix(1/4) für n = 6 bis 12 und acht λ gegen eine frühere unabhängige Rechnung (tests/frueher_t_mix.py), live (n ≤ 9) und in der vorgerechneten Tabelle (n ≤ 14);
(d) Eigenwerte gegen die Formel g(λ) für gerades n und gegen scipy.sparse.linalg.eigsh;
(e) Satz 12.4 als Einschluss in jeder Zelle der vorgerechneten Tabelle;
(f) Kreis: Fourier-Weg gegen die Matrixpotenz (n = 5 bis 25, 1e-12) und gegen die Eigenwerte (12.17);
(g) Hyperwürfel: Zählkette gegen die volle 2^n-Kette (n ≤ 8, 1e-14);
(h) Karten: Formel der Bemerkung 8.14 gegen die Aufzählung des inversen Riffle (n ≤ 4) und gegen die Formel über alle n! Anordnungen (n ≤ 5)."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import msz_chain as K
import msz_families as F
import msz_mixing as M
import msz_reference as REF
import oracle_helfer as O
from frueher_t_mix import HEAT_BATH, METROPOLIS

KETTEN = ("Heat-bath", "Metropolis")


# ---------------------------------------------------------------------------------------------------- (a) Matrix aus der Definition
@pytest.mark.parametrize("kette", KETTEN)
@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("lam", [Fr(1, 2), Fr(1), Fr(2), Fr(5)])
def test_a_matrix_gleich_der_aufgezaehlten_definition(n, lam, kette):
    S, P = O.uebergang(n, lam, kette)
    ch = K.build_ring_chain(n, float(lam), kette)
    assert list(ch.states) == S
    assert all(sum(z) == 1 for z in P)                                      # in Brüchen exakt
    pi = O.gleichgewicht(S, lam)
    assert all(sum(pi[i] * P[i][j] for i in range(len(S))) == pi[j] for j in range(len(S)))
    assert np.allclose(ch.dense(), np.array([[float(a) for a in z] for z in P]), atol=1e-15, rtol=0)
    assert np.allclose(ch.pi, [float(a) for a in pi], atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------------------------------- (b) Abstände gegen Bruch-Potenzen
@pytest.mark.parametrize("kette,n,lam,t_max", [("Heat-bath", 4, 1, 40), ("Heat-bath", 4, 2, 40), ("Metropolis", 4, 2, 40), ("Heat-bath", 5, 2, 30),
                                              ("Metropolis", 5, 2, 30), ("Heat-bath", 6, 1, 25)])
def test_b_d_und_dbar_gegen_bruchpotenz(n, lam, kette, t_max):
    S, P = O.uebergang(n, Fr(lam), kette)
    d_exakt, db_exakt = O.abstaende(S, P, Fr(lam), t_max)
    sp = M.ring_spectrum(n, float(lam), kette)
    for t in range(t_max + 1):
        assert M.d_t(sp, t) == pytest.approx(float(d_exakt[t]), abs=1e-12)
        assert M.dbar_t(sp, t) == pytest.approx(float(db_exakt[t]), abs=1e-12)
        assert d_exakt[t] <= db_exakt[t] <= 2 * d_exakt[t]                    # in Brüchen exakt


def test_b_mischzeit_n4_aus_bruechen():
    S, P = O.uebergang(4, Fr(1), "Heat-bath")
    d, db = O.abstaende(S, P, Fr(1), 60)
    assert d[0] == Fr(6, 7) and d[1] == Fr(17, 28) and d[2] == Fr(121, 224)
    assert next(t for t, x in enumerate(d) if x <= Fr(1, 4)) == 12
    assert next(t for t, x in enumerate(db) if x <= Fr(1, 4)) == 21
    assert next(t for t, x in enumerate(db) if float(x) <= 1 / math.e) == 16


# ---------------------------------------------------------------------------------------------------- (c) t_mix gegen frühere Rechnung
@pytest.mark.parametrize("n", [6, 7, 8, 9])
def test_c_t_mix_live_gegen_fruehere_rechnung(n):
    for kette, tabelle in (("Heat-bath", HEAT_BATH), ("Metropolis", METROPOLIS)):
        for lam, soll in tabelle[n].items():
            if lam in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0) or n <= 7:
                assert M.t_mix(M.ring_spectrum(n, lam, kette), 0.25) == soll, (n, lam, kette)


def test_c_t_mix_n12_zwei_zellen_live():
    assert M.t_mix(M.ring_spectrum(12, 2.0, "Heat-bath")) == HEAT_BATH[12][2.0] == 93
    assert M.t_mix(M.ring_spectrum(12, 2.0, "Metropolis")) == METROPOLIS[12][2.0] == 62


def test_c_vorgerechnete_tabelle_gegen_fruehere_rechnung_n6_bis_16():
    zellen = 0
    for kette, tabelle in (("Heat-bath", HEAT_BATH), ("Metropolis", METROPOLIS)):
        for n, zeile in tabelle.items():
            for lam, soll in zeile.items():
                assert REF.ring_zelle(n, lam, kette)["t_mix"] == soll, (n, lam, kette)
                zellen += 1
    assert zellen == 11 * 8 * 2


def test_c_schlechtester_start_ist_die_maximalbelegung_in_der_tabelle():
    """t_mix über alle Starts gleich t_mix aus dem Maximalstart: in allen Zellen n = 3 bis 14 (alle Starts) der vorgerechneten Tabelle."""
    z = [v for k, v in REF.ring_zellen().items() if int(k.split("|")[0]) <= 14]
    assert len(z) == 12 * 8 * 2 and all(c["t_mix"] == c["t_mix_maximalstart"] for c in z)


# ---------------------------------------------------------------------------------------------------- (d) Eigenwerte
@pytest.mark.parametrize("n", [4, 6, 8, 10, 12])
@pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_d_zweitgroesster_eigenwert_gerades_n_gegen_formel(n, lam):
    sp = M.ring_spectrum(n, lam, "Heat-bath")
    assert sp.mu[0] == pytest.approx(1.0, abs=1e-12)
    assert n * (1 - sp.mu[1]) == pytest.approx(M.g_gerades_n(lam), rel=1e-9)
    assert sp.mu[-1] > -1e-12                                              # Heat-bath: kein negativer Eigenwert
    assert M.t_rel(sp.mu) == pytest.approx(n / M.g_gerades_n(lam), rel=1e-9)


def test_d_eigenwerte_gegen_eigsh_n12():
    from scipy.sparse.linalg import eigsh
    ch = K.ring_chain(12, 2.0, "Heat-bath")
    s = np.sqrt(ch.pi)
    A = (ch.dense() * s[:, None]) / s[None, :]
    A = (A + A.T) / 2
    gross = np.sort(eigsh(A, k=3, which="LA", return_eigenvectors=False))[::-1]
    klein = float(eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
    sp = M.ring_spectrum(12, 2.0, "Heat-bath")
    assert gross == pytest.approx(sp.mu[:3], abs=1e-10) and klein == pytest.approx(sp.mu[-1], abs=1e-10)


def test_d_invarianz_der_zwei_funktionen_in_bruechen():
    """P f_A = f_A + ((2p−1) f_A − p f_B)/n und P f_B = f_B + (2p f_A − 2 f_B)/n, exakt in Brüchen für n = 4, 6, 8 (gerade) auf der aufgezählten Matrix."""
    for n in (4, 6, 8):
        for lam in (Fr(1, 2), Fr(1), Fr(2), Fr(5)):
            S, P = O.uebergang(n, lam, "Heat-bath")
            p = lam / (1 + lam)
            fA = [sum((-1) ** v * ((x >> v) & 1) for v in range(n)) for x in S]
            fB = [sum((-1) ** v * ((x >> v) & 1) * ((x >> ((v + 2) % n)) & 1) for v in range(n)) for x in S]
            N = len(S)
            PfA = [sum(P[i][j] * fA[j] for j in range(N)) for i in range(N)]
            PfB = [sum(P[i][j] * fB[j] for j in range(N)) for i in range(N)]
            assert all(PfA[i] == fA[i] + ((2 * p - 1) * fA[i] - p * fB[i]) / n for i in range(N))
            assert all(PfB[i] == fB[i] + (2 * p * fA[i] - 2 * fB[i]) / n for i in range(N))


def test_d_ungerades_n_schliesst_die_invarianz_nicht():
    n = 5
    S, P = O.uebergang(n, Fr(2), "Heat-bath")
    p = Fr(2, 3)
    fA = [sum((-1) ** v * ((x >> v) & 1) for v in range(n)) for x in S]
    fB = [sum((-1) ** v * ((x >> v) & 1) * ((x >> ((v + 2) % n)) & 1) for v in range(n)) for x in S]
    N = len(S)
    PfA = [sum(P[i][j] * fA[j] for j in range(N)) for i in range(N)]
    assert not all(PfA[i] == fA[i] + ((2 * p - 1) * fA[i] - p * fB[i]) / n for i in range(N))


# ---------------------------------------------------------------------------------------------------- (e) Satz 12.4
def test_e_satz_12_4_in_jeder_zelle_der_tabelle():
    zellen = REF.ring_zellen()
    assert len(zellen) == 14 * 8 * 2
    for k, z in zellen.items():
        assert z["unten"] <= z["t_mix"] <= z["oben"], k
        assert z["t_rel"] == pytest.approx(1.0 / (1.0 - max(abs(z["lambda_2"]), abs(z["lambda_min"]))), rel=1e-12)


def test_e_satz_12_4_live_ring_kreis_wuerfel():
    for n, lam, kette in [(6, 0.5, "Heat-bath"), (8, 20.0, "Metropolis"), (9, 5.0, "Heat-bath"), (10, 2.0, "Heat-bath")]:
        sp = M.ring_spectrum(n, lam, kette)
        unten, oben = M.schranken(M.t_rel(sp.mu), float(sp.pi.min()))
        assert unten <= M.t_mix(sp) <= oben
    for n in (5, 25, 100):
        t, g = F.kreis_t(n), F.kreis_luecke(n)
        unten, oben = M.schranken(1 / g, 1 / n)
        assert unten <= t <= oben
    for n in (12, 50):
        t = F.wuerfel_ts(n, (0.25,))[0.25]
        unten, oben = M.schranken(float(n), 2.0 ** -n)
        assert unten <= t <= oben


# ---------------------------------------------------------------------------------------------------- (f) Kreis
@pytest.mark.parametrize("traege", [True, False])
@pytest.mark.parametrize("n", [5, 7, 9, 12, 25])
def test_f_kreis_fourier_gegen_matrixpotenz(n, traege):
    P = np.zeros((n, n))
    for i in range(n):
        if traege:
            P[i, i] += 0.5
            P[i, (i - 1) % n] += 0.25
            P[i, (i + 1) % n] += 0.25
        else:
            P[i, (i - 1) % n] += 0.5
            P[i, (i + 1) % n] += 0.5
    Pt = np.eye(n)
    for t in range(0, 41):
        assert F.kreis_d(n, t, traege) == pytest.approx(0.5 * np.abs(Pt[0] - 1.0 / n).sum(), abs=1e-12)
        Pt = Pt @ P


def test_f_kreis_eigenwerte_gegen_beispiel_12_17():
    n = 11
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i], P[i, (i - 1) % n], P[i, (i + 1) % n] = 0.5, 0.25, 0.25
    ev = np.sort(np.linalg.eigvalsh(P))[::-1]
    soll = np.sort((1 + np.cos(2 * np.pi * np.arange(n) / n)) / 2)[::-1]
    assert ev == pytest.approx(soll, abs=1e-13) and F.kreis_luecke(n) == pytest.approx(1 - ev[1], abs=1e-13)


# ---------------------------------------------------------------------------------------------------- (g) Hyperwürfel
@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_g_zaehlkette_gegen_volle_kette(n):
    t_max = 30
    voll = O.wuerfel_voll_d(n, t_max)
    zaehl = F.wuerfel_kurve(n, t_max)
    assert zaehl == pytest.approx(voll, abs=1e-14)


# ---------------------------------------------------------------------------------------------------- (h) Karten
@pytest.mark.parametrize("n,k", [(3, 1), (3, 2), (3, 3), (4, 1), (4, 2)])
def test_h_riffle_formel_gegen_inverse_aufzaehlung(n, k):
    assert F.riffle_tv_bruch(n, k) == O.inverses_riffle_tv(n, k)


@pytest.mark.parametrize("n,k", [(3, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 4)])
def test_h_riffle_euler_zahlen_gegen_summe_ueber_alle_anordnungen(n, k):
    assert F.riffle_tv_bruch(n, k) == O.vorwaerts_riffle_tv(n, k)
