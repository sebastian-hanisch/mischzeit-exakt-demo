"""Eigenschaftstests (metamorph): Aussagen, die für jede Kette gelten müssen, an je 20 bis 50 festen Zufallsinstanzen (random.Random mit festem Seed).

Jede Eigenschaft ist eine belegte Aussage über Markov-Ketten oder über diese Ketten (Heat-bath, Metropolis, Hard-Core-Ring), nicht nur das heutige Verhalten des Codes."""

import math
import random

import numpy as np
import pytest

import msz_chain as K
import msz_families as F
import msz_mixing as M
import msz_simulation as S


def _instanzen(anzahl, seed, n_max=8):
    rng = random.Random(seed)
    return [(rng.randint(3, n_max), rng.choice([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]), rng.choice(("Heat-bath", "Metropolis"))) for _ in range(anzahl)]


INST = _instanzen(30, 2026)


@pytest.mark.parametrize("n,lam,kette", INST)
def test_d_faellt_monoton_und_dbar_liegt_zwischen_d_und_2d(n, lam, kette):
    sp = M.ring_spectrum(n, lam, kette)
    ts = list(range(0, 61, 3))
    d = [M.d_t(sp, t) for t in ts]
    db = [M.dbar_t(sp, t) for t in ts]
    assert all(a >= b - 1e-12 for a, b in zip(d, d[1:])) and all(a >= b - 1e-12 for a, b in zip(db, db[1:]))
    assert all(x - 1e-12 <= y <= 2 * x + 1e-12 for x, y in zip(d, db))


@pytest.mark.parametrize("n,lam,kette", INST[:20])
def test_mischzeit_faellt_in_epsilon_und_erfuellt_die_umrechnung(n, lam, kette):
    sp = M.ring_spectrum(n, lam, kette)
    epsilons = [0.5, 0.25, 1 / math.e, 0.1, 0.05, 0.01]
    t = [M.t_mix(sp, e) for e in epsilons]
    reihenfolge = sorted(zip(epsilons, t), reverse=True)                    # größeres ε zuerst
    assert all(a[1] <= b[1] for a, b in zip(reihenfolge, reihenfolge[1:]))  # kleineres ε braucht nicht weniger Schritte
    t4 = M.t_mix(sp, 0.25)
    assert all(M.t_mix(sp, e) <= math.ceil(math.log2(1 / e)) * t4 for e in epsilons if e < 0.25)
    assert M.t_mix(sp, 0.25) <= M.t_mix_dbar(sp, 0.25) <= M.t_mix(sp, 0.125)       # aus d ≤ d̄ ≤ 2d


@pytest.mark.parametrize("n,lam,kette", [(n, lam, kette) for n, lam, kette in INST if n >= 4][:20])
def test_drehung_und_spiegelung_des_starts_aendern_t_x_nicht(n, lam, kette):
    ch = K.ring_chain(n, lam, kette)
    sp = M.ring_spectrum(n, lam, kette)
    tx = M.t_alle_starts(sp)
    rng = random.Random(n * 7 + int(lam * 10))
    full = (1 << n) - 1
    for _ in range(6):
        x = int(ch.states[rng.randrange(ch.N)])
        r = rng.randrange(n)
        gedreht = ((x << r) | (x >> (n - r))) & full
        gespiegelt = sum(((x >> v) & 1) << ((n - v) % n) for v in range(n))
        assert tx[ch.index(gedreht)] == tx[ch.index(x)] == tx[ch.index(gespiegelt)]


@pytest.mark.parametrize("n,lam", [(n, lam) for n, lam, _ in INST][:12])
def test_alle_maximalbelegungen_sind_gleich_schlecht_und_der_start_aus_pi_hat_d_null(n, lam):
    ch = K.ring_chain(n, lam, "Heat-bath")
    sp = M.ring_spectrum(n, lam, "Heat-bath")
    tx = M.t_alle_starts(sp)
    voll = ch.teilchen == n // 2
    assert len(set(tx[voll])) == 1
    # ein Start, der selbst aus π gezogen wurde: π P^t = π, also Abstand 0 in jedem Schritt
    for t in (1, 5, 40):
        assert np.abs(ch.pi @ M.rows_at(sp, t) - ch.pi).max() < 1e-12


@pytest.mark.parametrize("n,lam,kette", INST)
def test_zeilensummen_gleichgewicht_und_detailed_balance(n, lam, kette):
    ch = K.ring_chain(n, lam, kette)
    P = ch.dense()
    assert np.allclose(P.sum(axis=1), 1.0) and np.allclose(ch.pi @ P, ch.pi, atol=1e-14)
    assert np.allclose(ch.pi[:, None] * P, (ch.pi[:, None] * P).T, atol=1e-15)


@pytest.mark.parametrize("n", range(3, 11))
def test_heat_bath_hat_keinen_negativen_eigenwert_metropolis_bei_lambda_1_schon(n):
    for lam in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0):
        assert M.ring_spectrum(n, lam, "Heat-bath").mu[-1] >= -1e-12
    mp = M.ring_spectrum(n, 1.0, "Metropolis")
    assert mp.mu[-1] < -0.1


def test_kreis_t_mix_waechst_streng_in_n_ab_vier_punkten():
    """Streng wachsend ab n = 4 (n = 4 bis 60); n = 3 und n = 4 haben beide t_mix = 1 (Abweichung von der Erwartung „streng für alle n“)."""
    ts = [F.kreis_t(n) for n in range(4, 61)]
    assert all(a < b for a, b in zip(ts, ts[1:]))
    assert F.kreis_t(3) == F.kreis_t(4) == 1
    # n = 3, träge: d(1) = ½(|1/2 − 1/3| + 2·|1/4 − 1/3|) = 1/6 ≤ 1/4
    assert F.kreis_d(3, 1) == pytest.approx(1 / 6, abs=1e-14)


def test_kreis_nicht_traege_gerade_periodisch_ungerade_nicht():
    for n in range(3, 31):
        assert (F.kreis_t(n, 0.25, False) is None) == (n % 2 == 0)


def test_karten_52_abstand_faellt_streng_ab_vier_mischungen():
    d = [F.riffle_tv(52, k) for k in range(4, 20)]
    assert all(a > b for a, b in zip(d, d[1:]))


def test_riffle_abstand_faellt_in_k_fuer_viele_kartenzahlen():
    for n in (2, 3, 7, 16, 40, 100):
        d = [F.riffle_tv(n, k) for k in range(1, 12)]
        assert all(a >= b for a, b in zip(d, d[1:]))


def test_rauschgrenze_faellt_in_r():
    for R in (30, 100, 300, 1000, 3000):
        assert S.rauschgrenzen(R)[0] > S.rauschgrenzen(2 * R)[0]


def test_gleiches_seed_gleiche_ausgabe_anderes_seed_andere():
    a, b, c = S.lauf(200, 11), S.lauf(200, 11), S.lauf(200, 12)
    assert np.array_equal(a.tv, b.tv) and not np.array_equal(a.tv, c.tv)


def test_wuerfel_und_kreis_d_faellt_monoton():
    kurve = F.wuerfel_kurve(30, 300)
    assert (np.diff(kurve) <= 1e-13).all()
    ts = range(0, 200, 5)
    d = [F.kreis_d(40, t) for t in ts]
    assert all(a >= b - 1e-13 for a, b in zip(d, d[1:]))
