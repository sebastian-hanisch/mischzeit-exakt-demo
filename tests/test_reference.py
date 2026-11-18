"""Vorgerechnete Tabellen (msz_reference.json): die eingefrorenen Zahlen gleichen der Neurechnung mit dem Code der App.

Der Hauptlauf steht in `python tools/msz_make_tables.py --check` (schneller Teil) und `--check --lang` (alles); hier die Teile, die in Sekunden laufen."""

import math

import numpy as np
import pytest

import msz_chain as K
import msz_constants as C
import msz_families as F
import msz_mixing as M
import msz_reference as REF
import msz_simulation as S

E = 1 / math.e


def test_datei_ist_vollstaendig():
    d = REF.daten()
    assert set(d) == {"hinweis", "ring_tabelle", "start_tabelle", "dbar_tabelle", "sim_tabelle", "sim_tabelle_zweiter_seed", "ring_fenster", "mischer", "karten"}
    assert len(d["ring_tabelle"]) == 14 * 8 * 2 and len(d["dbar_tabelle"]) == 14
    assert set(d["sim_tabelle"]) == {str(r) for r in C.SIM_R_TABELLE} and set(d["ring_fenster"]) == {str(n) for n in C.RING_FENSTER_N}
    assert set(d["mischer"]["Transpositionen"]) == {str(n) for n in C.MISCHER_N}


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_ring_tabelle_n_bis_8_gleich_neurechnung(n):
    for lam in C.LAM_TABELLE:
        for kette in ("Heat-bath", "Metropolis"):
            z = REF.ring_zelle(n, lam, kette)
            sp = M.ring_spectrum(n, lam, kette)
            assert z["N"] == sp.N and z["t_mix"] == M.t_mix(sp)
            assert z["t_rel"] == pytest.approx(M.t_rel(sp.mu), rel=1e-9)
            unten, oben = M.schranken(z["t_rel"], float(sp.pi.min()))
            assert z["unten"] == pytest.approx(unten, rel=1e-9) and z["oben"] == pytest.approx(oben, rel=1e-9)


def test_ring_tabelle_zwei_zellen_n12_gleich_neurechnung():
    for lam in (2.0, 20.0):
        z = REF.ring_zelle(12, lam, "Heat-bath")
        sp = M.ring_spectrum(12, lam, "Heat-bath")
        assert z["t_mix"] == M.t_mix(sp) and z["t_rel"] == pytest.approx(M.t_rel(sp.mu), rel=1e-9)


def test_start_tabelle_gleich_neurechnung():
    for lam in C.E2_LAMS:
        z = REF.start_zeile(lam)
        sp = M.ring_spectrum(10, lam, "Heat-bath")
        ch = K.ring_chain(10, lam, "Heat-bath")
        tx = M.t_alle_starts(sp)
        assert z["t_mix"] == tx.max() and z["t_leer"] == tx[ch.index(0)] and z["t_min"] == tx.min()
        assert z["t_mittel"] == pytest.approx(tx.mean()) and z["t_mittel_pi"] == pytest.approx((tx * sp.pi).sum())
        assert z["schlechteste_sind_maximalbelegungen"] and z["alle_maximalbelegungen_gleich"] and z["zustaende"] == 123


def test_dbar_tabelle_gleich_neurechnung_fuer_n_bis_10():
    for z in REF.dbar_zellen():
        if z["n"] > 10 or (z["n"] == 10 and z["lam"] > 2.0):
            continue
        sp = M.ring_spectrum(z["n"], z["lam"], "Heat-bath")
        assert (z["t_d"], z["t_d_e"], z["t_dbar"], z["T_e"]) == (M.t_mix(sp), M.t_mix(sp, E), M.t_mix_dbar(sp), M.t_mix_dbar(sp, E))


def test_dbar_tabelle_ungleichung_und_verhaeltnisse():
    z = REF.dbar_zellen()
    assert all(c["ungleichung_gilt"] and 1.0 - 1e-12 <= c["verhaeltnis_min"] and c["verhaeltnis_max"] <= 2.0 + 1e-12 for c in z)
    assert all(c["t_d"] <= c["t_dbar"] for c in z) and all(c["t_d_e"] <= c["T_e"] for c in z)


def test_fenster_n8_und_n12_gleich_neurechnung():
    for n in (8, 12):
        f = REF.fenster(n)
        sp = M.ring_spectrum(n, 1.0, "Heat-bath")
        x0 = K.maximalstart(K.ring_chain(n, 1.0, "Heat-bath"))
        for e, t in f["t"].items():
            assert t == M.t_start(sp, x0, float(e))
        ts, ds = f["kurve"]["t"], f["kurve"]["d"]
        assert ds == pytest.approx([M.d_start_t(sp, x0, t) for t in ts], abs=1e-12)


def test_mischer_tabelle_n_bis_5_gleich_neurechnung():
    for art in ("Transpositionen", "Top-zu-zufällig"):
        for n in (3, 4, 5):
            assert {float(k): v for k, v in REF.mischer(art)[str(n)].items()} == F.mischer_ts(n, art)


def test_karten_tabelle_gleich_neurechnung():
    assert REF.karten_52() == pytest.approx([F.riffle_tv(52, k) for k in range(0, C.KARTEN_K_MAX + 1)], abs=1e-15)
    assert {int(n): t for n, t in REF.daten()["karten"]["t_mix"].items()} == {n: F.riffle_t(n) for n in C.KARTEN_N_TABELLE}


def test_simulationstabelle_kleine_r_gleich_neurechnung_im_zufalls_seed_5000():
    for R in (100, 300, 1000):
        z = REF.sim_zeile(R)
        w = S.wiederholungen(R, C.TABELLE_REPS[R], C.TABELLE_SEED)
        assert z["t"] == w["t"] and z["t_zahl"] == w["t_zahl"] and z["wiederholungen"] == len(w["t"])
        assert z["rauschgrenze"] == pytest.approx(S.rauschgrenzen(R)[0], rel=1e-9) and z["rauschgrenze_zahl"] == pytest.approx(S.rauschgrenzen(R)[1], rel=1e-9)


def test_simulationstabelle_bandeigenschaften():
    exakt, exakt_zahl = 73, 11
    for R in (3000, 10000, 100000):
        z = REF.sim_zeile(R)
        assert z["ganz"]["nie"] == 0 and abs(z["ganz"]["mittel"] - exakt) <= 3
        assert abs(z["zahl"]["mittel"] - exakt_zahl) <= 1
    assert REF.sim_zeile(100)["ganz"]["nie"] == REF.sim_zeile(100)["ganz"]["von"] == 20
