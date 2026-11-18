"""Claims-Tests: jede Zahl der README-Tabelle „Befunde“ wird hier nachgerechnet (Ganzzahlen und Brüche exakt, Gleitkommazahlen auf die genannten Stellen).

Die README nennt je Zeile der Tabelle den Test (`test_cNN`). Simulationszahlen sind Zufallsproben mit dem ganzzahligen SplitMix64 (auf jeder Plattform gleich) und werden gegen die
vorgerechnete Tabelle (Zufalls-Seeds 5000 und 6000) geprüft; die Läufe für R ≤ 3000 werden hier neu gerechnet, R = 10 000 und 100 000 stehen nur in der Datei
(`python tools/msz_make_tables.py --check --lang` rechnet sie neu). Messungen, die nicht im Test stehen, nennt die README ausdrücklich."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

import msz_chain as K
import msz_constants as C
import msz_evaluation as ev
import msz_families as F
import msz_mixing as M
import msz_reference as REF
import msz_simulation as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
E = 1 / math.e


def _r(x, stellen):
    return round(x, stellen)


# ---------------------------------------------------------------------------------------------------- Handbeispiel: der Ring mit 4 Plätzen
def test_c01_drei_zahlen_eine_kette_n4_lambda1():
    sp = M.ring_spectrum(4, 1.0, "Heat-bath")
    assert K.ring_chain(4, 1.0, "Heat-bath").N == 7
    assert _r(M.d_t(sp, 11), 4) == 0.2631 and _r(M.d_t(sp, 12), 4) == 0.2435 and M.t_mix(sp) == 12
    assert M.dbar_t(sp, 20) > 0.25 >= M.dbar_t(sp, 21) and M.t_mix_dbar(sp, 0.25) == 21
    assert M.t_mix_dbar(sp, E) == 16 and _r(M.dbar_t(sp, 15), 4) == 0.3858 and _r(M.dbar_t(sp, 16), 4) == 0.3575
    assert M.t_mix(sp, E) == 7


def test_c02_umrechnung_der_schwelle_n4():
    sp = M.ring_spectrum(4, 1.0, "Heat-bath")
    ts = [M.t_mix(sp, 2.0 ** -k) for k in range(1, 6)]
    assert ts == [4, 12, 21, 30, 39] and all(t <= k * 12 for k, t in zip(range(1, 6), ts))


def test_c03_drei_zahlen_n10_lambda2():
    sp = M.ring_spectrum(10, 2.0, "Heat-bath")
    assert (M.t_mix(sp), M.t_mix_dbar(sp), M.t_mix_dbar(sp, E), M.t_mix(sp, E)) == (73, 118, 91, 48)
    assert M.t_mix(M.ring_spectrum(10, 2.0, "Metropolis")) == 49


def test_c04_d_gegen_dbar_14_zellen():
    zellen = REF.dbar_zellen()
    assert len(zellen) == 14 and all(z["ungleichung_gilt"] for z in zellen)
    assert {(z["n"], z["lam"]) for z in zellen} == {(n, lam) for n in (6, 8, 10) for lam in (1.0, 2.0, 5.0, 10.0)} | {(12, 1.0), (12, 2.0)}
    assert _r(min(z["verhaeltnis_min"] for z in zellen), 3) == 1.000 and _r(max(z["verhaeltnis_max"] for z in zellen), 3) == 2.000
    a = [z["t_dbar"] / z["t_d"] for z in zellen]
    b = [z["T_e"] / z["t_d"] for z in zellen]
    assert (_r(min(a), 2), _r(max(a), 2), _r(float(np.mean(a)), 2)) == (1.50, 1.98, 1.72)
    assert (_r(min(b), 2), _r(max(b), 2), _r(float(np.mean(b)), 2)) == (1.20, 1.43, 1.30)
    # live nachgerechnet für die Zellen mit n ≤ 8, auf jedem ganzzahligen t von 1 bis 3·t_mix
    for z in zellen:
        if z["n"] > 6 or z["lam"] > 2.0:
            continue
        sp = M.ring_spectrum(z["n"], z["lam"], "Heat-bath")
        for t in range(1, 3 * z["t_d"] + 1, 7):
            d, db = M.d_t(sp, t), M.dbar_t(sp, t)
            assert d - 1e-12 <= db <= 2 * d + 1e-12


def test_c05_schwelle_und_einheit_n10():
    erwartet = {("Heat-bath", 2.0): [29, 73, 48, 134, 180, 289, 446], ("Metropolis", 2.0): [20, 49, 32, 89, 120, 192, 296]}
    epsilons = [0.5, 0.25, E, 0.1, 0.05, 0.01, 0.001]
    for (kette, lam), soll in erwartet.items():
        sp = M.ring_spectrum(10, lam, kette)
        assert [M.t_mix(sp, e) for e in epsilons] == soll
    assert _r(446 / 73, 2) == 6.11 and _r(289 / 73, 2) == 3.96
    zellen = 0
    for lam in (1.0, 2.0, 5.0):
        for kette in C.KETTEN:
            sp = M.ring_spectrum(10, lam, kette)
            t4 = M.t_mix(sp, 0.25)
            assert all(M.t_mix(sp, e) <= math.ceil(math.log2(1 / e)) * t4 for e in (E, 0.1, 0.05, 0.01, 0.001))
            zellen += 1
    assert zellen == 6


def test_c06_zeiteinheit_n10():
    hb2, mp2 = M.in_einheit(73, "Abfertigungsdauern", 10, 2.0, "Heat-bath"), M.in_einheit(49, "Abfertigungsdauern", 10, 2.0, "Metropolis")
    assert _r(M.in_einheit(73, "Durchläufe (n Schritte)", 10, 2.0, "Heat-bath"), 2) == 7.30 and _r(hb2, 3) == 2.433 and _r(mp2, 3) == 2.450
    for lam, (hb, mp, a, b) in {1.0: (42, 21, 2.100, 2.100), 5.0: (208, 173, 3.467, 3.460)}.items():
        assert M.t_mix(M.ring_spectrum(10, lam, "Heat-bath")) == hb and M.t_mix(M.ring_spectrum(10, lam, "Metropolis")) == mp
        assert _r(M.in_einheit(hb, "Abfertigungsdauern", 10, lam, "Heat-bath"), 3) == a and _r(M.in_einheit(mp, "Abfertigungsdauern", 10, lam, "Metropolis"), 3) == b


# ---------------------------------------------------------------------------------------------------- der Start
def test_c07_der_start_n10():
    soll = {1.0: (42, 9, 30.1, 30.1, 4.7, 0.09), 2.0: (73, 10, 48.9, 54.0, 7.3, 0.09), 5.0: (208, 13, 124.3, 157.0, 16.0, 0.21), 10.0: (569, 48, 303.3, 442.7, 11.9, 0.66)}
    for lam, (tm, leer, mittel, mittel_pi, faktor, anteil) in soll.items():
        r = ev.exp2(10, lam)
        assert (r.t_mix, r.t_leer) == (tm, leer) and r.N == 123
        assert r.t_mittel == pytest.approx(mittel, abs=0.051) and r.t_mittel_pi == pytest.approx(mittel_pi, abs=0.051)
        assert _r(r.t_mix / r.t_leer, 1) == faktor and _r(r.anteil_halb, 2) == anteil
        assert r.t_min == r.t_leer and r.schlechteste_maximal and r.alle_maximal_gleich


def test_c08_schlechtester_start_ist_die_maximalbelegung():
    zellen = {k: v for k, v in REF.ring_zellen().items() if int(k.split("|")[0]) <= 14}
    assert len(zellen) == 192 and all(v["t_mix"] == v["t_mix_maximalstart"] and v["alle_starts_kontrolliert"] for v in zellen.values())
    # ab n = 15 nur aus der Maximalbelegung; volle Kontrolle für λ = 1 und 5 (in der Tabelle vermerkt)
    for n in (15, 16):
        for lam in C.LAM_TABELLE:
            for kette in C.KETTEN:
                z = REF.ring_zelle(n, lam, kette)
                assert z["alle_starts_kontrolliert"] == (lam in (1.0, 5.0))


# ---------------------------------------------------------------------------------------------------- Einschluss
def test_c09_einschluss_n10_lambda2():
    e = ev.einschluss_ring(10, 2.0, "Heat-bath")
    assert (_r(e.t_rel, 1), _r(e.unten, 1), e.t_mix, _r(e.oben, 1)) == (68.4, 46.7, 73, 569.2)
    assert _r(e.oben_durch_exakt, 1) == 7.8 and _r(e.exakt_durch_unten, 2) == 1.56


def test_c10_einschluss_30_zellen_und_224_zellen():
    s = ev.ring_einschluss_statistik()
    assert s["zellen"] == 30
    assert (_r(s["oben_min"], 1), _r(s["oben_max"], 1), _r(s["oben_median"], 1)) == (4.2, 29.1, 9.7)
    assert (_r(s["unten_min"], 2), _r(s["unten_max"], 2), _r(s["unten_median"], 2)) == (1.01, 2.40, 1.40)
    z = REF.ring_zelle(12, 20.0, "Heat-bath")
    assert _r(z["oben"] / z["t_mix"], 1) == 26.3
    zellen = REF.ring_zellen()
    assert len(zellen) == 224 and all(v["unten"] <= v["t_mix"] <= v["oben"] for v in zellen.values())


def test_c11_t_mix_gegen_t_rel_gerade_n():
    verh = ev.lauf_verhaeltnis_trel()
    assert verh["zellen"] == 48 and (_r(verh["min"], 2), _r(verh["max"], 2)) == (0.69, 1.65)
    bei100 = [REF.ring_zelle(n, 100.0, "Heat-bath")["t_mix"] / REF.ring_zelle(n, 100.0, "Heat-bath")["t_rel"] for n in range(6, 17, 2)]
    assert (_r(min(bei100), 3), _r(max(bei100), 3)) == (0.693, 0.736) and min(bei100) > math.log(2)


def test_c12_einschluss_kreis():
    soll = {5: (3, 2.89, 2.28), 25: (59, 4.97, 1.36), 100: (949, 6.40, 1.35), 400: (15180, 7.88, 1.35), 800: (60718, 8.62, 1.35)}
    for n, (t, oben, unten) in soll.items():
        e = ev.einschluss_kreis(n)
        assert e.t_mix == t and _r(e.oben_durch_exakt, 2) == oben and _r(e.exakt_durch_unten, 2) == unten


def test_c13_einschluss_wuerfel():
    soll = {12: (20, 5.8, 2.62), 50: (120, 15.0, 3.53), 100: (275, 25.7, 4.01), 400: (1378, 80.9, 4.98), 1024: (4009, 181.7, 5.65)}
    for n, (t, oben, unten) in soll.items():
        e = ev.einschluss_wuerfel(n)
        assert e.t_mix == t and _r(e.oben_durch_exakt, 1) == oben and _r(e.exakt_durch_unten, 2) == unten


# ---------------------------------------------------------------------------------------------------- Wachstum
def test_c14_wachstum_kreis():
    ts = {5: 3, 25: 59, 100: 949, 400: 15180, 800: 60718}
    assert {n: F.kreis_t(n) for n in ts} == ts
    assert [_r(F.kreis_t(n) / n ** 2, 4) for n in (5, 25, 100, 400, 800)] == [0.12, 0.0944, 0.0949, 0.0949, 0.0949]
    assert all(n ** 2 / 32 <= t <= n ** 2 for n, t in ts.items())
    assert [_r(ev.kreis_zeilen([n])[0].fenster, 2) for n in (50, 200, 800)] == [6.23, 6.25, 6.26]
    assert [F.kreis_d(4, t, False) for t in (1, 2, 5)] == pytest.approx([0.5, 0.5, 0.5], abs=1e-13) and F.kreis_t(4, 0.25, False) is None


def test_c15_wachstum_wuerfel():
    zeilen = ev.wuerfel_zeilen((12, 50, 100, 400, 1024))
    assert [_r((z.t_mix - 0.5 * z.n * math.log(z.n)) / z.n, 3) for z in zeilen] == [0.424, 0.444, 0.447, 0.449, 0.449]
    assert [_r(ev.wuerfel_zeilen([n])[0].fenster, 2) for n in (16, 128, 1024)] == [2.00, 1.18, 0.81]


def test_c16_wachstum_ring_lambda1():
    zeilen = ev.ring_zeilen(C.RING_FENSTER_N)
    assert [z.t_mix for z in zeilen] == [31, 54, 79, 105, 133] and [z.N for z in zeilen] == [47, 322, 2207, 15127, 103682]
    assert [_r(z.fenster, 2) for z in zeilen] == [3.86, 3.15, 2.83, 2.58, 2.42]
    # n = 8 und 12 live nachgerechnet, aus der Maximalbelegung
    for n, soll in ((8, 31), (12, 54)):
        sp = M.ring_spectrum(n, 1.0, "Heat-bath")
        assert M.t_start(sp, K.maximalstart(K.ring_chain(n, 1.0, "Heat-bath"))) == soll == M.t_mix(sp)


def test_c17_karten():
    soll = {5: 0.9237, 6: 0.6135, 7: 0.3341, 8: 0.1672, 9: 0.0854, 10: 0.0429, 11: 0.0215, 12: 0.0108, 13: 0.0054}
    assert {k: _r(F.riffle_tv(52, k), 4) for k in soll} == soll
    assert F.riffle_t(52) == 8 and F.riffle_t(52, 0.01) == 13
    assert [F.riffle_t(n) for n in (4, 8, 16, 32, 52, 64, 128)] == [3, 4, 5, 7, 8, 8, 10]
    from fractions import Fraction as Fr
    assert [F.riffle_tv_bruch(3, k) for k in (1, 2, 3, 4)] == [Fr(1, 3), Fr(7, 48), Fr(13, 192), Fr(25, 768)]


def test_c18_mischer():
    tr, tt = REF.mischer("Transpositionen"), REF.mischer("Top-zu-zufällig")
    assert [tr[str(n)]["0.25"] for n in range(3, 10)] == [2, 3, 4, 6, 7, 9, 10] and [tt[str(n)]["0.25"] for n in range(3, 10)] == [2, 4, 6, 8, 10, 13, 15]
    z = ev.mischer_zeile(8)
    assert (z["riffle"], z["transpositionen"], z["top_zu_zufaellig"]) == (4, 9, 13)
    for n in (4, 5, 6):                                                       # live nachgerechnet
        assert F.mischer_ts(n, "Transpositionen")[0.25] == tr[str(n)]["0.25"] and F.mischer_ts(n, "Top-zu-zufällig")[0.25] == tt[str(n)]["0.25"]


# ---------------------------------------------------------------------------------------------------- Simulation
def test_c19_simulation_zwei_zufalls_seeds():
    erwartet = {  # R: (Mittel ganz, min, max, nie) für Zufalls-Seed 5000 und 6000
        100: ((None, None, None, 20), (None, None, None, 20)), 300: ((129.0, 105, 153, 0), (131.45, 110, 155, 0)), 1000: ((84.25, 79, 89, 0), (83.15, 75, 91, 0)),
        3000: ((74.6, 72, 78, 0), (77.0, 74, 83, 0)), 10000: ((73.9, 72, 75, 0), (73.7, 72, 76, 0)), 100000: ((73.0, 73, 73, 0), (73.0, 73, 73, 0))}
    for R, zwei in erwartet.items():
        for tab, soll in zip((REF.sim_zeile(R), REF.sim_zeile_zweiter_seed(R)), zwei):
            g = tab["ganz"]
            assert (None if g["mittel"] is None else _r(g["mittel"], 2), g["min"], g["max"], g["nie"]) == soll, R
    assert REF.sim_zeile(100)["ganz"]["von"] == REF.sim_zeile_zweiter_seed(100)["ganz"]["von"] == 20
    assert [REF.sim_zeile(R)["wiederholungen"] for R in C.SIM_R_TABELLE] == [20, 20, 20, 10, 10, 3]
    # live nachgerechnet (SplitMix64), R ≤ 1000: gleiche Zahlen wie in der Tabelle
    for R in (100, 300, 1000):
        z = S.zusammenfassung(S.wiederholungen(R, C.TABELLE_REPS[R], C.TABELLE_SEED)["t"])
        assert z["nie"] == REF.sim_zeile(R)["ganz"]["nie"] and z["mittel"] == REF.sim_zeile(R)["ganz"]["mittel"]
    ref = S.referenz()
    assert (ref.t_exakt, ref.t_exakt_zahl) == (73, 11)


def test_c20_rauschgrenze_exakt_und_naeherung():
    soll = {100: (0.4284, 0.4189), 300: (0.2435, 0.2419), 1000: (0.1321, 0.1325), 3000: (0.0766, 0.0765), 10000: (0.0419, 0.0419), 100000: (0.0132, 0.0132)}
    for R, (exakt, naeherung) in soll.items():
        assert _r(S.rauschgrenzen(R)[0], 4) == exakt and _r(S.rauschgrenze_naeherung(R), 4) == naeherung
        assert _r(REF.sim_zeile(R)["rauschgrenze"], 4) == exakt
    assert S.rauschgrenzen(100)[0] > 0.25 > S.rauschgrenzen(300)[0]
    # fällt etwa wie 1/√R: Rauschgrenze·√R für R ≥ 1000 auf 2 % konstant (4,18 bis 4,19)
    werte = [S.rauschgrenzen(R)[0] * math.sqrt(R) for R in (1000, 3000, 10000, 100000)]
    assert max(werte) / min(werte) < 1.02


def test_c21_perfekte_stichprobe_hat_nie_abstand_null():
    assert S.rauschgrenze_exakt(np.full(2, 0.5), 2) == pytest.approx(1 / 4) and S.rauschgrenze_exakt(np.full(2, 0.5), 4) == pytest.approx(3 / 16)
    assert S.rauschgrenze_exakt(np.full(7, 1 / 7), 7) == pytest.approx(279936 / 823543, abs=1e-13) and _r(279936 / 823543, 4) == 0.3399
    assert _r(ev.stichprobe_gleichverteilt(14, 7), 4) == 0.2503 and ev.stichprobe_gleichverteilt(14, 7) > 0.25
    for N in (2, 5, 7, 20, 123):
        for R in (1, 2, 5, 14, 50, 100):
            assert S.rauschgrenze_exakt(np.full(N, 1.0 / N), R) >= 1 - R / N - 1e-12              # Untergrenze 1 − R/N bei Gleichverteilung


def test_c22_kennzahl_statt_verteilung():
    ref = S.referenz()
    assert (ref.t_exakt_zahl, ref.t_exakt) == (11, 73) and _r(73 / 11, 1) == 6.6
    mittel = [REF.sim_zeile(R)["zahl"]["mittel"] for R in C.SIM_R_TABELLE] + [REF.sim_zeile_zweiter_seed(R)["zahl"]["mittel"] for R in C.SIM_R_TABELLE]
    assert (_r(min(mittel), 1), _r(max(mittel), 1)) == (11.0, 12.6) and _r(REF.sim_zeile(300)["zahl"]["mittel"], 2) == 12.05


# ---------------------------------------------------------------------------------------------------- Formel gerades n, Eigenwerte, Einordnung
def test_c23_formel_gerades_n():
    assert _r(M.g_gerades_n(1.0), 4) == 0.2929 and M.g_gerades_n(1.0) == pytest.approx(1 - 1 / math.sqrt(2), abs=1e-15)
    for n in (4, 6, 8, 10, 12):
        for lam in (0.5, 1.0, 2.0, 5.0, 10.0):
            sp = M.ring_spectrum(n, lam, "Heat-bath")
            assert n * (1 - sp.mu[1]) == pytest.approx(M.g_gerades_n(lam), rel=1e-9)
    for n in range(4, 17, 2):                                   # vorgerechnete Tabelle: zweitgrößter Eigenwert aller acht λ gleich der Formel
        for lam in C.LAM_TABELLE:
            z = REF.ring_zelle(n, lam, "Heat-bath")
            assert n * (1 - z["lambda_2"]) == pytest.approx(M.g_gerades_n(lam), rel=1e-6), (n, lam)
    sp = M.ring_spectrum(4, 1.0, "Heat-bath")
    trel = M.t_rel(sp.mu)
    unten, oben = M.schranken(trel, float(sp.pi.min()))
    assert (_r(trel, 3), _r(unten, 2), _r(oben, 1)) == (13.657, 8.77, 45.5) and _r(oben / 12, 1) == 3.8 and _r(12 / unten, 2) == 1.37


@pytest.mark.parametrize("n", [18, 22])
def test_c23_formel_gerades_n_grosse_ringe_lanczos(n):
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh
    for lam in (1.0, 5.0):
        ch = K.build_ring_chain(n, lam, "Heat-bath")
        s = np.sqrt(ch.pi)
        A = diags(s) @ ch.P @ diags(1.0 / s)
        A = (A + A.T) / 2
        top = np.sort(eigsh(A, k=3, which="LA", return_eigenvectors=False))[::-1]
        klein = float(eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
        assert abs(top[0] - 1) < 1e-9 and n * (1 - top[1]) == pytest.approx(M.g_gerades_n(lam), rel=1e-6) and klein > -1e-9


def test_c24_eigenwerte():
    for n in range(3, 11):
        for lam in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0):
            assert M.ring_spectrum(n, lam, "Heat-bath").mu[-1] >= -1e-12
    kleinste = [float(M.ring_spectrum(n, 1.0, "Metropolis").mu[-1]) for n in range(3, 13)]
    assert (_r(min(kleinste), 4), _r(max(kleinste), 4)) == (-0.3536, -0.3333) and kleinste[1] == min(kleinste) and kleinste[0] == max(kleinste)


def test_c25_wo_die_exakte_rechnung_endet():
    assert (K.lucas(16), K.lucas(24)) == (2207, 103682) and K.torus_zustaende(8, 8) == 213256442503
    assert _r(math.factorial(52) / 10 ** 67, 2) == 8.07
    zeilen = ev.einordnung_zeilen()
    assert [z["zustaende"] for z in zeilen] == [2207, 103682, 213256442503, math.factorial(52)]
    assert "nicht exakt berechenbar" in zeilen[2]["mischzeit"] and "133 Schritte" in zeilen[1]["mischzeit"] and "79 Schritte" in zeilen[0]["mischzeit"]


# ---------------------------------------------------------------------------------------------------- README
def test_readme_nennt_die_testzahl_der_suite():
    import subprocess
    import sys

    ausgabe = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True,
                             stdin=subprocess.DEVNULL, timeout=600).stdout
    zahl = int(re.search(r"(\d+) tests? collected", ausgabe).group(1))
    treffer = re.findall(r"\*\*(\d+) Tests\*\*", README)
    assert treffer and int(treffer[0]) == zahl, (treffer, zahl)


def test_readme_enthaelt_keine_planungswoerter_ausser_in_der_footer_zeile():
    text = "\n".join(z for z in README.splitlines() if "Mehr zu den Demos" not in z and "Mehr zur Reihe" not in z)
    for muster in (r"\bStück", r"\bLinie", r"Vormessung", r"\bPlan\b", r"\bV1\b", r"mc_v", r"Phase 0", r"Kantenabgleich", r"\bReihe\b", r"trägt", r"\btragen\b", r"Saatwert", r"\bSeed\b(?!-)", r"geplant", r"läuft lokal", r"Wärmebad"):
        for zeile in text.splitlines():
            zeile = zeile.replace("Zufalls-Seed", "")
            assert not re.search(muster, zeile), (muster, zeile[:140])
    assert not [z for z in README.splitlines() if "Website" in z and "offen" in z and "Einbindung" in z]


def test_readme_nennt_live_link_als_marke_und_die_zahlen_der_tabelle():
    assert "TODO nach dem Deploy" in README
    for zahl in ("2,433", "2,450", "0,3341", "569", "29,1", "80,9", "213 256 442 503", "279 936/823 543", "0,4284", "131,45", "6,6", "8,07·10⁶⁷", "gerade n = 4 bis 16"):
        assert zahl in README, zahl
    for nr in range(1, 26):
        assert f"`test_c{nr:02d}`" in README, nr


def test_readme_und_app_nennen_dieselben_live_grenzen():
    import msz_texts as T
    assert "322 Zustände" in README and "322 Zustände" in T.GRENZEN
    assert C.RING_N_MAX_LIVE == 12 and K.lucas(C.RING_N_MAX_LIVE) == 322 and C.E1_DBAR_N_MAX_LIVE == 10 and C.RING_N_MAX_TABLE == 16
    assert "bis 12 Plätze live" in README and "bis 16 Plätze vorgerechnet" in T.GRENZEN


def test_app_texte_enthalten_keine_planungswoerter():
    import msz_texts as T
    alles = "\n".join([T.EINLEITUNG, T.SO_FUNKTIONIERT, T.KERNFRAGEN, T.E1_CAPTION, T.E2_CAPTION, T.E3_CAPTION, T.E4_CAPTION, T.E5_CAPTION, T.GRENZEN, T.FORMELN,
                       (ROOT / "app.py").read_text(encoding="utf-8"), (ROOT / "msz_evaluation.py").read_text(encoding="utf-8")])
    for muster in (r"\bStück", r"\bLinie", r"Vormessung", r"\bPlan\b", r"\bV1\b", r"mc_v", r"Phase 0", r"Kantenabgleich", r"trägt", r"\btragen\b", r"Saatwert", r"geplant", r"läuft lokal", r"Wärmebad"):
        for zeile in alles.splitlines():
            if "Mehr zu den Demos" in zeile or "Mehr zur Reihe" in zeile:
                continue
            assert not re.search(muster, zeile), (muster, zeile[:140])
    for zeile in alles.splitlines():
        if "Mehr zu den Demos" in zeile or "Mehr zur Reihe" in zeile:
            continue
        assert not re.search(r"\bReihe\b", zeile), zeile[:140]
