"""Rechnet die vorgerechneten Tabellen der App neu und schreibt oder prüft `msz_reference.json` (Zahlen „vorgerechnet“ in App und README).

Aufruf (aus dem Repo-Ordner, Python des Test-venv):
    python tools/msz_make_tables.py             schreibt die ganze Datei neu (mehrere Minuten, ein schwerer Prozess)
    python tools/msz_make_tables.py --check     rechnet den schnellen Teil (n ≤ 12, R ≤ 3000, kleine Mischer) neu und vergleicht mit der Datei
    python tools/msz_make_tables.py --check --lang   vergleicht auch den langen Teil (n = 13 bis 16, d̄-Verhältnisse, R = 10 000 und 100 000, n = 16 bis 24)

Der lange Teil: Ring n = 13 bis 16 (dichte Eigenzerlegung bis 2207 Zustände; ab n = 15 nur aus dem Start in der Maximalbelegung, mit voller Kontrolle für
λ = 1 und 5), die Verhältnisse d̄/d auf dem ganzen Zeitraster, die Simulationstabelle für R = 10 000 und 100 000, die Fensterbreiten des Rings bis n = 24
(Vektoriteration). Zufall nur aus dem ganzzahligen SplitMix64 (msz_simulation.py), Zufalls-Seed der Tabelle 5000, zweiter Zufalls-Seed 6000 (Breite der Bänder)."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import msz_chain as K  # noqa: E402
import msz_constants as C  # noqa: E402
import msz_families as F  # noqa: E402
import msz_mixing as M  # noqa: E402
import msz_simulation as S  # noqa: E402

DATEI = ROOT / "msz_reference.json"
LAMS = C.LAM_TABELLE
N_SCHNELL = 12
EINSCHLUSS_KETTEN = ("Heat-bath", "Metropolis")


def _zelle(n: int, lam: float, kette: str) -> dict:
    """t_mix, t_rel und Schranken für eine Zelle.

    n ≤ 12: t_mix ist das Maximum über alle Starts (Halbierungssuche). n = 13, 14: t_mix aus dem Maximalstart, dazu eine volle Kontrolle (d(t_mix) über alle Starts
    ≤ 1/4; d(t_mix − 1) > 1/4 gilt schon im Maximalstart, die Zahl ist also t_mix über alle Starts). n = 15, 16: Maximalstart, volle Kontrolle nur für λ = 1 und 5."""
    sp = M.ring_spectrum(n, lam, kette)
    ch = K.ring_chain(n, lam, kette)
    t_ms = M.t_start(sp, K.maximalstart(ch))
    if n <= N_SCHNELL:
        t, alle = M.t_mix(sp), True
    else:
        t, alle = t_ms, n <= 14
        if n <= 14 or lam in (1.0, 5.0):
            assert M.d_t(sp, t_ms) <= 0.25 + 1e-12, ("Kontrolle verletzt", n, lam, kette)
            alle = True
    trel = M.t_rel(sp.mu)
    unten, oben = M.schranken(trel, float(sp.pi.min()))
    return {"N": int(sp.N), "t_mix": int(t), "t_mix_maximalstart": int(t_ms), "t_rel": trel, "unten": unten, "oben": oben,
            "alle_starts_kontrolliert": bool(alle), "lambda_2": float(sp.mu[1]), "lambda_min": float(sp.mu[-1])}


def ring_tabelle(n_von: int, n_bis: int) -> dict:
    out = {}
    for n in range(n_von, n_bis + 1):
        for lam in LAMS:
            for kette in EINSCHLUSS_KETTEN:
                out[f"{n}|{lam:g}|{kette}"] = _zelle(n, lam, kette)
        M.ring_spectrum.cache_clear()
        K.ring_chain.cache_clear()
        print("  Ring-Tabelle n =", n, flush=True)
    return out


def start_tabelle() -> dict:
    out = {}
    for lam in C.E2_LAMS:
        sp = M.ring_spectrum(10, lam, "Heat-bath")
        ch = K.ring_chain(10, lam, "Heat-bath")
        tx = M.t_alle_starts(sp)
        tm = int(tx.max())
        schlecht = ch.teilchen[tx == tm]
        out[f"{lam:g}"] = {
            "t_mix": tm, "t_leer": int(tx[ch.index(0)]), "t_min": int(tx.min()), "t_mittel": float(tx.mean()),
            "t_mittel_pi": float((tx * sp.pi).sum()), "anteil_halb": float((tx <= tm / 2).mean()),
            "schlechteste_sind_maximalbelegungen": bool((schlecht == 5).all()),
            "alle_maximalbelegungen_gleich": bool((tx[ch.teilchen == 5] == tm).all()),
            "zustaende": int(sp.N)}
    return out


A_ZELLEN = [(6, 1.0), (6, 2.0), (6, 5.0), (6, 10.0), (8, 1.0), (8, 2.0), (8, 5.0), (8, 10.0), (10, 1.0), (10, 2.0), (10, 5.0), (10, 10.0), (12, 1.0), (12, 2.0)]


def dbar_zelle(n: int, lam: float, verhaeltnis: bool) -> dict:
    sp = M.ring_spectrum(n, lam, "Heat-bath")
    e = 1.0 / math.e
    z = {"n": n, "lam": lam, "t_d": M.t_mix(sp, 0.25), "t_d_e": M.t_mix(sp, e), "t_dbar": M.t_mix_dbar(sp, 0.25), "T_e": M.t_mix_dbar(sp, e)}
    if verhaeltnis:
        # d ≤ d̄ ≤ 2d auf jedem ganzzahligen t von 1 bis 3·t_mix
        rmin, rmax, ok = 9.0, 0.0, True
        for t in range(1, 3 * z["t_d"] + 1):
            Pt = M.rows_at(sp, t)
            d = float(M.tv_rows(Pt, sp.pi).max())
            db = M.dbar_rows(Pt, sp.pi)
            ok &= bool(d <= db + 1e-12 and db <= 2 * d + 1e-12)
            rmin, rmax = min(rmin, db / d), max(rmax, db / d)
        z.update({"verhaeltnis_min": rmin, "verhaeltnis_max": rmax, "ungleichung_gilt": ok})
    return z


def dbar_tabelle(lang: bool) -> list:
    out = []
    for n, lam in A_ZELLEN:
        out.append(dbar_zelle(n, lam, verhaeltnis=lang or n <= 8))
        print("  d̄-Zelle", n, lam, flush=True)
    return out


def sim_tabelle(rs, seed: int = C.TABELLE_SEED) -> dict:
    out = {}
    for R in rs:
        reps = C.TABELLE_REPS[R]
        w = S.wiederholungen(R, reps, seed)
        rg = S.rauschgrenzen(R)
        out[str(R)] = {"wiederholungen": reps, "t": w["t"], "t_zahl": w["t_zahl"], "ganz": S.zusammenfassung(w["t"]),
                       "zahl": S.zusammenfassung(w["t_zahl"]), "rauschgrenze": rg[0], "rauschgrenze_zahl": rg[1]}
        print("  Simulation R =", R, flush=True)
    return out


def _kurve_ring(n: int, lam: float = 1.0, t_ende: int | None = None) -> tuple[dict, dict]:
    """Maximalstart, Heat-bath: t(ε) für ε = 0,9 / 0,5 / 1/4 / 0,1 und die Kurve d(t) auf einem Raster (Vektoriteration)."""
    ch = K.build_ring_chain(n, lam, "Heat-bath")
    x0 = K.maximalstart(ch)
    PT = ch.P.T.tocsr()
    p = np.zeros(ch.N)
    p[x0] = 1.0
    d = []
    while True:
        d.append(0.5 * np.abs(p - ch.pi).sum())
        if d[-1] <= 0.1 * 0.5 and len(d) > 5:
            break
        p = PT @ p
    d = np.array(d)
    ts = {}
    for e in (0.9, 0.5, 0.25, 0.1):
        ts[str(e)] = int(np.nonzero(d <= e)[0][0])
    raster = M.zeitraster(len(d) - 1, 60)
    return {"N": int(ch.N), "t": ts}, {"t": [int(t) for t in raster], "d": [float(d[t]) for t in raster]}


def ring_fenster(ns) -> dict:
    out = {}
    for n in ns:
        kopf, kurve = _kurve_ring(n)
        out[str(n)] = {**kopf, "kurve": kurve}
        print("  Fenster n =", n, flush=True)
    return out


def mischer(ns) -> dict:
    out = {}
    for art in ("Transpositionen", "Top-zu-zufällig"):
        out[art] = {}
        for n in ns:
            out[art][str(n)] = {str(e): t for e, t in F.mischer_ts(n, art).items()}
    return out


def karten() -> dict:
    return {"52": [F.riffle_tv(52, k) for k in range(0, C.KARTEN_K_MAX + 1)],
            "t_mix": {str(n): F.riffle_t(n) for n in C.KARTEN_N_TABELLE}}


def alles(lang: bool) -> dict:
    t0 = time.time()
    daten = {"hinweis": "Vorgerechnet von tools/msz_make_tables.py (Ring, Simulation, Mischer); nicht von Hand ändern. Zufalls-Seed der Simulation: %d." % C.TABELLE_SEED}
    daten["ring_tabelle"] = ring_tabelle(3, 16 if lang else N_SCHNELL)
    daten["start_tabelle"] = start_tabelle()
    daten["dbar_tabelle"] = dbar_tabelle(lang)
    daten["sim_tabelle"] = sim_tabelle(C.SIM_R_TABELLE if lang else [r for r in C.SIM_R_TABELLE if r <= 3000])
    daten["sim_tabelle_zweiter_seed"] = sim_tabelle(C.SIM_R_TABELLE if lang else [r for r in C.SIM_R_TABELLE if r <= 3000], C.TABELLE_SEED_2)
    daten["ring_fenster"] = ring_fenster(C.RING_FENSTER_N if lang else [n for n in C.RING_FENSTER_N if n <= 12])
    daten["mischer"] = mischer(C.MISCHER_N if lang else [n for n in C.MISCHER_N if n <= C.MISCHER_N_LIVE_MAX])
    daten["karten"] = karten()
    print("Fertig in %.0f s" % (time.time() - t0))
    return daten


def _gleich(a, b, pfad="") -> list:
    """Unterschiede zwischen zwei JSON-Bäumen (Gleitkommazahlen relativ 1e-9, Ganzzahlen und Texte exakt)."""
    fehler = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                fehler.append(f"{pfad}/{k}: fehlt auf einer Seite")
            else:
                fehler += _gleich(a[k], b[k], f"{pfad}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            fehler.append(f"{pfad}: Länge {len(a)} gegen {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                fehler += _gleich(x, y, f"{pfad}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            fehler.append(f"{pfad}: {a} gegen {b}")
    elif a != b:
        fehler.append(f"{pfad}: {a} gegen {b}")
    return fehler


def pruefen(lang: bool) -> int:
    alt = json.loads(DATEI.read_text(encoding="utf-8"))
    neu = alles(lang)
    fehler = []
    for abschnitt, wert in neu.items():
        if abschnitt == "hinweis":
            continue
        if abschnitt == "mischer":
            for art, nach_n in wert.items():                      # zwei Ebenen: Verfahren, dann Kartenzahl; nur die neu gerechneten n vergleichen
                for n, v in nach_n.items():
                    fehler += _gleich(v, alt[abschnitt][art].get(n), f"{abschnitt}/{art}/{n}")
        elif abschnitt in ("ring_tabelle", "ring_fenster", "sim_tabelle", "sim_tabelle_zweiter_seed"):
            # nur die neu gerechneten Schlüssel vergleichen
            for k, v in wert.items():
                if k not in alt[abschnitt]:
                    fehler.append(f"{abschnitt}/{k}: fehlt in der Datei")
                else:
                    fehler += _gleich(v, alt[abschnitt][k], f"{abschnitt}/{k}")
        elif abschnitt == "dbar_tabelle":
            for z_neu, z_alt in zip(wert, alt[abschnitt]):
                z_alt_teil = {k: z_alt[k] for k in z_neu if k in z_alt}
                fehler += _gleich(z_neu, z_alt_teil, f"dbar/{z_neu['n']}|{z_neu['lam']}")
        else:
            fehler += _gleich(wert, alt[abschnitt], abschnitt)
    if fehler:
        print("UNTERSCHIEDE (%d):" % len(fehler))
        for f in fehler[:40]:
            print("  ", f)
        return 1
    print("Datei stimmt mit der Neurechnung überein (%s)." % ("voll" if lang else "schneller Teil"))
    return 0


def main() -> int:
    args = sys.argv[1:]
    lang = "--lang" in args
    if "--nur-zweiter-seed" in args:
        # ergänzt nur die Tabelle des zweiten Zufalls-Seeds in der vorhandenen Datei
        daten = json.loads(DATEI.read_text(encoding="utf-8"))
        daten["sim_tabelle_zweiter_seed"] = sim_tabelle(C.SIM_R_TABELLE, C.TABELLE_SEED_2)
        DATEI.write_text(json.dumps(daten, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print("zweiter Zufalls-Seed ergänzt:", DATEI)
        return 0
    if "--check" in args:
        return pruefen(lang)
    daten = alles(True)
    DATEI.write_text(json.dumps(daten, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben:", DATEI)
    return 0


if __name__ == "__main__":
    sys.exit(main())
